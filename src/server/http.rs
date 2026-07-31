//! A minimal, dependency-free HTTP/1.1 server (`std::net`, thread-per-request).
//!
//! `dlm`'s OpenAI-compatible API needs an HTTP
//! surface. Rather than pull in an async stack, this is a small blocking server:
//! it parses a request, dispatches to a handler closure, and writes the
//! response. It is sufficient for the local, single-node serving `dlm` targets
//! and keeps the whole engine buildable and testable with no extra dependencies.

use crate::error::{DlmError, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// A parsed HTTP request.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    /// Request path with any query string removed, e.g. `/v1/models`.
    ///
    /// Routing and the public-path check both compare this by equality, so the
    /// query must not be part of it. It used to be the raw request target, which
    /// meant `/health?probe=1` was not recognised as the health route — it failed
    /// the auth exemption and 401'd, and `/v1/models?x=1` 404'd. Every consumer
    /// wants the path; the one that wants parameters can have [`Request::query`].
    pub path: String,
    /// Raw query string with no leading `?`, if the target had one.
    pub query: Option<String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    /// Body interpreted as UTF-8 (lossy).
    pub fn body_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }
}

/// Split a raw request target into its path and query halves.
///
/// A `?` with nothing after it yields `Some("")` rather than `None`: the client
/// did send a query delimiter, and flattening that to "no query" loses the
/// distinction. A fragment (`#…`) is never sent by a conforming client and is
/// dropped with the query if one appears.
fn split_target(target: &str) -> (String, Option<String>) {
    match target.split_once('?') {
        Some((path, query)) => (path.to_string(), Some(query.to_string())),
        None => (target.to_string(), None),
    }
}

/// Writes a streaming body straight to the socket (SSE), returning when done.
pub type StreamWriter = Box<dyn FnOnce(&mut dyn Write) -> std::io::Result<()> + Send>;

/// A response body: either a complete buffer or a streaming writer (for SSE).
pub enum Body {
    /// A complete, length-known body.
    Full(Vec<u8>),
    /// A streaming body: the closure writes directly to the socket and returns
    /// when the stream is complete. Sent without `Content-Length`.
    Stream(StreamWriter),
}

/// An HTTP response.
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Body,
    /// Extra headers emitted after the standard set, as `(name, value)`.
    ///
    /// Empty for almost every response. It exists because a few statuses are only
    /// actionable with one — `503` and `429` both need `Retry-After` to tell a
    /// client the difference between "come back shortly" and "give up".
    pub extra_headers: Vec<(&'static str, String)>,
}

impl Response {
    /// A JSON response.
    pub fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: "application/json".to_string(),
            body: Body::Full(body.into()),
            extra_headers: Vec::new(),
        }
    }

    /// A plain-text response.
    pub fn text(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: "text/plain; charset=utf-8".to_string(),
            body: Body::Full(body.into()),
            extra_headers: Vec::new(),
        }
    }

    /// A streaming response — `write` is invoked with the socket to emit the body
    /// incrementally (e.g. Server-Sent Events).
    pub fn stream<F>(status: u16, content_type: impl Into<String>, write: F) -> Self
    where
        F: FnOnce(&mut dyn Write) -> std::io::Result<()> + Send + 'static,
    {
        Self {
            status,
            content_type: content_type.into(),
            body: Body::Stream(Box::new(write)),
            extra_headers: Vec::new(),
        }
    }

    /// Attach an extra header, builder-style.
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.extra_headers.push((name, value.into()));
        self
    }
}

/// Maximum accepted request-body size (16 MiB). A larger `Content-Length` is
/// rejected before allocating, so a malicious header can't exhaust memory.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Maximum length of the request line or any single header line (8 KiB).
///
/// `BufReader::read_line` grows its `String` until it sees a newline, so without
/// this a client that streams endless bytes and never sends `\n` drives unbounded
/// allocation and OOMs the whole process — not just its own connection.
const MAX_LINE_BYTES: u64 = 8 * 1024;

/// Maximum number of header lines accepted on one request.
const MAX_HEADERS: usize = 100;

/// How long a connection may stall without progress before it is dropped.
///
/// Without a read timeout, a client that opens a socket and sends nothing (or
/// promises a body it never delivers) pins a thread forever — the classic
/// slowloris. Each stalled connection costs a thread, so a handful of idle
/// sockets would otherwise take the server down.
const IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Maximum number of connections served concurrently.
///
/// The accept loop previously spawned an unbounded thread per connection. Past
/// the cap a connection is answered with `503 Service Unavailable` and closed,
/// rather than accepted: shedding with a status code, so an overloaded server is
/// distinguishable from a crashed one. It does *not* wait for a slot — queueing
/// here would convert a fast rejection into an unbounded backlog.
const MAX_CONNECTIONS: usize = 256;

/// How long a shed client is asked to wait before retrying, in seconds.
const RETRY_AFTER_SECS: u32 = 1;

/// How long in-flight requests are given to finish after shutdown is triggered.
///
/// Shorter than the grace period supervisors allow before `SIGKILL` (Docker and
/// Kubernetes both default to 10s), so the drain completes on its own terms
/// rather than being killed halfway.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(8);

/// A trigger that stops a running [`HttpServer`] and drains it.
///
/// Cloneable and cheap: hand one to a signal handler and keep another for the
/// server. Triggering is idempotent.
#[derive(Clone)]
pub struct Shutdown {
    flag: Arc<std::sync::atomic::AtomicBool>,
    /// The address the server is listening on, published once it is serving.
    ///
    /// `TcpListener::accept` blocks, and there is no portable way to interrupt
    /// it. Setting the flag alone would not be noticed until the *next*
    /// connection arrived, which on an idle server is never. So the trigger also
    /// dials the listener itself: the loop wakes, sees the flag, and breaks.
    addr: Arc<std::sync::Mutex<Option<SocketAddr>>>,
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            addr: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// True once [`trigger`](Self::trigger) has been called.
    pub fn is_triggered(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    /// Stop the server: set the flag, then wake the blocked `accept`.
    pub fn trigger(&self) {
        self.flag.store(true, Ordering::SeqCst);
        // Best-effort wake. If the connection fails the server is already down,
        // or it is mid-accept and will observe the flag on its next pass.
        if let Some(addr) = *self.addr.lock().unwrap() {
            let _ = TcpStream::connect_timeout(&addr, Duration::from_millis(500));
        }
    }

    /// Record where the server is listening, so `trigger` knows what to dial.
    fn bind_addr(&self, addr: SocketAddr) {
        *self.addr.lock().unwrap() = Some(addr);
    }
}

/// Decrements the live-connection counter on drop.
///
/// The counter is incremented before the connection thread is spawned and must be
/// returned however that thread ends. It previously decremented on the last line
/// of the closure, which a panic in the handler unwinds straight past — leaking a
/// slot permanently. After [`MAX_CONNECTIONS`] such panics the server shed every
/// connection forever, with no log line and a process that still looked healthy.
/// `Drop` runs during unwinding, so the slot comes back either way.
struct ConnectionSlot(Arc<AtomicUsize>);

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

/// A request handler: maps a request to a response.
pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// A bound HTTP server, ready to serve.
pub struct HttpServer {
    listener: TcpListener,
}

impl HttpServer {
    /// Bind to `addr` (e.g. `"127.0.0.1:8000"`, or `":0"` for an ephemeral port).
    pub fn bind(addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(addr).map_err(|source| DlmError::Io {
            path: addr.into(),
            source,
        })?;
        Ok(Self { listener })
    }

    /// The actual bound address (useful when binding to port 0).
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(|source| DlmError::Io {
            path: "<socket>".into(),
            source,
        })
    }

    /// Serve forever, dispatching each connection to `handler` on its own thread.
    ///
    /// Concurrency is capped at [`MAX_CONNECTIONS`]. Past the cap the connection is
    /// answered with `503 Service Unavailable` and a `Retry-After`, then closed —
    /// shedding, not queueing, and not a silent TCP reset.
    ///
    /// Runs until the process ends. To stop cleanly, use
    /// [`serve_with_shutdown`](Self::serve_with_shutdown).
    pub fn serve(self, handler: Handler) -> Result<()> {
        self.serve_with_shutdown(handler, Shutdown::new())
    }

    /// Serve until `shutdown` is triggered, then drain in-flight requests.
    ///
    /// `SIGTERM` is how every supervisor stops a process — `docker stop`,
    /// systemd, a Kubernetes pod eviction — and each follows it with `SIGKILL`
    /// after a grace period. Without this the server had no stop path at all: the
    /// accept loop ran until the process died, cutting in-flight generations and
    /// leaving SSE streams without their terminating chunk, so every deploy
    /// dropped live requests.
    ///
    /// On trigger the listener stops accepting and existing connections are given
    /// up to [`DRAIN_TIMEOUT`] to finish. Requests already being served run to
    /// completion; new ones are refused by closing the socket, because a client
    /// that reconnects to a load balancer will be routed to a live instance.
    pub fn serve_with_shutdown(self, handler: Handler, shutdown: Shutdown) -> Result<()> {
        let live = Arc::new(AtomicUsize::new(0));
        // The waker connects to this to break `accept`'s block.
        shutdown.bind_addr(self.local_addr()?);

        for stream in self.listener.incoming() {
            if shutdown.is_triggered() {
                break;
            }
            let Ok(mut stream) = stream else { continue };

            // Over the cap: answer with a status the client can act on. A bare
            // `drop` here is indistinguishable from the server having crashed,
            // which sends callers into retry loops against a live process.
            if live.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
                let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
                let _ = write_response(&mut stream, overloaded_response());
                continue;
            }

            // Claim the slot before spawning; `ConnectionSlot` returns it on drop,
            // including while unwinding from a panic inside the handler.
            live.fetch_add(1, Ordering::Relaxed);
            let slot = ConnectionSlot(Arc::clone(&live));

            let handler = Arc::clone(&handler);
            std::thread::spawn(move || {
                let _slot = slot;
                if let Err(_e) = handle_connection(stream, handler) {
                    // Per-connection errors are non-fatal; drop the connection.
                }
            });
        }

        // Drain. Requests already in flight keep their slot until they finish, so
        // waiting for the counter to reach zero is waiting for them.
        let deadline = std::time::Instant::now() + DRAIN_TIMEOUT;
        while live.load(Ordering::Relaxed) > 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        Ok(())
    }
}

/// Read one CRLF-terminated line, refusing to buffer more than [`MAX_LINE_BYTES`].
///
/// Returns `Ok(None)` at clean EOF. An over-long line is an error (the connection
/// is dropped) rather than an unbounded allocation.
fn read_line_capped<R: BufRead>(reader: &mut R, buf: &mut String) -> std::io::Result<Option<()>> {
    buf.clear();
    let n = reader.take(MAX_LINE_BYTES).read_line(buf)?;
    if n == 0 {
        return Ok(None);
    }
    if n as u64 >= MAX_LINE_BYTES && !buf.ends_with('\n') {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "header line exceeds limit",
        ));
    }
    Ok(Some(()))
}

/// Parse one request, run the handler, write one response, close.
fn handle_connection(stream: TcpStream, handler: Handler) -> std::io::Result<()> {
    // A stalled peer must not pin this thread forever.
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;

    let mut reader = BufReader::new(stream);

    // Request line: METHOD PATH VERSION
    let mut request_line = String::new();
    if read_line_capped(&mut reader, &mut request_line)?.is_none() {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let (path, query) = split_target(parts.next().unwrap_or("/"));

    // Headers until a blank line.
    let mut headers = HashMap::new();
    let mut line = String::new();
    loop {
        if read_line_capped(&mut reader, &mut line)?.is_none() {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if headers.len() >= MAX_HEADERS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "too many headers",
            ));
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }

    // dlm speaks HTTP/1.0-style single-shot requests and reads bodies by
    // Content-Length. A chunked body would otherwise be silently read as empty
    // and fail JSON parsing with a misleading 400 — say what is actually wrong.
    if let Some(te) = headers.get("transfer-encoding") {
        if te.to_ascii_lowercase().contains("chunked") {
            return write_response(
                reader.get_mut(),
                Response::json(411, br#"{"error":{"message":"chunked transfer-encoding is not supported; send Content-Length","type":"invalid_request_error"}}"#.to_vec()),
            );
        }
    }

    // Body per Content-Length, capped to guard against a hostile header.
    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return write_response(
            reader.get_mut(),
            Response::json(
                413,
                br#"{"error":{"message":"request body too large","type":"invalid_request_error"}}"#
                    .to_vec(),
            ),
        );
    }
    // Read incrementally instead of pre-allocating the *declared* length: a
    // client could otherwise claim 16 MiB, send one byte, and hold that much
    // memory (times every open connection) without ever completing the request.
    let mut body = Vec::new();
    if content_length > 0 {
        reader
            .by_ref()
            .take(content_length as u64)
            .read_to_end(&mut body)?;
        if body.len() != content_length {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "body shorter than Content-Length",
            ));
        }
    }

    let request = Request {
        method,
        path,
        query,
        headers,
        body,
    };
    let response = handler(&request);
    write_response(reader.get_mut(), response)
}

/// The response sent when [`MAX_CONNECTIONS`] is already reached.
///
/// Built without touching the router: at this point no request has been read, so
/// there is nothing to dispatch. The client gets a status instead of a reset.
fn overloaded_response() -> Response {
    Response::json(
        503,
        br#"{"error":{"message":"server at capacity; retry shortly","type":"overloaded_error"}}"#
            .to_vec(),
    )
    .with_header("Retry-After", RETRY_AFTER_SECS.to_string())
}

/// Render `extra_headers` as trailing `Name: value\r\n` lines.
fn extra_header_lines(extra: &[(&'static str, String)]) -> String {
    extra.iter().map(|(k, v)| format!("{k}: {v}\r\n")).collect()
}

fn write_response(stream: &mut TcpStream, response: Response) -> std::io::Result<()> {
    let extra = extra_header_lines(&response.extra_headers);
    match response.body {
        Body::Full(bytes) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
                response.status,
                reason(response.status),
                response.content_type,
                bytes.len(),
                extra,
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(&bytes)?;
        }
        Body::Stream(write) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n{}\r\n",
                response.status,
                reason(response.status),
                response.content_type,
                extra,
            );
            stream.write_all(head.as_bytes())?;
            write(stream)?;
        }
    }
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal HTTP client for tests: send a raw request, return the raw response.
    fn round_trip(addr: SocketAddr, raw: &str) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        stream.write_all(raw.as_bytes()).unwrap();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).unwrap();
        resp
    }

    #[test]
    fn serves_a_request_and_echoes_body() {
        let server = HttpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let handler: Handler = Arc::new(|req: &Request| {
            Response::json(
                200,
                format!(r#"{{"path":"{}","echo":{}}}"#, req.path, req.body_str()),
            )
        });
        std::thread::spawn(move || server.serve(handler).unwrap());

        let raw = "POST /v1/x HTTP/1.1\r\nContent-Length: 4\r\n\r\ntrue";
        let resp = round_trip(addr, raw);
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
        assert!(resp.contains(r#""path":"/v1/x""#), "{resp}");
        assert!(resp.contains(r#""echo":true"#), "{resp}");
    }

    #[test]
    fn parses_headers_case_insensitively() {
        let server = HttpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let handler: Handler = Arc::new(|req: &Request| {
            let ct = req.headers.get("content-type").cloned().unwrap_or_default();
            Response::text(200, ct)
        });
        std::thread::spawn(move || server.serve(handler).unwrap());

        let raw = "GET / HTTP/1.1\r\nContent-Type: application/json\r\n\r\n";
        let resp = round_trip(addr, raw);
        assert!(resp.contains("application/json"), "{resp}");
    }

    /// Start a server on an ephemeral port and return its address.
    fn serve_with(handler: Handler) -> SocketAddr {
        let server = HttpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        std::thread::spawn(move || server.serve(handler).unwrap());
        addr
    }

    /// Echoes the routed path and the query, so tests can see the split.
    fn echo_target() -> Handler {
        Arc::new(|req: &Request| {
            Response::text(
                200,
                format!("{}|{}", req.path, req.query.as_deref().unwrap_or("-")),
            )
        })
    }

    #[test]
    fn splits_query_string_off_the_path() {
        assert_eq!(split_target("/v1/models"), ("/v1/models".into(), None));
        assert_eq!(
            split_target("/health?probe=1"),
            ("/health".into(), Some("probe=1".into()))
        );
        // A bare `?` is a query the client sent, not the absence of one.
        assert_eq!(split_target("/x?"), ("/x".into(), Some(String::new())));
        // Only the first `?` delimits; the rest is query data.
        assert_eq!(
            split_target("/x?a=1?2"),
            ("/x".into(), Some("a=1?2".into()))
        );
    }

    /// `A4`: routing compares `path` by equality, so a query string used to make
    /// `/health` unrecognisable -- it failed the auth exemption and 401'd, while
    /// `/v1/models?x=1` 404'd.
    #[test]
    fn query_string_does_not_change_the_routed_path() {
        let addr = serve_with(echo_target());
        let resp = round_trip(addr, "GET /health?probe=1 HTTP/1.1\r\n\r\n");
        assert!(resp.contains("/health|probe=1"), "{resp}");

        let resp = round_trip(addr, "GET /health HTTP/1.1\r\n\r\n");
        assert!(resp.contains("/health|-"), "{resp}");
    }

    /// `A1`: the live-connection counter is claimed before the thread spawns and
    /// returned by `ConnectionSlot::drop`. It previously decremented on the last
    /// line of the closure, which a panicking handler unwinds straight past, so
    /// every panic leaked a slot permanently -- and after `MAX_CONNECTIONS` of
    /// them the server shed every subsequent connection, silently and forever.
    ///
    /// Drives strictly more panics than the cap, then asserts the server still
    /// serves. Without the guard this hangs or 503s at request 257.
    #[test]
    fn panicking_handler_does_not_leak_connection_slots() {
        // A panicking handler is the point of the test; keep the backtrace spam
        // out of the test output. Restored before the assertions.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let boom: Handler = Arc::new(|req: &Request| {
            if req.path == "/boom" {
                panic!("handler exploded");
            }
            Response::text(200, "alive")
        });
        let addr = serve_with(boom);

        for _ in 0..(MAX_CONNECTIONS + 8) {
            // The panic kills the connection thread before a response is written,
            // so the read fails -- that is expected and is not what we assert on.
            if let Ok(mut s) = TcpStream::connect(addr) {
                let _ = s.write_all(b"GET /boom HTTP/1.1\r\n\r\n");
                let mut sink = String::new();
                let _ = s.read_to_string(&mut sink);
            }
        }

        std::panic::set_hook(previous);

        // Every slot must have come back.
        let resp = round_trip(addr, "GET /ok HTTP/1.1\r\n\r\n");
        assert!(
            resp.starts_with("HTTP/1.1 200 OK"),
            "server stopped serving after {} panics -- slots leaked: {resp}",
            MAX_CONNECTIONS + 8
        );
        assert!(resp.contains("alive"), "{resp}");
    }

    /// `A3`: past the cap the server answers `503` with `Retry-After` rather than
    /// dropping the socket. A bare drop is indistinguishable from a crash, which
    /// sends clients into retry loops against a live process.
    #[test]
    fn over_capacity_gets_503_with_retry_after() {
        let resp = overloaded_response();
        assert_eq!(resp.status, 503);
        assert_eq!(reason(503), "Service Unavailable");
        assert!(
            resp.extra_headers
                .iter()
                .any(|(k, v)| *k == "Retry-After" && !v.is_empty()),
            "503 must carry Retry-After"
        );
    }

    #[test]
    fn extra_headers_are_emitted_after_the_standard_ones() {
        let rendered = extra_header_lines(&[("Retry-After", "1".into())]);
        assert_eq!(rendered, "Retry-After: 1\r\n");
        assert_eq!(extra_header_lines(&[]), "");
    }

    // ---- `A9`: the hardening constants, none of which had a test ----

    /// `A7`: triggering shutdown stops the accept loop on an *idle* server.
    ///
    /// The flag alone is not enough — `accept` blocks, so an idle server would
    /// never notice it. `trigger` therefore dials the listener to wake it. Without
    /// that this test hangs, which is the whole point of it.
    #[test]
    fn shutdown_stops_an_idle_server() {
        let server = HttpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let shutdown = Shutdown::new();
        let handler: Handler = Arc::new(|_: &Request| Response::text(200, "ok"));

        let s = shutdown.clone();
        let joined = std::thread::spawn(move || server.serve_with_shutdown(handler, s).unwrap());

        // Serve one request so we know it is up, then stop it.
        assert!(round_trip(addr, "GET /x HTTP/1.1\r\n\r\n").starts_with("HTTP/1.1 200"));
        assert!(!shutdown.is_triggered());
        shutdown.trigger();
        assert!(shutdown.is_triggered());

        // `serve_with_shutdown` must return rather than block forever.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while !joined.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            joined.is_finished(),
            "serve_with_shutdown did not return after trigger"
        );
        joined.join().unwrap();
    }

    /// `A7`: a request already being served runs to completion during the drain.
    ///
    /// Shutdown that cut in-flight work would be a crash with extra steps; the
    /// point is that `docker stop` stops dropping live requests.
    #[test]
    fn shutdown_drains_a_request_already_in_flight() {
        let server = HttpServer::bind("127.0.0.1:0").unwrap();
        let addr = server.local_addr().unwrap();
        let shutdown = Shutdown::new();

        // Handler is slow enough that shutdown lands while it is still running.
        let handler: Handler = Arc::new(|_: &Request| {
            std::thread::sleep(Duration::from_millis(300));
            Response::text(200, "finished")
        });

        let s = shutdown.clone();
        std::thread::spawn(move || server.serve_with_shutdown(handler, s).unwrap());

        let client = std::thread::spawn(move || round_trip(addr, "GET /slow HTTP/1.1\r\n\r\n"));

        // Trigger while the handler is mid-flight.
        std::thread::sleep(Duration::from_millis(60));
        shutdown.trigger();

        let resp = client.join().unwrap();
        assert!(
            resp.starts_with("HTTP/1.1 200"),
            "in-flight request was cut: {resp}"
        );
        assert!(resp.contains("finished"), "{resp}");
    }

    /// `MAX_LINE_BYTES`: a client that streams bytes and never sends `\n` must not
    /// drive unbounded allocation in `read_line`.
    #[test]
    fn over_long_request_line_is_refused_not_buffered() {
        let addr = serve_with(echo_target());
        let mut s = TcpStream::connect(addr).unwrap();
        let huge = "x".repeat(MAX_LINE_BYTES as usize * 2);
        let _ = s.write_all(format!("GET /{huge} HTTP/1.1\r\n\r\n").as_bytes());
        let mut resp = String::new();
        let _ = s.read_to_string(&mut resp);
        // The connection is dropped rather than served.
        assert!(!resp.starts_with("HTTP/1.1 200"), "{resp:?}");
    }

    /// `MAX_HEADERS`: a header flood is refused.
    #[test]
    fn too_many_headers_is_refused() {
        let addr = serve_with(echo_target());
        let mut raw = String::from("GET /x HTTP/1.1\r\n");
        for i in 0..(MAX_HEADERS + 10) {
            raw.push_str(&format!("X-Pad-{i}: v\r\n"));
        }
        raw.push_str("\r\n");
        let mut s = TcpStream::connect(addr).unwrap();
        let _ = s.write_all(raw.as_bytes());
        let mut resp = String::new();
        let _ = s.read_to_string(&mut resp);
        assert!(!resp.starts_with("HTTP/1.1 200"), "{resp:?}");
    }

    /// Chunked bodies are refused with `411` and a message naming the cause,
    /// rather than being read as empty and failing JSON parsing with a confusing
    /// `400`.
    #[test]
    fn chunked_transfer_encoding_is_refused_with_411() {
        let addr = serve_with(echo_target());
        let raw = "POST /v1/x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n";
        let resp = round_trip(addr, raw);
        assert!(resp.starts_with("HTTP/1.1 411 Length Required"), "{resp}");
        assert!(resp.contains("chunked"), "{resp}");
    }

    /// `MAX_BODY_BYTES`: a hostile `Content-Length` is rejected before allocating.
    #[test]
    fn oversized_content_length_is_rejected_before_reading() {
        let addr = serve_with(echo_target());
        let raw = format!(
            "POST /v1/x HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY_BYTES + 1
        );
        let resp = round_trip(addr, &raw);
        assert!(resp.starts_with("HTTP/1.1 413 Payload Too Large"), "{resp}");
    }

    /// A body shorter than its declared `Content-Length` is an error, not a
    /// silently truncated request.
    #[test]
    fn body_shorter_than_content_length_is_an_error() {
        let addr = serve_with(echo_target());
        let mut s = TcpStream::connect(addr).unwrap();
        let _ = s.write_all(b"POST /x HTTP/1.1\r\nContent-Length: 100\r\n\r\nshort");
        s.shutdown(std::net::Shutdown::Write).unwrap();
        let mut resp = String::new();
        let _ = s.read_to_string(&mut resp);
        assert!(!resp.starts_with("HTTP/1.1 200"), "{resp:?}");
    }
}
