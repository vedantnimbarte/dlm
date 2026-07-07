//! A minimal, dependency-free HTTP/1.1 server (`std::net`, thread-per-request).
//!
//! `dlm`'s OpenAI-compatible API (`specs.md` §3.4 / `PRD.md` §3.1) needs an HTTP
//! surface. Rather than pull in an async stack, this is a small blocking server:
//! it parses a request, dispatches to a handler closure, and writes the
//! response. It is sufficient for the local, single-node serving `dlm` targets
//! and keeps the whole engine buildable and testable with no extra dependencies.

use crate::error::{DlmError, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;

/// A parsed HTTP request.
#[derive(Debug, Clone)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    /// Body interpreted as UTF-8 (lossy).
    pub fn body_str(&self) -> std::borrow::Cow<'_, str> {
        String::from_utf8_lossy(&self.body)
    }
}

/// A response body: either a complete buffer or a streaming writer (for SSE).
pub enum Body {
    /// A complete, length-known body.
    Full(Vec<u8>),
    /// A streaming body: the closure writes directly to the socket and returns
    /// when the stream is complete. Sent without `Content-Length`.
    Stream(Box<dyn FnOnce(&mut dyn Write) -> std::io::Result<()> + Send>),
}

/// An HTTP response.
pub struct Response {
    pub status: u16,
    pub content_type: String,
    pub body: Body,
}

impl Response {
    /// A JSON response.
    pub fn json(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: "application/json".to_string(),
            body: Body::Full(body.into()),
        }
    }

    /// A plain-text response.
    pub fn text(status: u16, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type: "text/plain; charset=utf-8".to_string(),
            body: Body::Full(body.into()),
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
        }
    }
}

/// Maximum accepted request-body size (16 MiB). A larger `Content-Length` is
/// rejected before allocating, so a malicious header can't exhaust memory.
pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
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
    pub fn serve(self, handler: Handler) -> Result<()> {
        for stream in self.listener.incoming() {
            let Ok(stream) = stream else { continue };
            let handler = Arc::clone(&handler);
            std::thread::spawn(move || {
                if let Err(_e) = handle_connection(stream, handler) {
                    // Per-connection errors are non-fatal; drop the connection.
                }
            });
        }
        Ok(())
    }
}

/// Parse one request, run the handler, write one response, close.
fn handle_connection(stream: TcpStream, handler: Handler) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream);

    // Request line: METHOD PATH VERSION
    let mut request_line = String::new();
    if reader.read_line(&mut request_line)? == 0 {
        return Ok(());
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    // Headers until a blank line.
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
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
            Response::json(413, br#"{"error":{"message":"request body too large","type":"invalid_request_error"}}"#.to_vec()),
        );
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    let request = Request {
        method,
        path,
        headers,
        body,
    };
    let response = handler(&request);
    write_response(reader.get_mut(), response)
}

fn write_response(stream: &mut TcpStream, response: Response) -> std::io::Result<()> {
    match response.body {
        Body::Full(bytes) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                response.status,
                reason(response.status),
                response.content_type,
                bytes.len(),
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(&bytes)?;
        }
        Body::Stream(write) => {
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
                response.status,
                reason(response.status),
                response.content_type,
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
            Response::json(200, format!(r#"{{"path":"{}","echo":{}}}"#, req.path, req.body_str()))
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
}
