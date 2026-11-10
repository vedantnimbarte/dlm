//! Constrained decoding: GBNF grammars, the matcher that decides whether a
//! token can come next, and JSON Schema → GBNF.
//!
//! A client that asks for JSON wants JSON every time, not usually. Prompting
//! gets "usually"; masking the tokens the grammar cannot take gets "every time".
//!
//! The grammar language is llama.cpp's GBNF, so grammars written for it work
//! here unchanged: `name ::= ...` rules, `"literals"`, `[a-z]` / `[^"]` classes,
//! `.`, `( groups )`, `|`, and the `*`, `+`, `?`, `{m}`, `{m,}`, `{m,n}`
//! repetitions. The matcher works on the bytes a token decodes to, keeping a set
//! of stacks of grammar positions (one per way the text so far can be read),
//! so a token is checked by feeding it its bytes — tokenization and grammar
//! never have to line up.

use crate::generate::{argmax, sample_from, Sampler, SplitMix64};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Each token id's decoded bytes (`None` for special tokens, which are never
/// text a grammar could match). Built once per tokenizer and shared.
pub type TokenBytes = Arc<Vec<Option<Vec<u8>>>>;

/// Most repetitions a `{m,n}` may ask for. Each one becomes a rule, and a
/// grammar comes from the client.
const MAX_REPEAT: usize = 1000;

/// A set of code points: inclusive ranges, matched or (negated) excluded.
#[derive(Debug, Clone, PartialEq)]
struct Class {
    ranges: Vec<(u32, u32)>,
    negated: bool,
}

impl Class {
    fn one(c: u32) -> Self {
        Class {
            ranges: vec![(c, c)],
            negated: false,
        }
    }

    fn matches(&self, c: u32) -> bool {
        self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != self.negated
    }

    /// Whether some code point in `lo..=hi` might match: the check for a token
    /// that ends partway through a UTF-8 character. A negated class is only
    /// ruled out when one range covers the whole span, which errs toward
    /// letting the token through; the next byte settles it either way.
    fn may_match(&self, lo: u32, hi: u32) -> bool {
        if self.negated {
            !self.ranges.iter().any(|&(a, b)| a <= lo && hi <= b)
        } else {
            self.ranges.iter().any(|&(a, b)| a <= hi && lo <= b)
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Elem {
    Char(Class),
    Rule(usize),
}

/// A parsed grammar: each rule is a list of alternatives, each a sequence.
#[derive(Debug)]
pub struct Grammar {
    rules: Vec<Vec<Vec<Elem>>>,
    root: usize,
}

/// A place in the grammar: element `idx` of alternative `alt` of `rule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Pos {
    rule: usize,
    alt: usize,
    idx: usize,
}

/// What is left to match, innermost rule last. Empty means the text so far is
/// a complete match.
type Stack = Vec<Pos>;

impl Grammar {
    /// Parse GBNF text. The start rule is `root`.
    pub fn parse(src: &str) -> Result<Self, String> {
        Parser {
            src: src.chars().collect(),
            pos: 0,
            names: HashMap::new(),
            rules: Vec::new(),
        }
        .grammar()
    }

    /// Any JSON object: OpenAI's `response_format: {"type": "json_object"}`.
    pub fn json_object() -> Self {
        Self::parse(&format!("root ::= object\n{JSON_RULES}")).expect("the JSON grammar parses")
    }

    /// JSON matching `schema` (the subset [`json_schema_to_gbnf`] supports).
    pub fn from_json_schema(schema: &Value) -> Result<Self, String> {
        Self::parse(&json_schema_to_gbnf(schema)?)
    }

    fn elem(&self, p: Pos) -> Option<&Elem> {
        self.rules[p.rule][p.alt].get(p.idx)
    }

    /// Resolve `stack` until its top is a character to match (or it is empty),
    /// pushing every way of doing so onto `out`.
    fn expand(&self, mut stack: Stack, out: &mut Vec<Stack>) {
        loop {
            let Some(&top) = stack.last() else {
                out.push(stack);
                return;
            };
            match self.elem(top) {
                None => {
                    stack.pop();
                }
                Some(Elem::Char(_)) => {
                    out.push(stack);
                    return;
                }
                Some(&Elem::Rule(r)) => {
                    // Step the caller past the reference, and drop it if that
                    // finishes it, so right recursion (`x ::= "a" x`) runs in
                    // constant stack.
                    let last = stack.last_mut().expect("non-empty");
                    last.idx += 1;
                    if self.elem(*last).is_none() {
                        stack.pop();
                    }
                    for alt in 0..self.rules[r].len() {
                        let mut s = stack.clone();
                        s.push(Pos {
                            rule: r,
                            alt,
                            idx: 0,
                        });
                        self.expand(s, out);
                    }
                    return;
                }
            }
        }
    }

    /// The stacks that remain after matching code point `c`.
    fn advance(&self, stacks: &[Stack], c: u32) -> Vec<Stack> {
        let mut out = Vec::new();
        for stack in stacks {
            let Some(&top) = stack.last() else { continue };
            if let Some(Elem::Char(class)) = self.elem(top) {
                if class.matches(c) {
                    let mut s = stack.clone();
                    s.last_mut().expect("non-empty").idx += 1;
                    self.expand(s, &mut out);
                }
            }
        }
        // Different readings often converge; keeping duplicates would double
        // the work on every byte after.
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// A request's grammar and how far through it the output is. Cloned fresh for
/// each request, then advanced token by token.
#[derive(Debug, Clone)]
pub struct Constraint {
    grammar: Arc<Grammar>,
    tokens: TokenBytes,
    /// End-of-sequence ids: allowed exactly when the grammar is satisfied.
    eos: Vec<u32>,
    stacks: Vec<Stack>,
    /// The start of a UTF-8 character whose remaining bytes are still to come.
    partial: Vec<u8>,
}

impl Constraint {
    /// Start matching `grammar` from its root. `tokens` gives each id's bytes.
    pub fn new(grammar: Arc<Grammar>, tokens: TokenBytes, eos: Vec<u32>) -> Self {
        let mut stacks = Vec::new();
        for alt in 0..grammar.rules[grammar.root].len() {
            let start = Pos {
                rule: grammar.root,
                alt,
                idx: 0,
            };
            grammar.expand(vec![start], &mut stacks);
        }
        stacks.sort_unstable();
        stacks.dedup();
        Constraint {
            grammar,
            tokens,
            eos,
            stacks,
            partial: Vec::new(),
        }
    }

    /// The output so far is a complete match.
    fn is_accepting(&self) -> bool {
        self.partial.is_empty() && self.stacks.iter().any(Vec::is_empty)
    }

    /// Nothing more can be added: the match is complete and admits no
    /// continuation (or the output left the grammar, which only the fallback in
    /// [`pick`](Self::pick) can do).
    pub fn is_exhausted(&self) -> bool {
        self.stacks.iter().all(Vec::is_empty)
    }

    /// The state after `bytes`, or `None` if the grammar cannot take them.
    fn feed(&self, bytes: &[u8]) -> Option<(Vec<Stack>, Vec<u8>)> {
        let mut stacks: Option<Vec<Stack>> = None; // None: still `self.stacks`
        let mut buf = [0u8; 4];
        let mut len = self.partial.len();
        buf[..len].copy_from_slice(&self.partial);
        for &b in bytes {
            buf[len] = b;
            len += 1;
            let current = stacks.as_deref().unwrap_or(&self.stacks);
            match std::str::from_utf8(&buf[..len]) {
                Ok(s) => {
                    let c = s.chars().next().expect("one char") as u32;
                    let next = self.grammar.advance(current, c);
                    if next.is_empty() {
                        return None;
                    }
                    stacks = Some(next);
                    len = 0;
                }
                // A character cut off by the token boundary: keep it if
                // anything it could still become is allowed.
                Err(e) if e.error_len().is_none() => {
                    let (lo, hi) = partial_range(&buf[..len])?;
                    let possible = current.iter().any(|s| {
                        s.last().is_some_and(|&top| {
                            matches!(self.grammar.elem(top), Some(Elem::Char(c)) if c.may_match(lo, hi))
                        })
                    });
                    if !possible {
                        return None;
                    }
                }
                Err(_) => return None,
            }
        }
        Some((
            stacks.unwrap_or_else(|| self.stacks.clone()),
            buf[..len].to_vec(),
        ))
    }

    /// Whether `token` may come next.
    fn allows(&self, token: u32) -> bool {
        if self.eos.contains(&token) {
            return self.is_accepting();
        }
        match self.tokens.get(token as usize) {
            // A token with no bytes would never move the match forward.
            Some(Some(bytes)) if !bytes.is_empty() => self.feed(bytes).is_some(),
            _ => false,
        }
    }

    /// Choose the next token: `sampler`'s draw if the grammar allows it,
    /// otherwise a draw from what remains of its distribution once every token
    /// the grammar refuses is dropped (llama.cpp's order: the full check over
    /// the candidates only runs when the cheap one fails). When truncation left
    /// nothing allowed, the highest-scoring allowed token in the vocabulary.
    pub fn pick(&self, sampler: &Sampler, logits: &[f32], rng: &mut SplitMix64) -> u32 {
        let dist = sampler.distribution(logits);
        let first = sample_from(&dist, rng);
        if self.allows(first) {
            return first;
        }
        let mut allowed: Vec<(u32, f32)> =
            dist.into_iter().filter(|&(t, _)| self.allows(t)).collect();
        if !allowed.is_empty() {
            let total: f32 = allowed.iter().map(|&(_, p)| p).sum();
            for (_, p) in &mut allowed {
                *p /= total;
            }
            return sample_from(&allowed, rng);
        }
        // ponytail: sorts the whole vocabulary (~10 ms at 152k) on each such
        // step; a partial select would do if that ever shows in a profile.
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .total_cmp(&logits[a as usize])
                .then(a.cmp(&b))
        });
        // No token fits (a grammar no token can continue): end the request if
        // there is an EOS to end it with.
        order
            .into_iter()
            .find(|&t| self.allows(t))
            .or_else(|| self.eos.first().copied())
            .unwrap_or_else(|| argmax(logits))
    }

    /// Advance past `token`, which [`pick`](Self::pick) chose.
    pub fn accept(&mut self, token: u32) {
        if self.eos.contains(&token) {
            return;
        }
        let fed = self
            .tokens
            .get(token as usize)
            .and_then(|b| b.as_deref())
            .and_then(|b| self.feed(b));
        match fed {
            Some((stacks, partial)) => {
                self.stacks = stacks;
                self.partial = partial;
            }
            None => {
                self.stacks.clear();
                self.partial.clear();
            }
        }
    }
}

/// The code points an incomplete UTF-8 sequence could still become, or `None`
/// if it can only end as an overlong (invalid) encoding.
fn partial_range(bytes: &[u8]) -> Option<(u32, u32)> {
    let lead = bytes[0];
    // Length, payload bits of the lead byte, and the least code point that
    // needs this many bytes (anything below is overlong, so not UTF-8).
    let (len, mask, least) = match lead {
        0xC0..=0xDF => (2, 0x1F, 0x80),
        0xE0..=0xEF => (3, 0x0F, 0x800),
        _ => (4, 0x07, 0x10000),
    };
    let mut lo = u32::from(lead & mask);
    for &b in &bytes[1..] {
        lo = lo << 6 | u32::from(b & 0x3F);
    }
    let missing = 6 * (len - bytes.len()) as u32;
    let hi = ((lo << missing) | ((1 << missing) - 1)).min(0x10FFFF);
    let lo = (lo << missing).max(least);
    (lo <= hi).then_some((lo, hi))
}

struct Parser {
    src: Vec<char>,
    pos: usize,
    names: HashMap<String, usize>,
    /// `None` for a rule referenced but not (yet) defined.
    rules: Vec<Option<Vec<Vec<Elem>>>>,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.src.get(self.pos).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        let hit = self.peek() == Some(c);
        if hit {
            self.pos += 1;
        }
        hit
    }

    fn err<T>(&self, msg: impl std::fmt::Display) -> Result<T, String> {
        let line = 1 + self.src[..self.pos].iter().filter(|&&c| c == '\n').count();
        Err(format!("grammar: {msg} (line {line})"))
    }

    /// Skip blanks and `#` comments; newlines too when `newlines` (inside
    /// parentheses, or after `::=` or `|`). At the top level a newline ends
    /// the rule, as in llama.cpp.
    fn skip(&mut self, newlines: bool) {
        while let Some(c) = self.peek() {
            match c {
                ' ' | '\t' | '\r' => self.pos += 1,
                '\n' if newlines => self.pos += 1,
                '#' => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn rule_id(&mut self, name: &str) -> usize {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        self.rules.push(None);
        self.names.insert(name.to_string(), self.rules.len() - 1);
        self.rules.len() - 1
    }

    fn new_rule(&mut self, alts: Vec<Vec<Elem>>) -> usize {
        self.rules.push(Some(alts));
        self.rules.len() - 1
    }

    fn name(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            self.pos += 1;
        }
        if start == self.pos {
            return self.err("expected a rule name");
        }
        Ok(self.src[start..self.pos].iter().collect())
    }

    fn grammar(mut self) -> Result<Grammar, String> {
        self.skip(true);
        while self.peek().is_some() {
            let name = self.name()?;
            self.skip(false);
            if !(self.eat(':') && self.eat(':') && self.eat('=')) {
                return self.err(format!("expected '::=' after '{name}'"));
            }
            self.skip(true);
            let alts = self.alternates(false)?;
            let id = self.rule_id(&name);
            if self.rules[id].is_some() {
                return self.err(format!("rule '{name}' is defined twice"));
            }
            self.rules[id] = Some(alts);
            self.skip(true);
        }
        let Some(&root) = self.names.get("root") else {
            return Err("grammar: no 'root' rule".into());
        };
        if let Some((name, _)) = self.names.iter().find(|(_, &id)| self.rules[id].is_none()) {
            return Err(format!("grammar: rule '{name}' is used but never defined"));
        }
        let grammar = Grammar {
            rules: self
                .rules
                .into_iter()
                .map(|r| r.expect("defined"))
                .collect(),
            root,
        };
        check_left_recursion(&grammar, &self.names)?;
        Ok(grammar)
    }

    fn alternates(&mut self, nested: bool) -> Result<Vec<Vec<Elem>>, String> {
        let mut alts = vec![self.sequence(nested)?];
        while self.eat('|') {
            self.skip(true);
            alts.push(self.sequence(nested)?);
        }
        Ok(alts)
    }

    fn sequence(&mut self, nested: bool) -> Result<Vec<Elem>, String> {
        let mut seq = Vec::new();
        loop {
            self.skip(nested);
            let term = match self.peek() {
                None | Some('|') | Some('\n') => break,
                Some(')') if nested => break,
                Some('"') => {
                    self.pos += 1;
                    let mut term = Vec::new();
                    while !self.eat('"') {
                        if self.peek().is_none() {
                            return self.err("unterminated string");
                        }
                        term.push(Elem::Char(Class::one(self.char_lit()?)));
                    }
                    term
                }
                Some('[') => vec![Elem::Char(self.class()?)],
                Some('.') => {
                    self.pos += 1;
                    vec![Elem::Char(Class {
                        ranges: Vec::new(),
                        negated: true,
                    })]
                }
                Some('(') => {
                    self.pos += 1;
                    self.skip(true);
                    let alts = self.alternates(true)?;
                    if !self.eat(')') {
                        return self.err("expected ')'");
                    }
                    vec![Elem::Rule(self.new_rule(alts))]
                }
                Some(c) if c.is_ascii_alphanumeric() || c == '-' || c == '_' => {
                    let name = self.name()?;
                    vec![Elem::Rule(self.rule_id(&name))]
                }
                Some(c) => return self.err(format!("unexpected '{c}'")),
            };
            self.skip(nested);
            let repeated = self.repetition(term)?;
            seq.extend(repeated);
        }
        Ok(seq)
    }

    /// Apply a `*`, `+`, `?` or `{m,n}` that follows `term`, if any.
    fn repetition(&mut self, term: Vec<Elem>) -> Result<Vec<Elem>, String> {
        let (min, max) = match self.peek() {
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('?') => (0, Some(1)),
            Some('{') => {
                self.pos += 1;
                self.skip(false);
                let min = self.number()?;
                self.skip(false);
                let max = if self.eat(',') {
                    self.skip(false);
                    if self.peek().is_some_and(|c| c.is_ascii_digit()) {
                        Some(self.number()?)
                    } else {
                        None
                    }
                } else {
                    Some(min)
                };
                self.skip(false);
                if self.peek() != Some('}') {
                    return self.err("expected '}'");
                }
                if max.is_some_and(|max| max < min) {
                    return self.err(format!("repetition {{{min},..}} has its bounds reversed"));
                }
                (min, max)
            }
            _ => return Ok(term),
        };
        self.pos += 1;
        let mut out: Vec<Elem> = (0..min).flat_map(|_| term.clone()).collect();
        match max {
            // r ::= term r | ""
            None => {
                let r = self.rules.len();
                let mut again = term;
                again.push(Elem::Rule(r));
                self.new_rule(vec![again, Vec::new()]);
                out.push(Elem::Rule(r));
            }
            // Nested optionals, r_k ::= term r_{k-1} | "", rather than a run of
            // independent ones: `x? x? x?` could read one `x` three ways.
            Some(max) => {
                let mut tail = None;
                for _ in min..max {
                    let mut alt = term.clone();
                    alt.extend(tail.map(Elem::Rule));
                    tail = Some(self.new_rule(vec![alt, Vec::new()]));
                }
                out.extend(tail.map(Elem::Rule));
            }
        }
        Ok(out)
    }

    fn number(&mut self) -> Result<usize, String> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.pos += 1;
        }
        let digits: String = self.src[start..self.pos].iter().collect();
        match digits.parse::<usize>() {
            Ok(n) if n <= MAX_REPEAT => Ok(n),
            Ok(_) | Err(_) if !digits.is_empty() => {
                self.err(format!("repetition count is over {MAX_REPEAT}"))
            }
            _ => self.err("expected a number"),
        }
    }

    /// `[...]` with ranges and an optional leading `^`.
    fn class(&mut self) -> Result<Class, String> {
        self.pos += 1;
        let negated = self.eat('^');
        let mut ranges = Vec::new();
        while !self.eat(']') {
            if self.peek().is_none() {
                return self.err("unterminated '['");
            }
            let lo = self.char_lit()?;
            let hi = if self.peek() == Some('-')
                && self.src.get(self.pos + 1).is_some_and(|&c| c != ']')
            {
                self.pos += 1;
                self.char_lit()?
            } else {
                lo
            };
            if hi < lo {
                return self.err("character range is reversed");
            }
            ranges.push((lo, hi));
        }
        Ok(Class { ranges, negated })
    }

    /// One character of a literal or class, with its escape resolved.
    fn char_lit(&mut self) -> Result<u32, String> {
        let c = self.peek().expect("caller checked");
        self.pos += 1;
        if c != '\\' {
            return Ok(c as u32);
        }
        let Some(e) = self.peek() else {
            return self.err("dangling '\\'");
        };
        self.pos += 1;
        Ok(match e {
            'n' => '\n' as u32,
            'r' => '\r' as u32,
            't' => '\t' as u32,
            'x' => self.hex(2)?,
            'u' => self.hex(4)?,
            'U' => self.hex(8)?,
            other => other as u32,
        })
    }

    fn hex(&mut self, digits: usize) -> Result<u32, String> {
        let s: String = self.src.iter().skip(self.pos).take(digits).collect();
        match u32::from_str_radix(&s, 16) {
            Ok(v) if s.len() == digits && char::from_u32(v).is_some() => {
                self.pos += digits;
                Ok(v)
            }
            _ => self.err(format!("bad escape: expected {digits} hex digits")),
        }
    }
}

/// Refuse left recursion (`a ::= a "x"`, or `a ::= b`, `b ::= a`): matching
/// would expand it forever, and a grammar is client input.
fn check_left_recursion(g: &Grammar, names: &HashMap<String, usize>) -> Result<(), String> {
    let n = g.rules.len();
    let mut nullable = vec![false; n];
    loop {
        let mut changed = false;
        for r in 0..n {
            if !nullable[r]
                && g.rules[r].iter().any(|alt| {
                    alt.iter()
                        .all(|e| matches!(e, Elem::Rule(x) if nullable[*x]))
                })
            {
                nullable[r] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // r → x when x can be the first thing r matches.
    let starts: Vec<Vec<usize>> = g
        .rules
        .iter()
        .map(|alts| {
            let mut v = Vec::new();
            for alt in alts {
                for e in alt {
                    let Elem::Rule(x) = *e else { break };
                    v.push(x);
                    if !nullable[x] {
                        break;
                    }
                }
            }
            v
        })
        .collect();
    // A cycle in that graph is left recursion. Iterative DFS: 1 = on the path.
    let mut state = vec![0u8; n];
    for start in 0..n {
        if state[start] != 0 {
            continue;
        }
        state[start] = 1;
        let mut path = vec![(start, 0usize)];
        while let Some((r, i)) = path.last_mut() {
            let Some(&x) = starts[*r].get(*i) else {
                state[*r] = 2;
                path.pop();
                continue;
            };
            *i += 1;
            match state[x] {
                0 => {
                    state[x] = 1;
                    path.push((x, 0));
                }
                1 => {
                    let what = names.iter().find(|(_, &id)| id == x).map_or_else(
                        // `("a"?)*`: a repetition of something that can be empty.
                        || "a repeated group that can match nothing".to_string(),
                        |(n, _)| format!("rule '{n}'"),
                    );
                    return Err(format!("grammar: {what} is left-recursive"));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// The JSON rules every JSON grammar here builds on. Whitespace and number
/// lengths are bounded (as llama.cpp's are), so a model that keeps choosing
/// spaces or digits still has to finish.
const JSON_RULES: &str = r##"
value   ::= object | array | string | number | boolean | null
object  ::= "{" ws ( string ":" ws value ( "," ws string ":" ws value )* )? "}" ws
array   ::= "[" ws ( value ( "," ws value )* )? "]" ws
string  ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\"" ws
number  ::= int ( "." [0-9]{1,16} )? ( [eE] [-+]? [0-9]{1,16} )? ws
integer ::= int ws
int     ::= "-"? ( "0" | [1-9] [0-9]{0,15} )
boolean ::= ( "true" | "false" ) ws
null    ::= "null" ws
ws      ::= | " " | "\n" [ \t]{0,20}
"##;

/// GBNF for JSON matching `schema`. Supports `type` (object, array, string,
/// number, integer, boolean, null, or a list of them), `properties` /
/// `required` (no other keys allowed), `items`, `enum`, `const`, `anyOf` and
/// `oneOf`; refuses the keywords it would otherwise silently ignore.
/// Properties come out in key order, required ones first.
pub fn json_schema_to_gbnf(schema: &Value) -> Result<String, String> {
    Ok(format!("root ::= {}\n{JSON_RULES}", schema_expr(schema)?))
}

fn schema_expr(schema: &Value) -> Result<String, String> {
    let obj = match schema {
        Value::Object(o) => o,
        Value::Bool(true) => return Ok("value".into()),
        _ => return Err("json_schema: a schema must be an object".into()),
    };
    if let Some(c) = obj.get("const") {
        return Ok(format!("{} ws", literal(c)));
    }
    if let Some(e) = obj.get("enum") {
        let values = e
            .as_array()
            .filter(|a| !a.is_empty())
            .ok_or("json_schema: enum must be a non-empty array")?;
        let alts: Vec<String> = values.iter().map(literal).collect();
        return Ok(format!("( {} ) ws", alts.join(" | ")));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(v) = obj.get(key) {
            let schemas = v
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or(format!("json_schema: {key} must be a non-empty array"))?;
            let alts = schemas
                .iter()
                .map(schema_expr)
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(format!("( {} )", alts.join(" | ")));
        }
    }
    for key in ["$ref", "allOf", "not", "if", "patternProperties"] {
        if obj.contains_key(key) {
            return Err(format!("json_schema: '{key}' is not supported"));
        }
    }
    match obj.get("type") {
        None => Ok("value".into()),
        Some(Value::String(t)) => type_expr(t, obj),
        Some(Value::Array(types)) if !types.is_empty() => {
            let alts = types
                .iter()
                .map(|t| match t.as_str() {
                    Some(t) => type_expr(t, obj),
                    None => Err("json_schema: type must be a string".into()),
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("( {} )", alts.join(" | ")))
        }
        Some(_) => Err("json_schema: type must be a string or a list of them".into()),
    }
}

fn type_expr(t: &str, obj: &serde_json::Map<String, Value>) -> Result<String, String> {
    Ok(match t {
        "string" | "number" | "integer" | "boolean" | "null" => t.to_string(),
        "array" => {
            let item = match obj.get("items") {
                Some(items) => schema_expr(items)?,
                None => "value".into(),
            };
            format!(r#""[" ws ( {item} ( "," ws {item} )* )? "]" ws"#)
        }
        "object" => object_expr(obj)?,
        other => return Err(format!("json_schema: unknown type '{other}'")),
    })
}

fn object_expr(obj: &serde_json::Map<String, Value>) -> Result<String, String> {
    let Some(props) = obj.get("properties") else {
        return Ok("object".into());
    };
    let props = props
        .as_object()
        .ok_or("json_schema: properties must be an object")?;
    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let (mut req, mut opt) = (Vec::new(), Vec::new());
    for (name, schema) in props {
        let pair = format!(
            r#"{} ":" ws {}"#,
            literal(&Value::String(name.clone())),
            schema_expr(schema)?
        );
        if required.contains(&name.as_str()) {
            req.push(pair);
        } else {
            opt.push(pair);
        }
    }
    let maybe = |pair: &String| format!(r#"( "," ws {pair} )?"#);
    let body = if !req.is_empty() {
        let mut parts = vec![req.join(r#" "," ws "#)];
        parts.extend(opt.iter().map(maybe));
        parts.join(" ")
    } else if !opt.is_empty() {
        // Nothing required: any optional property may come first, and each
        // later one may follow it.
        let firsts: Vec<String> = (0..opt.len())
            .map(|i| {
                let rest: Vec<String> = opt[i + 1..].iter().map(maybe).collect();
                format!("{} {}", opt[i], rest.join(" "))
            })
            .collect();
        format!("( {} )?", firsts.join(" | "))
    } else {
        String::new()
    };
    Ok(format!(r#""{{" ws {body} "}}" ws"#))
}

/// A GBNF literal matching `value`'s JSON text exactly.
fn literal(value: &Value) -> String {
    let json = value.to_string();
    let mut out = String::with_capacity(json.len() + 2);
    out.push('"');
    for c in json.chars() {
        if c == '"' || c == '\\' {
            out.push('\\');
        }
        out.push(c);
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each byte of the string is its own token, like the byte tokenizer.
    fn bytes_table() -> TokenBytes {
        Arc::new((0..=255u8).map(|b| Some(vec![b])).collect())
    }

    fn matches(grammar: &str, text: &str) -> bool {
        let g = Arc::new(Grammar::parse(grammar).unwrap());
        let mut c = Constraint::new(g, bytes_table(), Vec::new());
        for &b in text.as_bytes() {
            if !c.allows(b as u32) {
                return false;
            }
            c.accept(b as u32);
        }
        c.is_accepting()
    }

    #[test]
    fn matches_literals_classes_and_repetition() {
        let g = r#"root ::= "a" [0-9]+ ("x" | "yz")? "!"{2}"#;
        assert!(matches(g, "a1!!"));
        assert!(matches(g, "a123yz!!"));
        assert!(!matches(g, "a!!"), "+ needs one digit");
        assert!(!matches(g, "a1y!!"), "half an alternative");
        assert!(!matches(g, "a1!"), "{{2}} is exact");
        assert!(!matches(g, "a1!!!"));
        let bounded = r#"root ::= "<" [a-c]{1,3} ">""#;
        assert!(matches(bounded, "<abc>"));
        assert!(!matches(bounded, "<abca>"));
        assert!(!matches(bounded, "<>"));
        assert!(matches(r#"root ::= [^"\\]* "\"""#, "it's\""));
        assert!(!matches(r#"root ::= [^"\\]* "\"""#, "a\\b\""));
    }

    #[test]
    fn rules_comments_and_multiline_groups() {
        let g = "# a list of words\nroot ::= word (\n  \",\" word\n)*\nword ::= [a-z]+ # letters\n";
        assert!(matches(g, "a,bc,d"));
        assert!(!matches(g, "a,,b"));
        assert!(!matches(g, ""));
    }

    #[test]
    fn utf8_split_across_tokens_is_matched_by_code_point() {
        let g = Arc::new(Grammar::parse(r#"root ::= "é" [α-ω]"#).unwrap());
        let mut c = Constraint::new(g, bytes_table(), Vec::new());
        let text = "éβ".as_bytes(); // four bytes, two characters
        assert!(c.allows(text[0] as u32));
        // 0xC3 starts é but also, say, ü: only the full character decides.
        c.accept(text[0] as u32);
        assert!(!c.allows(0xBC), "0xC3 0xBC is ü");
        for &b in &text[1..] {
            assert!(c.allows(b as u32));
            c.accept(b as u32);
        }
        assert!(c.is_accepting());
        // A byte starting a character outside every allowed range is refused.
        let g = Arc::new(Grammar::parse(r#"root ::= [a-z]"#).unwrap());
        let c = Constraint::new(g, bytes_table(), Vec::new());
        assert!(!c.allows(0xE2));
        // Nor one whose only completions in range would be overlong: E0 starts
        // a three-byte character, U+0800 or above, whatever its payload bits.
        assert!(!c.allows(0xE0));
        assert!(!c.allows(0xF0));
    }

    #[test]
    fn eos_only_when_the_grammar_is_satisfied() {
        let g = Arc::new(Grammar::parse(r#"root ::= "ab" "c"?"#).unwrap());
        let mut c = Constraint::new(g, bytes_table(), vec![0]);
        assert!(!c.allows(0));
        c.accept(b'a' as u32);
        assert!(!c.allows(0));
        c.accept(b'b' as u32);
        assert!(c.allows(0), "ab is complete");
        assert!(!c.is_exhausted(), "c may still follow");
        c.accept(b'c' as u32);
        assert!(c.is_exhausted());
        // The only thing left to pick is EOS, whatever the logits say.
        let logits: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let pick = c.pick(&Sampler::Greedy, &logits, &mut SplitMix64::new(0));
        assert_eq!(pick, 0);
    }

    #[test]
    fn pick_takes_the_best_token_the_grammar_allows() {
        let g = Arc::new(Grammar::parse(r#"root ::= [x-z]"#).unwrap());
        let c = Constraint::new(g, bytes_table(), Vec::new());
        let mut logits = vec![0.0f32; 256];
        logits[b'a' as usize] = 9.0; // the model's favourite, not allowed
        logits[b'y' as usize] = 5.0;
        logits[b'z' as usize] = 4.0;
        let mut rng = SplitMix64::new(1);
        assert_eq!(c.pick(&Sampler::Greedy, &logits, &mut rng), b'y' as u32);
        // Sampling: only x, y, z can ever come out.
        let sampler = Sampler::TopPK {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 1,
        };
        for _ in 0..50 {
            let t = c.pick(&sampler, &logits, &mut rng);
            assert!((b'x'..=b'z').contains(&(t as u8)), "{t}");
        }
    }

    #[test]
    fn bad_grammars_are_refused_with_a_reason() {
        for (g, why) in [
            ("root ::= a", "never defined"),
            ("start ::= \"a\"", "no 'root'"),
            ("root ::= root \"a\"", "left-recursive"),
            ("root ::= (\"a\"*)*", "left-recursive"),
            ("root ::= \"a", "unterminated"),
            ("root ::= [a-", "unterminated"),
            ("root ::= \"a\"{3,1}", "reversed"),
            ("root ::= \"a\"{5000}", "over"),
            ("root = \"a\"", "'::='"),
            ("root ::= \"a\" )", "unexpected ')'"),
            ("root ::= \"a\"\nroot ::= \"b\"", "defined twice"),
        ] {
            let e = Grammar::parse(g).unwrap_err();
            assert!(e.contains(why), "{g:?}: {e}");
        }
    }

    #[test]
    fn json_object_grammar_takes_json_objects_only() {
        let g = format!("root ::= object\n{JSON_RULES}");
        assert!(matches(
            &g,
            r#"{"a": [1, -2.5e3, true, null], "b": {"c": "d\né"}}"#
        ));
        assert!(matches(&g, "{}"));
        assert!(!matches(&g, "[1]"));
        assert!(!matches(&g, r#"{"a": 01}"#));
        assert!(!matches(&g, r#"{"a" 1}"#));
        Grammar::json_object();
    }

    #[test]
    fn json_schema_becomes_a_grammar_for_exactly_that_shape() {
        let schema: Value = serde_json::from_str(
            r#"{"type":"object","properties":{
                "name":{"type":"string"},
                "age":{"type":"integer"},
                "tags":{"type":"array","items":{"enum":["a","b"]}},
                "kind":{"anyOf":[{"const":1},{"type":"null"}]},
                "note":{"type":["string","null"]}
            },"required":["name","age"]}"#,
        )
        .unwrap();
        let g = json_schema_to_gbnf(&schema).unwrap();
        // Required first (in key order), then the optional ones, in key order.
        assert!(matches(&g, r#"{"age": 3, "name": "x"}"#), "{g}");
        assert!(matches(
            &g,
            r#"{"age":3,"name":"x","kind":1,"tags":["a","b"]}"#
        ));
        assert!(matches(&g, r#"{"age":3,"name":"x","note":null}"#));
        assert!(!matches(&g, r#"{"age":3}"#), "name is required");
        assert!(!matches(&g, r#"{"age":3.5,"name":"x"}"#), "integer");
        assert!(!matches(&g, r#"{"age":3,"name":"x","tags":["c"]}"#), "enum");
        assert!(!matches(&g, r#"{"age":3,"name":"x","kind":2}"#), "const");
        assert!(
            !matches(&g, r#"{"age":3,"name":"x","extra":1}"#),
            "no extra keys"
        );

        // With nothing required, any subset in order — including none.
        let opt: Value = serde_json::from_str(
            r#"{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"null"}}}"#,
        )
        .unwrap();
        let g = json_schema_to_gbnf(&opt).unwrap();
        for ok in [
            "{}",
            r#"{"a":true}"#,
            r#"{"b":null}"#,
            r#"{"a":false,"b":null}"#,
        ] {
            assert!(matches(&g, ok), "{ok}");
        }
        assert!(!matches(&g, r#"{,"b":null}"#));

        for (bad, why) in [
            (r##"{"$ref":"#/x"}"##, "'$ref' is not supported"),
            (r#"{"type":"date"}"#, "unknown type"),
            (r#"{"enum":[]}"#, "non-empty"),
            ("3", "must be an object"),
        ] {
            let e = json_schema_to_gbnf(&serde_json::from_str(bad).unwrap()).unwrap_err();
            assert!(e.contains(why), "{bad}: {e}");
        }
    }
}
