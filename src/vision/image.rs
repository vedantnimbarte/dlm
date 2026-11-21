//! Image decoding and Gemma 3's preprocessing: what `Gemma3ImageProcessor`
//! does to a picture before the vision tower sees it.
//!
//! The processor converts to RGB, resizes to the tower's square input with
//! antialiased bilinear filtering (ignoring the aspect ratio), and maps each
//! byte to `(x / 255 - 0.5) / 0.5`. Pan-and-scan is off by default and not
//! implemented. The resize is computed on the bytes, the way `transformers`'
//! default (torchvision) backend does it: separable, horizontal pass first,
//! each pass rounding back to a byte, with fixed-point weights.

use crate::error::{DlmError, Result};

/// An 8-bit RGB image, rows top to bottom, `width * height * 3` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// The largest image dlm decodes, in pixels. A small compressed file can claim
/// an enormous canvas; this bounds what one request can make the server
/// allocate (64 Mpx is 192 MiB of RGB).
const MAX_PIXELS: usize = 64 << 20;

fn bad(msg: impl Into<String>) -> DlmError {
    DlmError::InvalidConfig(format!("image: {}", msg.into()))
}

/// Decode a PNG or JPEG file's bytes to RGB. Alpha is dropped (as PIL's
/// `convert("RGB")` drops it, without compositing) and grey is replicated.
pub fn decode(bytes: &[u8]) -> Result<Rgb> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(bytes)
    } else if bytes.starts_with(&[0xFF, 0xD8]) {
        decode_jpeg(bytes)
    } else {
        Err(bad("not a PNG or JPEG file"))
    }
}

fn check_size(width: usize, height: usize) -> Result<()> {
    if width == 0 || height == 0 || width.saturating_mul(height) > MAX_PIXELS {
        return Err(bad(format!(
            "{width}x{height} is empty or larger than {MAX_PIXELS} pixels"
        )));
    }
    Ok(())
}

fn decode_png(bytes: &[u8]) -> Result<Rgb> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Palette and low bit depths expand to 8-bit samples, 16-bit ones are cut
    // to their high byte: 8-bit grey, grey+alpha, RGB or RGBA comes out.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| bad(e.to_string()))?;
    let (width, height) = {
        let info = reader.info();
        (info.width as usize, info.height as usize)
    };
    check_size(width, height)?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| bad("PNG too large"))?;
    let mut buf = vec![0u8; size];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| bad(e.to_string()))?;
    let channels = match frame.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(bad("unexpanded palette PNG")),
    };
    let mut data = Vec::with_capacity(width * height * 3);
    for row in buf.chunks(frame.line_size).take(height) {
        for px in row[..width * channels].chunks_exact(channels) {
            match channels {
                1 | 2 => data.extend_from_slice(&[px[0]; 3]),
                _ => data.extend_from_slice(&px[..3]),
            }
        }
    }
    Ok(Rgb {
        width,
        height,
        data,
    })
}

fn decode_jpeg(bytes: &[u8]) -> Result<Rgb> {
    use zune_jpeg::zune_core::{
        bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions,
    };
    let options = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(bytes), options);
    decoder
        .decode_headers()
        .map_err(|e| bad(format!("{e:?}")))?;
    let info = decoder.info().ok_or_else(|| bad("JPEG has no frame"))?;
    let (width, height) = (info.width as usize, info.height as usize);
    check_size(width, height)?;
    let data = decoder.decode().map_err(|e| bad(format!("{e:?}")))?;
    if data.len() != width * height * 3 {
        return Err(bad("JPEG did not decode to RGB"));
    }
    Ok(Rgb {
        width,
        height,
        data,
    })
}

/// One output sample's taps: the first input index and a fixed-point weight
/// per input sample from there.
struct Taps {
    first: usize,
    weights: Vec<i32>,
}

/// Antialiased bilinear taps for resampling `inp` samples to `out`, and the
/// fixed-point precision they are in.
///
/// The filter is the triangle, widened by the scale when shrinking (that is
/// the antialiasing) -- PIL's `precompute_coeffs`, which torchvision's uint8
/// kernel shares. The weights are then rounded to integers as that kernel
/// rounds them: with as many fraction bits as keep the largest weight inside
/// 15 bits.
fn taps(inp: usize, out: usize) -> (Vec<Taps>, u32) {
    let scale = inp as f64 / out as f64;
    let filterscale = scale.max(1.0);
    let support = filterscale; // the triangle's support is 1
    let mut all = Vec::with_capacity(out);
    for o in 0..out {
        let center = (o as f64 + 0.5) * scale;
        let lo = ((center - support + 0.5) as i64).max(0) as usize;
        let hi = ((center + support + 0.5) as i64).min(inp as i64) as usize;
        let w: Vec<f64> = (lo..hi)
            .map(|i| {
                let x = ((i as f64 - center + 0.5) / filterscale).abs();
                if x < 1.0 {
                    1.0 - x
                } else {
                    0.0
                }
            })
            .collect();
        let sum: f64 = w.iter().sum();
        all.push((lo, w.into_iter().map(|v| v / sum).collect::<Vec<f64>>()));
    }
    let max = all
        .iter()
        .flat_map(|(_, w)| w.iter().copied())
        .fold(0.0f64, f64::max);
    let mut precision = 0u32;
    while precision < 22 && ((0.5 + max * (1u64 << (precision + 1)) as f64) as i64) < (1 << 15) {
        precision += 1;
    }
    let taps = all
        .into_iter()
        .map(|(first, w)| Taps {
            first,
            weights: w
                .iter()
                .map(|v| (v * (1u64 << precision) as f64).round() as i32)
                .collect(),
        })
        .collect();
    (taps, precision)
}

/// Resample one axis. `len` samples along it, `count` lines across it; sample
/// `i` of line `l` (channel `c`) lives at `l * line + i * step + c`.
fn pass(
    src: &[u8],
    (taps, precision): &(Vec<Taps>, u32),
    count: usize,
    at: impl Fn(usize, usize) -> usize,
    out: &mut [u8],
    out_at: impl Fn(usize, usize) -> usize,
) {
    let round = 1i32 << (precision.saturating_sub(1));
    for l in 0..count {
        for (o, t) in taps.iter().enumerate() {
            for c in 0..3 {
                let mut acc = round;
                for (k, &w) in t.weights.iter().enumerate() {
                    acc += w * src[at(l, t.first + k) + c] as i32;
                }
                out[out_at(l, o) + c] = (acc >> precision).clamp(0, 255) as u8;
            }
        }
    }
}

/// Resize to `width` x `height` with antialiased bilinear filtering (see the
/// module docs). An axis already the right length is left as it is.
pub fn resize(img: &Rgb, width: usize, height: usize) -> Rgb {
    let mut cur = img.clone();
    if cur.width != width {
        let mut out = vec![0u8; width * cur.height * 3];
        let (w_in, h) = (cur.width, cur.height);
        pass(
            &cur.data,
            &taps(w_in, width),
            h,
            |row, x| (row * w_in + x) * 3,
            &mut out,
            |row, x| (row * width + x) * 3,
        );
        cur = Rgb {
            width,
            height: h,
            data: out,
        };
    }
    if cur.height != height {
        let mut out = vec![0u8; width * height * 3];
        pass(
            &cur.data,
            &taps(cur.height, height),
            width,
            |col, y| (y * width + col) * 3,
            &mut out,
            |col, y| (y * width + col) * 3,
        );
        cur = Rgb {
            width,
            height,
            data: out,
        };
    }
    cur
}

/// `pixel_values` for one image, `[3, size, size]` channel-major: resized to
/// the tower's input and mapped to `[-1, 1]` (mean and std 0.5, as every Gemma 3
/// checkpoint's processor config states).
pub fn preprocess(img: &Rgb, size: usize) -> Vec<f32> {
    let r = resize(img, size, size);
    let mut out = vec![0.0f32; 3 * size * size];
    for (i, px) in r.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            out[c * size * size + i] = (px[c] as f32 - 127.5) / 127.5;
        }
    }
    out
}

/// The bytes of a `data:` URL (`data:image/png;base64,...`). Only base64 data
/// URLs are taken: an image is fetched by the client, never by the server.
pub fn data_url_bytes(url: &str) -> Result<Vec<u8>> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| bad("only data: URLs are accepted (send the image inline, base64)"))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| bad("malformed data: URL"))?;
    if !meta.ends_with(";base64") {
        return Err(bad("data: URL is not base64"));
    }
    base64_decode(payload).ok_or_else(|| bad("data: URL is not valid base64"))
}

/// Standard base64 (RFC 4648), padding optional; whitespace is skipped.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for b in s.bytes().filter(|b| !b.is_ascii_whitespace()) {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            _ => return None,
        };
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_round_trips_the_rfc_vectors() {
        for (enc, dec) in [
            ("", ""),
            ("Zg==", "f"),
            ("Zm8=", "fo"),
            ("Zm9v", "foo"),
            ("Zm9vYg", "foob"),
            ("Zm9vYmE=", "fooba"),
            ("Zm9v\nYmFy", "foobar"),
        ] {
            assert_eq!(base64_decode(enc).unwrap(), dec.as_bytes(), "{enc}");
        }
        assert!(base64_decode("Zm9v!").is_none());
        assert!(data_url_bytes("https://example.com/a.png").is_err());
        assert_eq!(
            data_url_bytes("data:image/png;base64,Zm9v").unwrap(),
            b"foo"
        );
    }

    /// Resizing to the same size is the identity, and a flat image stays flat
    /// whichever way it is scaled (the weights sum to one in fixed point too).
    #[test]
    fn resize_keeps_flat_images_flat() {
        let img = Rgb {
            width: 37,
            height: 23,
            data: [10u8, 128, 250].repeat(37 * 23),
        };
        assert_eq!(resize(&img, 37, 23), img);
        for (w, h) in [(14, 14), (64, 50), (5, 90)] {
            let r = resize(&img, w, h);
            assert!(r.data.chunks(3).all(|p| p == [10, 128, 250]), "{w}x{h}");
        }
    }

    #[test]
    fn png_decodes_to_rgb() {
        // A 2x1 RGBA PNG, encoded with the same crate.
        let mut bytes = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut bytes, 2, 1);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            let mut w = enc.write_header().unwrap();
            w.write_image_data(&[1, 2, 3, 255, 4, 5, 6, 0]).unwrap();
        }
        let img = decode(&bytes).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.data, vec![1, 2, 3, 4, 5, 6]);
        assert!(decode(b"GIF89a").is_err());
    }
}
