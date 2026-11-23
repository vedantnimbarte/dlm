//! The vision tower's encoder layers on the GPU (`dlm_vit_block` in
//! `src/gpu/kernels.cu`).
//!
//! Each layer's weights are uploaded, run over every patch and dropped before
//! the next, and the activations live in buffers freed when the image is done:
//! the tower holds about 300 MB of VRAM while it runs (Gemma 3's 4096 patches)
//! and none after, so it fits beside a resident or a streamed text model.

use super::{VisionConfig, VisionTower, VitLayer};
use crate::error::{DlmError, Result};
use crate::gpu::device::DeviceBuffer;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    fn dlm_vit_block(
        t: i32,
        hidden: i32,
        heads: i32,
        head_dim: i32,
        inter: i32,
        eps: f32,
        attn_scale: f32,
        ln1_w: *const f32,
        ln1_b: *const f32,
        wq: *const f32,
        bq: *const f32,
        wk: *const f32,
        bk: *const f32,
        wv: *const f32,
        bv: *const f32,
        wo: *const f32,
        bo: *const f32,
        ln2_w: *const f32,
        ln2_b: *const f32,
        w1: *const f32,
        b1: *const f32,
        w2: *const f32,
        b2: *const f32,
        x: *mut f32,
        normed: *mut f32,
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        ctx: *mut f32,
        mlp: *mut f32,
        table: *const i32,
    ) -> i32;
}

/// The padded shapes the device layer runs at: heads widened to what the tiled
/// attention takes, the MLP to a multiple of the GEMM's 32.
struct Padded {
    head_dim: usize,
    inter: usize,
}

impl Padded {
    fn new(cfg: &VisionConfig) -> Result<Self> {
        let d = cfg.hidden / cfg.heads;
        let head_dim = if d <= 64 {
            64
        } else if d <= 128 {
            128
        } else {
            return Err(DlmError::InvalidConfig(format!(
                "head size {d} is wider than the device attention takes"
            )));
        };
        if cfg.hidden % 32 != 0 {
            return Err(DlmError::InvalidConfig(format!(
                "hidden size {} is not a multiple of 32",
                cfg.hidden
            )));
        }
        Ok(Self {
            head_dim,
            inter: cfg.intermediate.div_ceil(32) * 32,
        })
    }
}

/// `[heads * d, in]` rows (with their bias) spread to `[heads * dp, in]`.
fn pad_heads(w: &[f32], b: &[f32], heads: usize, d: usize, dp: usize) -> (Vec<f32>, Vec<f32>) {
    let k = w.len() / b.len();
    let (mut pw, mut pb) = (vec![0.0f32; heads * dp * k], vec![0.0f32; heads * dp]);
    for h in 0..heads {
        for j in 0..d {
            let (src, dst) = (h * d + j, h * dp + j);
            pw[dst * k..(dst + 1) * k].copy_from_slice(&w[src * k..(src + 1) * k]);
            pb[dst] = b[src];
        }
    }
    (pw, pb)
}

/// `[out, in]` with its input columns `src` spread to `dst` (zeros elsewhere).
fn pad_columns(
    w: &[f32],
    out: usize,
    cols: usize,
    new_cols: usize,
    map: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut p = vec![0.0f32; out * new_cols];
    for r in 0..out {
        for c in 0..cols {
            p[r * new_cols + map(c)] = w[r * cols + c];
        }
    }
    p
}

/// Run every encoder layer of `tower` over `x` (`[patches, hidden]`) in place.
pub(super) fn encode_layers(tower: &VisionTower, x: &mut [f32]) -> Result<()> {
    let cfg = &tower.cfg;
    let pad = Padded::new(cfg)?;
    let (h, n) = (cfg.hidden, x.len() / cfg.hidden);
    let d = h / cfg.heads;
    let qd = cfg.heads * pad.head_dim;
    let d_x = DeviceBuffer::from_slice(x)?;
    let normed = DeviceBuffer::new(n * h)?;
    let [q, k, v, ctx] = [
        DeviceBuffer::new(n * qd)?,
        DeviceBuffer::new(n * qd)?,
        DeviceBuffer::new(n * qd)?,
        DeviceBuffer::new(n * qd)?,
    ];
    let mlp = DeviceBuffer::new(n * pad.inter)?;
    let table = DeviceBuffer::from_bytes(&0i32.to_le_bytes(), 1)?;
    for i in 0..cfg.layers {
        let VitLayer {
            ln1,
            q: wq,
            k: wk,
            v: wv,
            o: wo,
            ln2,
            fc1,
            fc2,
        } = tower.layer(i)?;
        let heads = |(w, b): &(Vec<f32>, Vec<f32>)| pad_heads(w, b, cfg.heads, d, pad.head_dim);
        let (wq, wk, wv) = (heads(&wq), heads(&wk), heads(&wv));
        let wo_p = pad_columns(&wo.0, h, h, qd, |c| (c / d) * pad.head_dim + c % d);
        let mut b1 = fc1.1.clone();
        b1.resize(pad.inter, 0.0);
        let mut w1 = fc1.0.clone();
        w1.resize(pad.inter * h, 0.0);
        let w2 = pad_columns(&fc2.0, h, cfg.intermediate, pad.inter, |c| c);
        let up = |v: &[f32]| DeviceBuffer::from_slice(v);
        let dev = [
            up(&ln1.0)?,
            up(&ln1.1)?,
            up(&wq.0)?,
            up(&wq.1)?,
            up(&wk.0)?,
            up(&wk.1)?,
            up(&wv.0)?,
            up(&wv.1)?,
            up(&wo_p)?,
            up(&wo.1)?,
            up(&ln2.0)?,
            up(&ln2.1)?,
            up(&w1)?,
            up(&b1)?,
            up(&w2)?,
            up(&fc2.1)?,
        ];
        let p = |i: usize| dev[i].as_ptr();
        // SAFETY: every buffer is live and sized as `dlm_vit_block` documents:
        // weights padded above, activations allocated for `n` patches.
        let code = unsafe {
            dlm_vit_block(
                n as i32,
                h as i32,
                cfg.heads as i32,
                pad.head_dim as i32,
                pad.inter as i32,
                cfg.eps,
                1.0 / (d as f32).sqrt(),
                p(0),
                p(1),
                p(2),
                p(3),
                p(4),
                p(5),
                p(6),
                p(7),
                p(8),
                p(9),
                p(10),
                p(11),
                p(12),
                p(13),
                p(14),
                p(15),
                d_x.as_mut_ptr(),
                normed.as_mut_ptr(),
                q.as_mut_ptr(),
                k.as_mut_ptr(),
                v.as_mut_ptr(),
                ctx.as_mut_ptr(),
                mlp.as_mut_ptr(),
                table.as_ptr() as *const i32,
            )
        };
        if code != 0 {
            return Err(DlmError::Gpu {
                api: "dlm_vit_block",
                code,
            });
        }
        // The layer's weights are freed here; wait for the kernels reading
        // them first.
        crate::gpu::device::synchronize()?;
    }
    d_x.download(x)
}
