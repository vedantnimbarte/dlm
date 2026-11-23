"""A tiny random Gemma 3 vision-language checkpoint (f16), an image, and what
transformers makes of them, as a test fixture: dlm must preprocess the image,
encode it and score a prompt around it the same.

usage: python tests/fixtures/make_gemma3_vision.py tests/fixtures/gemma3_vision

The geometry is the real model's in miniature: a SigLIP tower (patch 7 over a
28x28 image, so 16 patches, pooled 2x2 into 4 image tokens), a two-layer Gemma 3
text model with one windowed layer (window 6, shorter than the prompt) and one
global layer with linear RoPE scaling. The vision MLP is 48 wide, not a multiple
of 32, as the real 4304 is not.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import (Gemma3Config, Gemma3ForConditionalGeneration, Gemma3ImageProcessor,
                          Gemma3TextConfig, SiglipVisionConfig)

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(3)

text = dict(
    vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, head_dim=16, query_pre_attn_scalar=16,
    hidden_activation="gelu_pytorch_tanh", max_position_embeddings=64, rms_norm_eps=1e-6,
    rope_theta=10000.0, rope_local_base_freq=1000.0,
    rope_scaling={"rope_type": "linear", "factor": 2.0},
    sliding_window=6, layer_types=["sliding_attention", "full_attention"],
    bos_token_id=2, eos_token_id=1, pad_token_id=0, tie_word_embeddings=True,
)
vision = dict(
    hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=2,
    image_size=28, patch_size=7, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
    vision_use_head=False,
)
mm = dict(mm_tokens_per_image=4, boi_token_index=60, eoi_token_index=61, image_token_index=62)

config = Gemma3Config(text_config=Gemma3TextConfig(**text),
                      vision_config=SiglipVisionConfig(**vision), **mm)
config._attn_implementation = "eager"
model = Gemma3ForConditionalGeneration(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and p.dim() == 1 and "vision_tower" not in name:
            p.normal_(0.0, 0.3)        # Gemma's (1 + w) norms
        elif "layer_norm" in name or "layernorm" in name:
            p.normal_(1.0, 0.1) if name.endswith("weight") else p.normal_(0.0, 0.1)
        elif p.dim() >= 2:
            p.normal_(0.0, 0.2)
        else:
            p.normal_(0.0, 0.1)        # biases
        # Round to what the f16 checkpoint holds, so the reference is computed
        # from exactly the weights dlm reads.
        p.copy_(p.half().float())

sd = {k: v.half().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
# config.json as Google's checkpoints spell it (transformers 4.x keys), which is
# what dlm reads.
cfg_json = {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",
            "torch_dtype": "float16", **mm,
            "text_config": {"model_type": "gemma3_text", **text},
            "vision_config": {"model_type": "siglip_vision_model", **vision}}
json.dump(cfg_json, open(out / "config.json", "w"), indent=1)

# An image whose size is not the model's, so the resize is exercised both ways:
# 41 wide is shrunk (antialiased), 19 high is stretched.
rng = np.random.default_rng(7)
img = rng.integers(0, 256, size=(19, 41, 3), dtype=np.uint8)
img[:, :20] //= 2                       # some structure for the encoder to see
Image.fromarray(img).save(out / "image.png")
proc = Gemma3ImageProcessor(size={"height": 28, "width": 28})
pixel_values = proc(Image.open(out / "image.png"), return_tensors="pt")["pixel_values"]

# Prompt ids around the image: <bos>, text, <start_of_image>, the four image
# tokens, <end_of_image>, more text -- long enough that the window of 6 cuts
# into both the text before the image and the image itself.
ids = [2, 10, 11, 12, 13, 60, 62, 62, 62, 62, 61, 14, 15, 16, 17, 18, 19]
input_ids = torch.tensor([ids])
token_type_ids = (input_ids == 62).long()
with torch.no_grad():
    logits = model(input_ids=input_ids, pixel_values=pixel_values,
                   token_type_ids=token_type_ids).logits[0].float()
    image_embeds = model.model.get_image_features(pixel_values).pooler_output[0]
lp = torch.log_softmax(logits, dim=-1)
pv = pixel_values[0].numpy()
json.dump({
    "ids": ids,
    "token_logprobs": [round(lp[i, ids[i + 1]].item(), 7) for i in range(len(ids) - 1)],
    "next_logprobs": [round(x, 6) for x in lp[-1].tolist()],
    # pixel_values is (u8 - 127.5) / 127.5 of the resized image, so the bytes
    # say it exactly.
    "pixels_u8": [int(round(x)) for x in (pv * 127.5 + 127.5).reshape(-1).tolist()],
    "image_embeds": [round(x, 6) for x in image_embeds.reshape(-1).tolist()],
}, open(out / "reference.json", "w"))
print(out, sum(t.numel() * 2 for t in sd.values()), "bytes of weights")
