# Environment Requirements

Each watermark method was originally implemented and run in its **own** isolated
Python environment because their upstream repos pin different `torch` /
`diffusers` / OpenCV versions. We provide both:

* **Minimal `*_env.txt`** — hand-curated minimal pip requirements per method.
  Use these as a starting point.
* **Full `*_env.full.txt`** — `pip freeze` of the actual environment we used,
  for full reproducibility (some entries are not needed for the scripts in
  this repo and may be safely pruned).

| Environment | Used by |
|---|---|
| `tree_ring_env`     | tree_ring + stable_signature watermarking, decoding, attack |
| `waves_env`         | pixelseal, videoseal, wam, rosteals, stegastamp watermarking, decoding, attack; classifier |
| `zodiac_env`        | ZoDiac watermarking, decoding, forgery attack |
| `pixelseal_env`     | (legacy) standalone pixelseal env |

Quick install commands:

```bash
# Tree-Ring & Stable Signature (also used for stable_sig encode/decode)
python3.9 -m venv tree_ring_env
source tree_ring_env/bin/activate
pip install -r tree_ring_env.txt

# WAVES / general (PixelSeal, VideoSeal, WAM, RoSteALS, StegaStamp, classifier)
python3.10 -m venv waves_env
source waves_env/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r waves_env.txt

# ZoDiac (Stable Diffusion 2.1-base)
conda create -n zodiac python=3.9
conda activate zodiac
pip install -r zodiac_env.txt
```
