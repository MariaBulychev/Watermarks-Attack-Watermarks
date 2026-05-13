# Watermarks Attack Watermarks: Re-Watermarking as a Generic Removal Strategy

This repository contains the code we used to:

1. **Embed** watermarks into images using 8 published watermarking methods.
2. **Forge** the watermarks (apply a different watermark on top of an
   already-watermarked image).
3. **Decode / detect** watermarks from clean and attacked images.
4. **Classify** which watermarking method was used (ConvNeXt-V2 Large).

All TPR / FPR / quality / robustness curves and tables in the paper are
computed using the **WAVES** benchmark framework
(<https://github.com/umd-huang-lab/WAVES>).
We forked WAVES and added support for the additional watermarking methods,
the `Pipeline Attack` evaluation tab, and per-image normalised-quality
plots; the helper utilities `dev/io.py`, `dev/find.py`, `dev/parse.py`,
`dev/eval.py`, `dev/aggregate.py` referenced from the decode scripts in
this repo come directly from WAVES.

> If you only want to reproduce the *plots*, install WAVES, run the
> `metric.py` / `pipeline_metrics.py` scripts on the JSON outputs produced
> by our decoders, and use the WAVES Gradio app for the figures.

---

## Repository layout

```
Watermarks-Attack-Watermarks/
├── README.md                         ← this file
├── watermark/                        ← embed each watermark into clean images
│   ├── generate_pixelseal.py
│   ├── generate_rosteals.py
│   ├── generate_stable_signature.py
│   ├── generate_stegastamp.py
│   ├── generate_tree_ring.py
│   ├── generate_videoseal.py
│   ├── generate_wam.py
│   └── generate_zodiac.py            ← ZoDiac as a watermark (offset 0)
├── attack/                           ← forgery attacks (re-watermark on top)
│   ├── forge_pixelseal.py
│   ├── forge_rosteals.py
│   ├── forge_stegastamp.py
│   ├── forge_videoseal.py
│   ├── forge_wam.py
│   └── forge_zodiac.py               ← ZoDiac as forgery (offset 2_000_000)
├── decode/                           ← detect / decode watermark
│   ├── decode_pixelseal.py
│   ├── decode_rosteals.py
│   ├── decode_stable_sig.py
│   ├── decode_stegastamp.py
│   ├── decode_tree_ring.py
│   ├── decode_videoseal.py
│   ├── decode_wam.py
│   └── decode_zodiac.py
├── classifier/                       ← ConvNeXt-V2 Large 9-class classifier
│   ├── train.py
│   ├── test_mscoco.py
│   ├── config.py
│   ├── dataset.py
│   └── requirements.txt
└── requirements/                     ← per-method environment requirements
    ├── README.md
    ├── tree_ring_env.txt   /  tree_ring_env.full.txt
    ├── waves_env.txt       /  waves_env.full.txt
    ├── zodiac_env.txt      /  zodiac_env.full.txt
    └── pixelseal_env.txt   /  pixelseal_env.full.txt
```

No SLURM/cluster scripts are included — all scripts are plain `python3`
entry points.

---

## Path configuration (read me first)

All scripts resolve external dependencies and data through **environment
variables** with sensible relative-path defaults:

| Env var               | Default                                              | What it points to |
|-----------------------|------------------------------------------------------|-------------------|
| `VIDEOSEAL_PATH`      | `./external/videoseal`                               | VideoSeal / PixelSeal repo |
| `WAM_PATH`            | `./external/watermark-anything`                      | WAM repo |
| `ROSTEALS_PATH`       | `./external/RoSteALS`                                | RoSteALS repo |
| `TREE_RING_PATH`      | `./external/tree-ring-watermark`                     | Tree-Ring repo |
| `TREE_RING_IMPL_PATH` | `./external/tree_ring_implementation`                | Tree-Ring optim_utils |
| `STABLE_SIG_PATH`     | `./external/stable_signature/hidden`                 | Stable Signature `hidden/` dir |
| `STABLE_SIG_CKPT`     | `./checkpoints/stable_signature/hidden_replicate.pth`| Stable-Sig encoder/decoder weights |
| `ZODIAC_PATH`         | `./external/ZoDiac`                                  | ZoDiac repo |
| `WAVES_PATH`          | `./external/WAVES`                                   | WAVES benchmark repo |
| `DATA_DIR`            | `./data`                                             | Image dataset root |
| `RESULTS_DIR`         | `./results`                                          | Decoded JSON output dir |
| `HF_HOME`             | `./cache/hf`                                         | HuggingFace cache |
| `TORCH_HOME`          | `./cache/torch`                                      | torch cache |

Either set them in your shell, or place clones / data at the default
relative locations and run scripts from the repo root.

---

## Unified CLI

All scripts in a folder share the same argument conventions:

### `watermark/` — embed
```
python watermark/generate_<method>.py \
    --input  <dir of clean images> \
    --output <dir for watermarked> \
    --mode   {fixed | rand} \         # 'fixed': single shared message
                                      # 'rand':  per-image random message
    [--limit N]
```

* `generate_tree_ring.py` and `generate_stable_signature.py` additionally
  accept `--prompts_file` (run Stable Diffusion at embed time).
* `generate_zodiac.py` additionally accepts `--seed-offset` (default `0`)
  and `--fixed-seed` (default `10`).

### `attack/` — forge
```
python attack/forge_<method>.py \
    --method   <victim watermark> \
    --dataset  {mscoco | diffusiondb | diffusiondb_2} \
    --mode     {fixed | rand} \
    [--limit N]
```

* `forge_zodiac.py` accepts `--input/--output` directly (does not use
  `--method/--dataset`) and defaults `--seed-offset 2_000_000` to avoid
  colliding with a victim ZoDiac watermark embedded at offset 0.

### `decode/` — detect
```
python decode/decode_<method>.py \
    --path        <image dir> \
    --num-images  N            \
    --device      cuda:0       \
    [--key <output-json-key>]
```

The output is written to `${RESULTS_DIR}/<dataset>/<dirname>-decode.json`
where `<dataset>` is the second-to-last path component of `--path`.

* `decode_zodiac.py` accepts `--rand --seed-offset N` to detect ZoDiac
  embedded with a non-zero offset.
* `decode_tree_ring.py` accepts `--seeds-json <path>` to use per-image
  seeds saved by `generate_tree_ring.py --mode rand`.

### `classifier/`
```
python classifier/train.py
python classifier/test_mscoco.py --data-root <dir> --model-dir <dir>
```

---

## Watermarking methods — where to clone the upstream implementations

Each script imports from the upstream repository via the `*_PATH` env var
listed above. Clone the corresponding repo and either set the env var or
place the clone at the default location.

| Method | Upstream repository | Notes |
|---|---|---|
| **Tree-Ring** | <https://github.com/YuxinWenRick/tree-ring-watermark> | Frequency-domain Stable Diffusion watermark. We also use this repo's `inverse_stable_diffusion.py` for Stable-Signature and ZoDiac. |
| **Stable Signature** | <https://github.com/facebookresearch/stable_signature> | Use the `hidden/` directory and place `hidden_replicate.pth` at `STABLE_SIG_CKPT`. |
| **StegaStamp** | <https://github.com/tancik/StegaStamp> | We use the **ONNX-converted** encoder/decoder for batch inference. Expects `${WAVES_PATH}/decoders/stega_stamp.onnx`. 100-bit BCH(t=5, prim_poly=137) payload. |
| **RoSteALS** | <https://github.com/TuBui/RoSteALS> | Includes BCH ECC encoding (100-bit packet). |
| **PixelSeal** | <https://github.com/facebookresearch/videoseal> | PixelSeal lives inside the VideoSeal repo. |
| **VideoSeal** | <https://github.com/facebookresearch/videoseal> | Same repo as PixelSeal; different model weights. |
| **WAM (Watermark-Anything)** | <https://github.com/facebookresearch/watermark-anything> | Multi-bit, region-aware. |
| **ZoDiac** | <https://github.com/zhuangqh/ZoDiac> | DDIM-inversion based. Requires Stable Diffusion 2.1-base (HF id `stabilityai/stable-diffusion-2-1-base`). |

### Other dependencies pulled in by the scripts

* **Stable Diffusion 2.1** (HF id `stabilityai/stable-diffusion-2-1`) for
  Tree-Ring / Stable-Signature inverse pipeline.
* **Stable Diffusion 2.1-base** (HF id `stabilityai/stable-diffusion-2-1-base`)
  for ZoDiac.
* **HuggingFace `diffusers`** for diffusion-based watermarks.
* **`bchlib`** for StegaStamp / RoSteALS BCH encoding.
* **`onnxruntime-gpu`** for StegaStamp.

---

## Datasets used in the paper

* **MS-COCO 2017** (<https://cocodataset.org>) — 5 000 randomly sampled
  prompts.
* **DiffusionDB** (<https://github.com/poloclub/diffusiondb>) —
  5 000 randomly sampled prompts; second subset (`diffusiondb_2`) of 500
  used for some forgery experiments.

The watermark generators take a directory of un-watermarked PNGs as
input. For diffusion-based methods (Tree-Ring, Stable-Signature,
ZoDiac), the input is the prompt list and Stable Diffusion is run at
embed time to synthesise watermarked images directly.

The actual image data is **not** included in this repository.

---

## End-to-end pipeline

```
prompts ─┐
         ▼
  watermark/generate_<m>.py    →   $DATA_DIR/main/<dataset>/<m>/*.png
         │
         ▼  (apply quality / removal / forgery attacks)
  attack/forge_<m>.py          →   $DATA_DIR/attacked/<dataset>/<attack>-<victim>/*.png
  WAVES attacks (regen_diff, rinse_*, regen_vae, rotation, …)
         │
         ▼
  decode/decode_<m>.py         →   $RESULTS_DIR/<dataset>/<attack>-<victim>-decode.json
         │
         ▼
  WAVES dev/aggregate.py       →   metric.json + figures
  classifier/test_mscoco.py    →   per-image method predictions (used by adaptive pipeline)
```

The decoders write a per-image JSON of distances / bit-strings / p-values
that WAVES consumes via `dev/parse.py:get_distances_from_json` and
`dev/eval.py:detection_perforamance`.

---

## Classifier (cl1 — 9 classes)

`classifier/` is a fine-tuned
[ConvNeXt-V2 Large 22k](https://huggingface.co/facebook/convnextv2-large-22k-224)
that distinguishes 9 classes:

```
0 pixelseal   1 rosteals    2 stable_sig
3 stegastamp  4 tree_ring   5 wam
6 zodiac      7 videoseal   8 real
```

Trained at 512×512 with label smoothing (0.1), cosine LR + warmup,
fp16, batch size 8 × grad-accum 2, lr 2e-5, early stopping (patience 5),
stratified 80/10/10 split. See `classifier/config.py` for the full
hyperparameter list and `classifier/requirements.txt` for dependencies.

```bash
cd classifier
python train.py
python test_mscoco.py --data-root <dir> --model-dir ./outputs/best_model
```

The classifier output JSON is consumed by the WAVES `Pipeline Attack`
tab and `pipeline/attack/pipeline_metrics.py` to compute final TPR /
classifier-accuracy numbers.

---

## How we used WAVES

We used the WAVES benchmark
(<https://github.com/umd-huang-lab/WAVES>; An et al., 2024) for **all
quality, robustness, and detection plots** in the paper:

* `dev/aggregate.py` — combines per-attack `*-decode.json` files into
  the master `metric.json` table.
* `dev/eval.py:detection_perforamance` — TPR@FPR / AUC / accuracy.
* `dev/quality.py` (PSNR, SSIM, LPIPS, FID, Aesthetic, CLIP-FID, …) —
  per-attack quality measurements.
* The Gradio `app.py` from our WAVES fork renders all figures.

We added the following on top of vanilla WAVES (forks not included
here for review-blinding; full diff will be released):

1. Method-specific decoders → `dev/parse.py` registry.
2. `pipeline_attack.py` + `pipeline_metrics.py` — adaptive attack that
   uses the classifier to pick the per-image attack.
3. The "Attack Comparison Clean" and "Pipeline Attack" Gradio tabs.
4. Per-image normalised quality degradation aggregation.

If you just want to reproduce a plot:

```bash
git clone https://github.com/umd-huang-lab/WAVES
cd WAVES
pip install -r requirements.txt
python -m metric --dataset diffusiondb --source pixelseal --attack rinse_4xDiff
```

with the JSON outputs produced by this repo's `decode/*.py`.

---

## How to watermark images

Pick a method and run the matching script in `watermark/`. Two modes are
available for every method:

* `--mode fixed` — embed the **same** message / seed in every image.
  Useful as a sanity-check baseline; the decoder compares against that one
  fixed message.
* `--mode rand`  — embed a **per-image** random message / seed. The
  per-image keys are written to `<output>/messages.json` (or, for
  diffusion-based methods, derived deterministically from `int(filename)`
  + `--seed-offset`). This is the setting used for all results in the
  paper.

### Bit-string / message-based watermarks (input: clean PNGs)
```bash
# PixelSeal (256-bit), VideoSeal (256-bit), WAM (32-bit),
# RoSteALS (100-bit BCH), StegaStamp (100-bit BCH-ONNX)
python watermark/generate_pixelseal.py  --input ./data/real --output ./data/pixelseal  --mode rand
python watermark/generate_videoseal.py  --input ./data/real --output ./data/videoseal  --mode rand
python watermark/generate_wam.py        --input ./data/real --output ./data/wam        --mode rand
python watermark/generate_rosteals.py   --input ./data/real --output ./data/rosteals   --mode rand
python watermark/generate_stegastamp.py --input ./data/real --output ./data/stegastamp --mode rand
```

### Diffusion-based watermarks (input: prompts JSON; SD synthesises the image)
```bash
# Tree-Ring (frequency-domain) and Stable-Signature (latent-decoder)
python watermark/generate_tree_ring.py        --prompts_file ./data/prompts.json --output ./data/tree_ring        --mode rand
python watermark/generate_stable_signature.py --prompts_file ./data/prompts.json --output ./data/stable_sig       --mode rand

# ZoDiac (DDIM-inversion based) — input is clean images
python watermark/generate_zodiac.py --input ./data/real --output ./data/zodiac --mode rand
```

## How to detect / decode watermarks

Every decoder in `decode/` shares the same flag set:

```bash
python decode/decode_<method>.py \
    --path        <image_dir>       \
    --num-images  500               \
    --device      cuda:0            \
    [--key <output-json-key>]
```

The decoder writes
`${RESULTS_DIR}/<dataset>/<dirname>-decode.json`, where `<dataset>` is the
second-to-last path component of `--path` and `<dirname>` is its last
component. The JSON is the input to WAVES `dev/aggregate.py` →
`metric.json`.

Examples:

```bash
# Detect PixelSeal in clean watermarked images
python decode/decode_pixelseal.py --path ./data/main/mscoco/pixelseal --num-images 500

# Detect ZoDiac under rand-mode embed (filename-derived seed, offset 0)
python decode/decode_zodiac.py --path ./data/main/mscoco/zodiac --num-images 500 --rand --seed-offset 0

# Detect Tree-Ring with per-image seeds saved by generate_tree_ring.py --mode rand
python decode/decode_tree_ring.py --path ./data/main/mscoco/tree_ring --num-images 500 \
    --seeds-json ./data/main/mscoco/tree_ring/seeds.json
```

To decode the same watermark on **attacked** images, just point `--path`
at the attacked directory:

```bash
python decode/decode_pixelseal.py \
    --path ./data/attacked/mscoco/forgery_videoseal_corrected-1.0-pixelseal --num-images 500
```

The output JSON layout is fixed by WAVES (`dev/parse.py`); the resulting
files can be combined and analysed with the WAVES Gradio app.

---

## Full pipeline attack (adaptive forgery)

The pipeline attack is the *adaptive* attack used in the paper: instead
of forging with a single fixed watermark, we first **classify** the
watermark used on a victim image, then forge it with a **different**
method picked according to that prediction. The flow has three stages.

### Stage 1 — Classify the victim watermark

Use the 9-class ConvNeXt-V2 classifier in `classifier/` to predict which
watermarking method is present in each image:

```bash
# Train once on the 9-class dataset (pixelseal / rosteals / stable_sig /
# stegastamp / tree_ring / wam / zodiac / videoseal / real)
python classifier/train.py

# Run inference on a directory of victim images
python classifier/test_mscoco.py \
    --data-root ./data/main/mscoco \
    --model-dir ./outputs/best_model
```

This writes a per-image prediction file:
`./outputs/best_model/mscoco_predictions.json`

```json
{ "0.png": "pixelseal", "1.png": "rosteals", "2.png": "tree_ring", ... }
```

### Stage 2 — Apply the per-image forgery

For every image, we re-watermark it using a *different* method than the
one the classifier predicted. The `attack/forge_<method>.py` scripts each
forge with one specific method; the pipeline driver picks the script
per image based on the classifier output.

In the simplest case (pick a single forging method for all images of a
predicted class) you can run the per-method forge directly:

```bash
# Re-watermark every "pixelseal-predicted" image with VideoSeal
python attack/forge_videoseal.py  --method pixelseal --dataset mscoco --mode rand --limit 500
# Re-watermark every "rosteals-predicted" image with PixelSeal
python attack/forge_pixelseal.py  --method rosteals  --dataset mscoco --mode rand --limit 500
# … and so on for each predicted class.
```

For the full adaptive pipeline (one forging method routed per image),
use the WAVES fork's pipeline driver, which reads the classifier JSON
and dispatches to the per-method scripts:

```bash
python ${WAVES_PATH}/pipeline_attack.py \
    --predictions  ./outputs/best_model/mscoco_predictions.json \
    --dataset      mscoco \
    --routing      adaptive   \   # or 'fixed' / 'random' for the baselines
    --mode         rand
```

The driver writes attacked images to
`${DATA_DIR}/attacked/mscoco/pipeline-<routing>-<mode>/`.

### Stage 3 — Decode and evaluate

Run the *victim's* decoder on the attacked images and aggregate with
WAVES to get the final TPR / quality numbers:

```bash
# Decode with each of the 8 method-decoders
for m in pixelseal videoseal wam rosteals stegastamp tree_ring stable_sig zodiac; do
    python decode/decode_${m}.py \
        --path ./data/attacked/mscoco/pipeline-adaptive-rand --num-images 500
done

# Combine into the master metric.json and produce plots
python ${WAVES_PATH}/dev/aggregate.py --dataset mscoco
python ${WAVES_PATH}/pipeline_metrics.py \
    --dataset      mscoco \
    --predictions  ./outputs/best_model/mscoco_predictions.json \
    --routing      adaptive
```

`pipeline_metrics.py` reports the *pipeline TPR* — i.e. the fraction of
victim images for which **any** of the 8 watermarks survives the
classifier-routed forgery — together with the matched classifier
accuracy. These are the "Pipeline Attack" rows in the paper's tables and
the corresponding tab in the WAVES Gradio app.

---

## Quick reproduction recipe

```bash
# 1. Clone the upstream method repos:
mkdir -p external
git clone https://github.com/facebookresearch/videoseal           external/videoseal
git clone https://github.com/facebookresearch/watermark-anything  external/watermark-anything
git clone https://github.com/TuBui/RoSteALS                       external/RoSteALS
git clone https://github.com/YuxinWenRick/tree-ring-watermark     external/tree-ring-watermark
git clone https://github.com/facebookresearch/stable_signature    external/stable_signature
git clone https://github.com/zhuangqh/ZoDiac                      external/ZoDiac
git clone https://github.com/umd-huang-lab/WAVES                  external/WAVES

# 2. Install the appropriate environment (see requirements/README.md).

# 3. Embed (rand-message mode):
python watermark/generate_pixelseal.py \
    --input ./data/real --output ./data/pixelseal --mode rand

# 4. Decode (clean):
python decode/decode_pixelseal.py --path ./data/pixelseal --num-images 500

# 5. Apply a forgery attack:
python attack/forge_videoseal.py --method pixelseal --dataset mscoco --mode rand

# 6. Decode the attacked images:
python decode/decode_pixelseal.py --path ./data/attacked/mscoco/videoseal_forgery-1.0-pixelseal

# 7. Aggregate via WAVES:
python external/WAVES/metric.py --dataset mscoco --source pixelseal
```

---

## Citing

If you use this code, please cite our preprint (details forthcoming) and WAVES:

@inproceedings{an2024waves,
  title = 	 {{WAVES}: Benchmarking the Robustness of Image Watermarks},
  author =       {An, Bang and Ding, Mucong and Rabbani, Tahseen and Agrawal, Aakriti and Xu, Yuancheng and Deng, Chenghao and Zhu, Sicheng and Mohamed, Abdirisak and Wen, Yuxin and Goldstein, Tom and Huang, Furong},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {1456--1492},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/an24a.html}
}

Plus the upstream papers for any watermarking method you reuse from
this repository (see the table above).

---

## License

Each upstream watermarking method retains its original license. The
glue-code and classifier code in this repository are released under the
MIT License.
