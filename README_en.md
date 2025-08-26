# 2025 Global AI Defense Challenge – Track 1: Image All-Factor Interactive Authentication – Generation (Implementation and Report)

This repository documents my implementation and report for the [2025 Global AI Defense Challenge – Track 1: Image All-Factor Interactive Authentication – Generation](https://tianchi.aliyun.com/competition/entrance/532389/information) competition.

## Intro
The generation track comprises four tasks, all centered on producing high‑quality **synthetic** images:
- T2I (Text‑to‑Image) — Generate images directly from textual prompts.
- TIE (Text‑Instructed Image Editing) — Edit a given image according to a natural‑language instruction.
- VTTIE (Vision‑Text‑To‑Image Editing) — Modify textual content within images (focus on text elements) guided by textual instructions.
- Deepfake (Face Swapping) — Transfer the target identity onto the origin face with high fidelity and realism.

My goal was to build a pipeline using only **open‑source** models covering generation, editing, as well as evaluation, then use the competition as a validation of the overall pipeline.

For detailed methodology, ablations, and reproducibility notes, see the code below and reports in the [docs](./docs) directory.

## Repo Structure
```text
root/
├── code/                 # Core code
│   ├── common/           # Utilities: translation, auto-scoring, helpers
│   ├── deepfake/         # Deepfake implementation (InsightFace + InSwapper, etc.)
│   ├── t2i/              # T2I implementation
│   ├── tie/              # TIE implementation
│   └── vttie/            # VTTIE implementation
│
├── configs/              # YAML configs per task
│   ├── deepfake.yaml
│   ├── t2i.yaml
│   ├── tie.yaml
│   └── vttie.yaml
│
├── data/                 # Inputs & intermediates
│   ├── imgs/             # Raw images from the competition
│   ├── imgs_deepfake_sr/ # Super-resolved images for Deepfake
│   ├── task.csv          # Original task table
│   ├── task_en.csv       # Translated task table
│   ├── prompt_map.csv    # Prompt mapping (translated)
│   └── translate_cache.json # Translation cache
│
├── results/              # Sample outputs (subsets)
│   ├── deepfake/
│   ├── t2i/
│   ├── tie/
│   └── vttie/
│
├── docs/                 # Reports & detailed write-ups
│   ├── report_zh.pdf     # Chinese report
│   └── report_en.pdf     # English report
│
├── README.md             # Chinese README
└── README_en.md          # English README
```

## Open-Source Models & Tools Used
### Task 1: T2I - Text-to-Image
- SDXL-base/refiner-1.0
- Qwen-Image
- Real-ESRGAN

### Task 2/3: TIE/VTTIE - Image Editing
- FLUX.1‑Kontext‑Dev
- Qwen‑Image‑Edit

### Task 4: Deepfake - Face Swapping
- InsightFace + InSwapper
- GFPGAN
- SDXL‑base‑1.0 + InstantID
- FLUX.1‑Kontext‑Dev

### Evaluation
- Qwen2.5‑VL‑72B‑Instruct

## Outro
Final [leaderboard](https://tianchi.aliyun.com/competition/entrance/532389/rankingList): 7/891 teams