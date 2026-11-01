# RECAP-Leaf

Capture-risk-guided recapture, guarded test-time adaptation and split-calibrated abstention for few-shot field diagnosis of crop diseases with a distilled edge model.

This repository contains every script used to produce the results, tables and figures of the accompanying manuscript. All files sit in one folder; run them from that folder.

## What the code does

| Stage | Script | Output (in `work/`) |
|---|---|---|
| Data manifest (PlantVillage, PlantDoc, MangoLeafBD, lemon leaf) | `prepare_data.py DATA_DIR` | `DATA_DIR/manifest.csv`, resized images |
| PlantDoc scenes for the capture benchmark | `prepare_scenes.py DATA_DIR` | `DATA_DIR/scenes.json` |
| GrabCut leaf masks for regional leaves | `leaf_masks.py DATA_DIR/manifest.csv` | `*_mask.png`, `regional_subset.txt` |
| Unlabelled non-target distillation corpus | `prepare_unlabeled.py DATA_DIR` | `DATA_DIR/unlabeled.txt` |
| Fixed evaluation sets and per-image statistics | `embed_sets.py` | `sets.json`, `image_stats.npz`, `regf/` |
| Student distillation (frozen DINOv2-S/14 teacher) | `train_student.py SEED EPOCHS [--labonly]` | `student_s*.pt` |
| Cross-entropy baselines | `train_baselines.py ARCH SEED EPOCHS` | `base_*.pt` |
| Embeddings | `embed.py MODEL SEED` | `emb_*.npz` |
| E1/E2 recognition | `exp_classification.py` | `results/cls_runs_all.json` |
| E3 capture benchmark frame bank | `recapture_frames.py BACKBONE SEED` | `recap_bank_*.npz`, `recap_index_*.json` |
| E3/E5 recapture policies | `exp_recapture.py BACKBONE SEED` | `results/recap_*.json` |
| E4 test-time adaptation | `exp_tta.py SEED SPLIT` (partitions 0–2) | `results/tta_*.json` |
| E5 acceptance and validity | `exp_abstention.py` | `results/abstention.json` |
| E6 efficiency | `exp_efficiency.py` | `results/efficiency.json` |
| Statistics and summary | `analyze.py` | `results/summary.json` |
| Figures | `fig_arch.py`, `fig_capture.py OUT`, `fig_results.py SUMMARY OUT` | PNG figures |

Core modules: `recap.py` (student, distillation loss, capture-risk model, Learn-then-Test threshold, normalisation updates), `capture.py` (capture-defect emulation and the five capture factors), `evalkit.py` (partitions, prototypes, metrics), `common.py` (paths, label space, loading).

## Data

All data are public and are not redistributed here.

* PlantVillage colour images: `git clone https://github.com/spMohanty/PlantVillage-Dataset` (only `raw/color` of bell pepper, potato, maize and tomato is used).
* PlantDoc classification and detection sets: `https://github.com/pratikkayal/PlantDoc-Dataset`, `https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset`.
* MangoLeafBD, lemon leaf (raw) and iNatAg-mini species archives: Hugging Face, organisation `Project-AgML`.

Place them under one data directory (`data/pv`, `data/plantdoc_cls`, `data/plantdoc_det`, `data/mango.parquet`, `data/lemon.parquet`, `data/inat/*.zip`) and set `RECAP_DATA` and `RECAP_WORK` if the defaults (`../data`, `../work` relative to this folder) do not suit.

## Running

```
pip install -r requirements.txt
bash run_all.sh            # full pipeline, resumable; skips stages whose outputs exist
```

The experiments were run on a two-core x86 CPU without a GPU (PyTorch 2.15, timm 1.0.30, OpenCV 5.0). Student distillation takes roughly one to two hours per seed on such a machine; the remaining stages take minutes to tens of minutes each.

## Reproducibility notes

* Partitions are seeded (`evalkit.splits`), and the capture benchmark is deterministic: every capture state is derived from a hash of the item and the action path.
* No field or regional test image is used for training, model selection or hyperparameters. Calibration data are used only for the temperature, the risk-model weights (half A) and the acceptance threshold (half B).
* Each trained model was trained once (seed 0); variability is reported over the five data partitions.
* Latencies are host-CPU measurements, not single-board-computer measurements.
* `fig_arch.py` writes `fig1_overview.png` to the current directory.
