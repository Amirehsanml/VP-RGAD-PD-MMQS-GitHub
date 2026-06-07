# VP-RGAD-PD for Multimodal Clinical Question Summarization

This repository contains the cleaned implementation of VP-RGAD-PD, a staged multimodal BART-based framework for multimodal clinical question summarization on MMQS.

## Final Model

The final selected model is G3 staged / VP-RGAD-PD.

Training strategy:

1. Stage 1: BART backbone adaptation. facebook/bart-base is fine-tuned on the MMQS training split.
2. Stage 2: Visual Prefix Adapter. Pre-extracted image embeddings are mapped into visual prefix tokens and prepended to BART encoder embeddings.
3. Stage 3: VP-RGAD-PD. The visual-prefix model is fine-tuned using label smoothing, sliding-window unlikelihood, and Reward-Guided Anti-Degeneration Preference Distillation.

## Final Test Result

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| VP-RGAD-PD | 57.0422 | 32.2543 | 48.7893 |

## Fixed Split

All stages use the same fixed split with seed 42.

| Split | Size |
|---|---:|
| Train | 2412 |
| Validation | 150 |
| Test | 453 |

Split index files are provided under splits/.

## Data

Raw datasets and image embeddings are not included in this repository.

See docs/DATA_AVAILABILITY.md for paper links, Google Drive dataset links, and expected local file names.

Expected local files:

- data/multimodal_final_updated.csv
- data/vgg_image_vector.pt

## Checkpoints

Model checkpoints are not included in git.

Expected local checkpoint paths after training:

- artifacts/checkpoints/BART_backbone_stage1/
- artifacts/checkpoints/G2_visual_prefix/
- artifacts/checkpoints/G3_VP_RGAD_PD_final/

Use GitHub Releases, Hugging Face Hub, or another external storage service if trained weights are distributed separately.

## Evaluation

After preparing the data and checkpoints locally:

```bash
bash scripts/eval_final_model.sh
```

## Final Decision

G4 ROUGE-aware preference fine-tuning was tested but did not improve held-out test performance, so the final model remains G3 staged / VP-RGAD-PD.

## Extra Evaluation Metrics

BLEU, METEOR, and BERTScore can be computed with:

```bash
python scripts/compute_test_extra_metrics.py --csv outputs/G3_test_no_rgad.csv --ref_col actual --pred_col predicted --bert_model microsoft/deberta-xlarge-mnli --device cuda
```
