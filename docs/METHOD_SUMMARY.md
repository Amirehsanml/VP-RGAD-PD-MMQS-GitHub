# Method Summary

VP-RGAD-PD is a staged multimodal sequence-to-sequence model.

## Stage 1

A BART-base backbone is fine-tuned on the MMQS training split.

## Stage 2

A Visual Prefix Adapter maps each 768-dimensional image embedding into a sequence of visual prefix tokens. These tokens are prepended to BART encoder embeddings.

## Stage 3

The visual-prefix model is further fine-tuned with label smoothing, sliding-window unlikelihood, and Reward-Guided Anti-Degeneration Preference Distillation.

The final model is evaluated without RGAD inference reranking.
