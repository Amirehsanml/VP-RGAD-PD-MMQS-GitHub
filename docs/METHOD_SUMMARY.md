# Method Summary

VP-RGAD-PD is a staged multimodal sequence-to-sequence framework built on a BART backbone for clinical question summarization.

## Stage 1
A BART-base model is fine-tuned on the MMQS training split using text only, producing the clinical backbone checkpoint.

## Stage 2
A lightweight dual-stream Visual Prefix Adapter processes two visual inputs in parallel: a 768-dimensional pre-computed image embedding and a raw-pixel image representation. Each stream is encoded separately, then fused to generate a sequence of learnable visual prefix tokens that are prepended to the BART encoder inputs.

## Stage 3
The Stage 2 model is further refined with label smoothing, sliding-window unlikelihood, and Reward-Guided Anti-Degeneration Preference Distillation (RGAD-PD) to reduce repetition and improve faithfulness.

## Evaluation
The final model is evaluated without RGAD inference reranking.
