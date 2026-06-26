# VP-RGAD-PD-MMQS

This repository contains the code for the **VP-RGAD-PD** framework for multimodal clinical question summarization.

## Overview

The project follows a three-stage training workflow:

**Stage 1 — Clinical Backbone Adaptation**  
Text-only adaptation of a pretrained BART backbone on the MMQS clinical summarization task.

**Stage 2 — Dual-Stream Visual Prefix Learning**  
A lightweight dual-stream visual adapter processes:
- a pre-computed visual embedding stream
- a raw-pixel visual stream

Each stream is processed separately, then fused into learnable visual prefix tokens that are prepended to the text embeddings before BART encoding.

**Stage 3 — Degeneration-Aware Refinement**  
The Stage 2 checkpoint is further refined with:
- label smoothing
- sliding-window unlikelihood
- RGAD preference distillation
- repetition-aware decoding constraints

## Repository Structure

- `code/` — Python source code for training and evaluation
- `scripts/` — Bash scripts for running stage-wise training and evaluation
- `docs/` — Notes and documentation
- `splits/` — Fixed train/validation/test split files for reproducibility
- `README.md` — Project overview and usage
- `requirements.txt` — Python dependencies

## What is intentionally not included

This repository does **not** include:
- raw dataset files
- trained checkpoints
- logs
- generated outputs
- intermediate experiments
- large artifacts or temporary files

Those files should be stored outside the repository and referenced through local paths.

## External Paths Expected by the Code

The code expects the following external resources to exist on disk:

- MMQS dataset CSV
- image folder for raw visual input
- pre-computed visual embedding files
- stage checkpoints produced during training

These paths are configured through environment variables in the run scripts.

## Training Workflow

### Stage 1
Train the text-only BART backbone on the MMQS dataset.

### Stage 2
Train the dual-stream visual prefix module using:
- pre-computed embeddings
- raw pixels from images

### Stage 3
Load the Stage 2 checkpoint and refine the model with degeneration-aware training objectives.

## Evaluation

Evaluation is performed with the trained Stage 3 checkpoint.  
Ablation comparisons may include:
- full multimodal input
- no image input
- shuffled embedding input

## Reproducibility

The `splits/` directory is kept in the repository so that the same train/validation/test partition can be reused across runs.

## Notes

- The main backbone for the paper is **BART**.
- The repository is organized to reflect the final architecture used in the paper.
- The code is intended to be run with local data and local checkpoints outside the repository.
