# Data Availability

This repository does not include raw datasets or large image embedding files.

## Dataset and Paper Sources

- MedSumm / MMCQS paper: https://arxiv.org/abs/2401.01596
- MMQS / CLIPSyntel paper: https://arxiv.org/abs/2312.11541

## Download Links

- MMQS dataset: https://drive.google.com/file/d/161VG5I0M-H1iaWtTIF3Fdt5Jo9hFVBka/view?usp=sharing
- Image embedding for MMQS: https://drive.google.com/file/d/1jdJLcPQ9Tia1QSKEKWmejjWbITzqksch/view?usp=sharing

## Expected Local Files

For running VP-RGAD-PD on MMQS, place the following files under data/:

- data/multimodal_final_updated.csv
- data/vgg_image_vector.pt

The CSV file should contain at least the columns Question and Question_summ. The image embedding tensor should be aligned row-by-row with the CSV file and should have shape [num_samples, 768].

Large data files are intentionally excluded from git.
