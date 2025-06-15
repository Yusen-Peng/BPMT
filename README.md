# CascadeFormer: Two-stage Cascading Transformer for Human Action Recognition

## CascadeFormer 1.0

### architecture

Pretraining:
![alt text](docs/baseline_pretraining_classification.png)

Cascading Finetuning:
![alt text](docs/baseline_finetuning_classification.png)

### result leaderboard - CascadeFormer 1.0

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **94.66%** > 93.4% (HDM-BG) |
| Penn Action, subtraction-bone | 2,326 | 15 | 2D | 13 | **92.32%** ~ 93.4% (HDM-BG) |
| Penn Action, concatenation-bone | 2,326 | 15 | 2D | 13 | **93.16%** ~ 93.4% (HDM-BG) |
| Penn Action, parameterization-bone | 2,326 | 15 | 2D | 13 | **93.91%** > 93.4% (HDM-BG) |
| N-UCLA | 1,494 | 12 | 3D | 20 | **88.79%** < 98.3% (SkateFormer) |
| N-UCLA, subtraction-bone | 1,494 | 12 | 3D | 20 | **85.56%** < 98.3% (SkateFormer) |
| N-UCLA, concatenation-bone | 1,494 | 12 | 3D | 20 | **88.15%** < 98.3% (SkateFormer) |
| NTU/CS | 56,880 | 60 | 3D | 25 | **75.22%** << 92.6% (SkateFormer) - cross subject |
| NTU/CS, subtraction-bone | 56,880 | 60 | 3D | 25 | **74.23%** << 92.6% (SkateFormer) - cross subject |
| NTU/CS, concatenation-bone | 56,880 | 60 | 3D | 25 | **73.81%** << 92.6% (SkateFormer) - cross subject |

corresponding model checkpoints:

1. Penn Action: 94.66% [google drive](https://drive.google.com/drive/folders/1Za50ZE9ZEKdEps_ZE-JIbatTpLuMW83k)
2. N-UCLA: 88.79% is not good enough
3. NTU/CS: 75.22% is not good enough
4. NTU/CV: TBD

## CascadeFormer 1.1 (convolution enhanced)

Pretraining:
![alt text](docs/cascadeformer_1_1_pretrain.png)

Cascading Finetuning:
![alt text](docs/cascadeformer_1_1_finetune.png)

### result leaderboard - CascadeFormer 1.1

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action | 2,326 | 15 | 2D | 13 | **94.10%** > 93.4% (HDM-BG) |
| N-UCLA | 1,494 | 12 | 3D | 20 | **91.16%** < 98.3% (SkateFormer) |
| NTU/CS | 56,880 | 60 | 3D | 25 | ??? << 92.6% (SkateFormer) |
| NTU/CV | 56,880 | 60 | 3D | 25 | ??? << 92.6% (SkateFormer) |

corresponding model checkpoints:

1. Penn Action: 94.10% [google drive](https://drive.google.com/drive/folders/1qbcT8DlhNyT3HgbM3j2aEQP2rSXoEJRS)
2. N-UCLA: 91.16% [google drive](https://drive.google.com/drive/folders/1b0IuO_XY-Gwv4RjS6gF9gPG36uvGwhha)
3. NTU/CS: TBD 
4. NTU/CV: TBD

Additional Note: 

1. N-UCLA training is very sensitive (1 epoch difference -> 5% performance gap!)

![alt text](docs/sensitive.png)

## Leaderboard - CascadeFormer 2.0

Baseline Transformer (T1 and T2):
![alt text](docs/baseline_transformer.png)

First-stage pretraining:
![alt text](docs/first_stage.png)

Second-stage pretraining:
![alt text](docs/second_stage.png)

Finetuning:
![alt text](docs/finetuning_classification.png)

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action, joint | 2,326 | 15 | 2D | 13 | ??? < 93.4% (HDM-BG) |
| N-UCLA, joint | 1,494 | 12 | 3D | 20 | ??? < 98.3% (SkateFormer) |
| NTU, joint | 56,880 | 60 | 3D | 25 | ??? << 92.6% (SkateFormer) - cross subject |

# Remaining potential datasets 

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| <tr><td colspan="6" align="center"> the remaining part of the leaderboard is not done yet... </td></tr> |
| NTU | 56,880 | 60 | 3D | 25 | N/A < 92.6% (SkateFormer) - cross view |
| NTU 120 | 114,480 | 120 | 3D | 25 | N/A < 87.7%  (SkateFormer) - cross subject |
| NTU 120 | 114,480 | 120 | 3D | 25 | N/A < 89.3%  (SkateFormer) - cross view |
| Skeletics-152 | 125,657 | 152 | 3D | 25 | N/A < 56.39% (MS-G3D) |
 
## Under Discussion

1. for NTU, batch-level padding (current setup) OR fixed-length sequence?

![alt text](docs/ntu_seq_length_distribution.png)
