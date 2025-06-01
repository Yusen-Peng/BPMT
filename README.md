# BPMT: Body Part as Modality Transformer for Efficient and Accurate Human Action and Gait Recognition

In this thesis project, I aim to design BPMT, Body Part as Modality Transformer, which can achieve efficient and accurate (i) Human Action Recognition and (ii) Human Gait Recognition, respectively.

## Baseline results

| dataset | #videos | #actions | dimension | #joints | outperform SoTA? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action (2013) | 2,326 | 15 | 2D | 13 | yes, **94.66%** > 93.4% (HDM-BG) |
| N-UCLA (2014) | 1,494 | 12 | 3D | 20 | not yet, **88.79%** < 98.3% (SkateFormer) - cross view |
| NTU RGB+D (2016) | 56,880 | 60 | 3D | 25 | not yet, **73.56%** << 92.6% (SkateFormer) - cross subject |
| NTU RGB+D (2016) | 56,880 | 60 | 3D | 25 | N/A < 92.6% (SkateFormer) - cross view |
| NTU RGB+D 120 (2019) | 114,480 | 120 | 3D | 25 | N/A < 87.7%  (SkateFormer) - cross subject |
| NTU RGB+D 120 (2019) | 114,480 | 120 | 3D | 25 | N/A < 89.3%  (SkateFormer) - cross view |
| Skeletics-152 (2020) | 125,657 | 152 | 3D | 25 | N/A < 56.39% (MS-G3D) |

## current issue with Skeletics-152

NOTE: can't keep Skeletics-152 dataset for now: disk will be full!!!!

## Issues with other datasets

Why other datasets are not considered because:

1. FineGym: three-level hierarchy (actions + sub-actions)
2. Kinetics (skeleton): too many videos: 500,000 - maybe wait for a sec

## Baseline - (TLCA: Transfer Learning with Cross Attention) 

Pretraining:
![alt text](docs/baseline_pretraining_classification.png)

Cascading Finetuning:
![alt text](docs/baseline_finetuning_classification.png)

## BPMT 1.0 (Action Recognition)

Baseline Transformer (T1 and T2):
![alt text](docs/baseline_transformer.png)

First-stage pretraining:
![alt text](docs/first_stage.png)

Second-stage pretraining:
![alt text](docs/second_stage.png)

Finetuning:
![alt text](docs/finetuning_classification.png)

## Baseline - Experiment (Penn Action) - ALREADY OUTPERFORM STATE-OF-THE-ART (94.66%)

3DA (best) with Pr-VIPE, UNIK, HDM-BG, 3D Deep, PoseMap, MultitaskCNN, STAR: 
![alt text](docs/3D_deformable_transformer.png)

The current best training setup (95%-5% train-val split):
| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | clf-acc | 
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | **1000** | 1e-5, wd=1e-4 | **1000** | **94.66%** |


The complete experiment tuning logs:

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | clf-acc | 
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| <tr><td colspan="11" align="center">Complete Experiments, 15% held-out validation (n = 2326)</td></tr> |
| no | linear | 64 | 4 | 2 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 84.93% |
| no | linear | 64 | 4 | 2 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 85.96% |
| no | linear | 64 | 4 | 2 | finetune layer #2 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 87.36% |
| <tr><td colspan="11" align="center">Complete Experiments. 5% held-out validation (n = 2326)</td></tr> |
| no | linear | 64 | 4 | 2 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 83.71% |
| no | linear | 64 | 4 | 2 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 86.89% |
| no | linear | 64 | 4 | 2 | finetune layer #2 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 88.95% |
| no | linear | 256 | 8 | 4 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 85.11% |
| no | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 200 | 89.70% |
| no | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | **400** | **91.10%** |
| no | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | **500** | **91.01%** |
| no | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 700 | 89.42% |
| no | linear | 256 | 8 | 4 | finetune layer #4 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 400 | 86.89% |
| no | linear | 256 | 8 | 4 | finetune layer #4 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 490 | 88.39% |
| no | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 600 | 89.51% |
| no | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 700 | 90.45% |
| no | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 1000 | 89.89% |
| <tr><td colspan="11" align="center"> 30% masked pretraining - random **frames** </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 87.17% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | 87.27% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | **89.23%** | 
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1500 | 87.73% | 
| 30% | MLP | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 86.89% |
| 30% | MLP | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 87.45% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1K | 1e-5, wd=1e-4 | 500 | 89.70% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1K | 1e-5, wd=1e-4 | 1000 | **89.98%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 2K | 1e-5, wd=1e-4 | 1000 | 87.36% |
| <tr><td colspan="11" align="center"> *cosine scheduler* didn't improve the performance... </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 500 | 86.99% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 1000 | **88.20%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 1200 | 87.92% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 800 | 86.52% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 1000 | 86.80% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 1500 | 86.80% |
| <tr><td colspan="11" align="center"> 40% masked pretraining - random frames </td></tr> |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 400 | 87.55% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | **88.20%** |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 600 | 87.36% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | 87.83% |
| <tr><td colspan="11" align="center"> 20% masked pretraining - random frames  </td></tr> |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 400 | 87.73% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 88.67% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | **89.04%** |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 88.30% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 88.39% |
| <tr><td colspan="11" align="center"> 30% masked pretraining - random global joints  </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 200 | 1e-5, wd=1e-4 | 200 | 91.48% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 200 | 1e-5, wd=1e-4 | 500 | 91.01% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 200 | 91.67% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 300 | 92.42% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 350 | 91.76% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 400 | 91.95% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 500 | 91.39% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 200 | 91.48% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 300 | 91.57% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 91.57% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 200 | 91.20% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 300 | 92.88% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 400 | 92.42% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 500 | 91.57% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 600 | 92.13% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 900 | 92.42% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 600 | 1e-5, wd=1e-4 | 1000 | 91.95% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | **1000** | 1e-5, wd=1e-4 | **1000** | **94.66%** |
| <tr><td colspan="11" align="center"> *ablation study*: **too many layers** can cause overfitting... </td></tr> |
| 30% | linear | 512 | 8 | 8 | no | 1e-5 | 300 | 1e-5, wd=1e-4 | 100 | 89.89% |
| 30% | linear | 512 | 8 | 8 | no | 1e-5 | 300 | 1e-5, wd=1e-4 | 300 | 86.52% |

## Baseline - Experiment (NTU RGB+D 60)

Many folks have done data augmentation (still using only skeleton data):

1. E1: joint modality only
2. E2: joint + bone modalities
3. E4: joint + bone + joint motion + bone motion modalities

People usually: 

*train separate networks for each modality and ensemble their outputs*

![alt text](docs/NTU_comparison.png)

The current best training setup (95%-5% train-val split):

cross-subject evaluation:

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | clf-acc | 
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 300 | **73.56%** |


## The complete experiment tuning logs

cross-subject evaluation:
| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|--------------------|---------|---------|--------|------------|------------|--------|----------|----------------|----------|----------|
| <tr><td colspan="11" align="center"> multiple bodies: **skipping** </td></tr> |
| <tr><td colspan="11" align="center"> **regular** pretraining </td></tr> |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 20 | 70.19% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 50 | 70.46% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 50 | 71.33% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 100 | 71.91% |
| <tr><td colspan="11" align="center"> 30% masked pretraining - random frames </td></tr>  |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 50 | 70.08% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 100 | 70.45% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 300 | 71.67% |
| <tr><td colspan="11" align="center"> 30% masked pretraining - random global joints </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 100 | 71.27% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 200 | 70.24% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 300 | 72.65% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 500 | 72.31% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 100 | 72.32% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 71.99% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 100 | 72.33% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 300 | **73.21%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 500 | 72.94% |
| <tr><td colspan="11" align="center"> strong backbone </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | **1000** | 3e-5, wd=1e-4, cosine + warmup | 300 | **73.56%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | **1000** | 3e-5, wd=1e-4, cosine + warmup | 500 | 73.17% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | **1000** | 3e-5, wd=1e-4, cosine + warmup | 1000 | 72.51% |



| <tr><td colspan="11" align="center"> larger model </td></tr> |


| 30% | linear | 512 | 8 | 8 | no | 1e-4 | 200 | 1e-5, wd=1e-4 | 200 | ? |




| <tr><td colspan="11" align="center"> cross-view evaluation </td></tr> |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 100 | Need to run one here... |


Attempted to use SkateFormer data loader instead of my own data loader - bad performance!

1. (really) comprehensive regularization
    1. Shear (deformation)
    2. rotate
    3. scale
    4. spatial flip
    5. temporal flip
    6. Gaussian noise
    7. Gaussian blur
    8. drop axis
    9. drop joint

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|--------------------|---------|---------|--------|------------|------------|--------|----------|----------------|----------|----------|
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100, reg: '123456789ab' | 1e-5, wd=1e-4 | 100, reg: '128' | 64.29% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100, reg: '23689' | 1e-5, wd=1e-4 | 100, reg: '23689' | 63.93% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100, reg: '23689' | 1e-5, wd=1e-4 | 100, reg: '' | 65.66% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100, reg: '23689' | 1e-5, wd=1e-4 | 200, reg: '' | 65.89% |

## Baseline - Experiment (NW-UCLA, cross-view)


![alt text](docs/NTU_comparison.png)

### potential bottleneck for N-UCLA

1. No Hip-Centering or Normalization
2. model is relatively small (hidden size = 256, only 4 layers)
3. using the first body when multiple people appear on the same frame
4. no data augmentation

The following bottleneck is ***under discussion***:

1. is (T, 20, 3) -> (T × 20, 3) better than -> (T, 20X3)?



The current best training setup (95%-5% train-val split):

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 300 | **88.79%** |



## The complete experiment tuning logs:

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|--------------------|---------|---------|--------|------------|------------|--------|----------|----------------|----------|----------|
| <tr><td colspan="11" align="center"> minimal backbone </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 3e-5, wd=1e-4, cosine + warmup | 100 | 57.97% (100 epochs pretraining is too weak) |
| <tr><td colspan="11" align="center"> medium backbone </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 50 | 65.95% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 63.15% |
| 30% | linear | 256 | 8 | 4 | yes | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 65.73% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 20 | 65.73% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 30 | 69.40% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 50 | **69.40%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 100 | 67.03% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 500 | 61.21% |
| <tr><td colspan="11" align="center"> strong backbone </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 50 | 65.73% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 100 | 66.59% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 200 | **69.83%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 300 | 66.16% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 400 | 64.44% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 500 | 69.18% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 600 | 65.52% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 800 | 66.59% |
| <tr><td colspan="11" align="center"> hip-centering + Z-normalization </td></tr> |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 200 | 63.79% |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 250 | running |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 300 | **71.55%** |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 400 | **70.47%** |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 500 | 69.18% |
| 30%, z-norm | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 1000 | 60.99% |

Use SkateFormer data loader instead of my own data loader because:

1. normalization: first-joint centering; min-max normalization
2. regularization: random rotation, random scaling, random sampling, random dropout (joint/axis)
3. data augmentation: dataset duplication (repeat N=10 times)

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|--------------------|---------|---------|--------|------------|------------|--------|----------|----------------|----------|----------|
| <tr><td colspan="11" align="center"> small backbone </td></tr> |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 3e-5, wd=1e-4, cosine + warmup | 20 | 84.48% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 3e-5, wd=1e-4, cosine + warmup | 50 | 86.64% | 
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 3e-5, wd=1e-4, cosine + warmup | 100 | 85.13% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 3e-5, wd=1e-4, cosine + warmup | 200 | 84.27% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 200 | 3e-5, wd=1e-4, cosine + warmup | 100 | 85.56% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 200 | 3e-5, wd=1e-4, cosine + warmup | 200 | 87.28% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 200 | 3e-5, wd=1e-4, cosine + warmup | 300 | 85.13% |
| <tr><td colspan="11" align="center"> medium backbone </td></tr> |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 100 | 84.05% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 200 | 84.05%  |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 500 | **87.93%** |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 3e-5, wd=1e-4, cosine + warmup | 1000 | 85.99% |
| <tr><td colspan="11" align="center"> strong backbone </td></tr> |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 200 | 86.21% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 300 | **88.79%** |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 500 | 88.15% |
| 30%, SF data loader | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 3e-5, wd=1e-4, cosine + warmup | 1000 | 87.72% |


## Baseline - Experiment (Skeletics-152, cross-view)

Current state of the art:

![alt text](docs/Benchmark_Results_on_Skeletics-152.png)


| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy |
|--------------------|---------|---------|--------|------------|------------|--------|----------|----------------|----------|----------|


## BPMT 1.0 - Experiment (Action Recognition)

TBD

# Human Gait Recognition: a open-set retrieval problem

The definition of "open-set retrieval" from Gait3D paper for gait recognition:

*"Given a **query** sequence, we measure its similarity between all sequences in the **gallery** set. Then a ranking list of the gallery set is returned by the descending order of the similarities. We report the average Rank-1 and Rank-5 identification rates over all query sequences. We also adopt the mean Average Precision (mAP) and mean Inverse Negative Penalty (mINP) [55] which consider the recall of multiple instances and hard samples."*

## Existing State-of-the-art

PoseGait, GaitGraph, GaitFormer, GaitPT:

![alt text](docs/results_gaitPT.png)

GaitGraph2, GaitTR, GPGait:

![alt text](docs/results_skeletonmap.png)

GaitDIF:

![alt text](docs/GaitDIF.png)

## Data Preprocessing of Gait3D: Camera-View-Aware Filtering

Camera-View-Aware Data Preprocessing:
![alt text](docs/camera-view-aware.png)

## Baseline - Design (Gait Recognition)

Baseline Transformer (T1 and T2):
![alt text](docs/baseline_transformer.png)

Pretraining:
![alt text](docs/baseline_pretraining_retrieval.png)

Finetuning:
![alt text](docs/baseline_finetuning_retrieval.png)

## BPMT 1.0 Design (Gait Recognition)

TBD

## Baseline - Experiment (Gait Recognition)
| #subject scanned | #subject actual | decoder | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | R1-acc (completely unseen people)|
|------------------|------------------|------------|------------|--------|-------------|-------------|--------|--------------------------|
| <tr><td colspan="10" align="center">Mini Experiments (n = 50/300)</td></tr> |
| 50 | 27 | linear | yes | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | 15.75% |
| 50 | 27 | linear | no  | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | 17.53% |
| 50 | 27 | linear | finetune layer #2 | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | 14.71% |
| 300 | 109 | linear | yes | 1e-4 | 5000 | 1e-5, wd=1e-4 | 500 | 12.54% |
| 300 | 109 | linear | no | 1e-4 | 5000 | 1e-5, wd=1e-4 | 30 | 6.24% |
| 300 | 109 | linear | finetune layer #2 | 1e-4 | 5000 | 1e-5, wd=1e-4 | 30 | 7.02% |
| <tr><td colspan="10" align="center">Complete Experiments (n = 3000)</td></tr> |
| 3000 | 3000 | linear | yes | 1e-5 | 500 | 1e-5, wd=1e-4 | 1000 | 2.08% |
| 3000 | 3000 | linear | no | 1e-5 | 500 | 1e-5, wd=1e-4 | 200 | 2.94% |
| 3000 | 3000 | linear | no | 1e-5 | 500 | 1e-5, wd=1e-4 | 500 | 2.24% |
| 3000 | 3000 | linear | no | 1e-5 | 500 | 1e-5, wd=1e-4 | **1000** | **3.57%** |
| 3000 | 3000 | linear | no | 1e-5 | 500 | 1e-5, wd=1e-4 | **1500** | **3.03%** |


## BPMT 1.0 Experiment (Gait Recognition)
| #subject scanned | #subject actual | decoder | freeze T1? | T1-lr | #epochs | freeze T2? | T1-lr | #epochs | ft-lr | ft-#epochs | R1-acc (seen people from training) | 
|------------------|------------------|------------|------------|--------|-------------|-------------|--------|-------------|----------------|--------------------|--------------|

# Misc.

## What "could" be the next step? (my brainstormed ideas...)

integrate IIP-Transformer and compare with the baseline (BPMT 2.0)

My Other ideas (from my past time series experience + NLP class):

1. try efficient attention mechanisms like FlowAttention, FlashAttention
2. dual encoder (noisy encoder + clean encoder) like DEPICT
3. add contrastive learning objective on top of cross attention like CLIP
4. add auxiliary classification objective (mix in fake samples) like DTCR
