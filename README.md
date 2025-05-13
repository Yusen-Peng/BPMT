# BPMT: Body Part as Modality Transformer for Efficient and Accurate Human Action and Gait Recognition

In this thesis project, I aim to design BPMT, Body Part as Modality Transformer, which can achieve efficient and accurate (i) Human Action Recognition and (ii) Human Gait Recognition, respectively.

# Human Action Recognition: a close-set classification problem

Dataset zoo: (I am currently using Penn Action)

| dataset | #videos | #actions | dimension | #joints | available? |
| ------- | ------- | -------- | --------- | ---------- | ------- |
| Penn Action (2013) | 2,326 | 15 | 2D | 13 | downloaded (3GB) |
| NTU RGB+D (2016) | 56,880 | 60 | 3D | 25 | downloaded (6GB) |
| NTU RGB+D 120 (2019) | ?? | 120 | 3d | ?? | downloadable |
| Skeletics-152 | 122,621 | 152 | 3D | ?? | downloadable (over 50GB!) |

## TLCA: Transfer Learning with Cross Attention 

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

## TLCA - Experiment (Penn Action Dataset)

3DA (best) with Pr-VIPE, UNIK, HDM-BG, 3D Deep, PoseMap, MultitaskCNN, STAR: 
![alt text](docs/3D_deformable_transformer.png)

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
| no | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | **700** | **90.45%** |
| no | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 1000 | 89.89% |
| <tr><td colspan="11" align="center"> now, let's do **30%** masked pretraining </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 87.17% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | 87.27% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | **89.23%** | 
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1500 | 87.73% | 
| 30% | MLP | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 86.89% |
| 30% | MLP | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 87.45% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 89.70% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4 | 1000 | 1e-5, wd=1e-4 | 1000 | **89.98%** |
| <tr><td colspan="11" align="center"> *cosine scheduler* didn't improve the performance... </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 500 | 86.99% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 1000 | **88.20%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4 | 1200 | 87.92% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 800 | 86.52% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 1000 | 86.80% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 300 | 1e-5, wd=1e-4, epoch-cosine | 1500 | 86.80% |
| <tr><td colspan="11" align="center"> now, try **40%** masked pretraining instead </td></tr> |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 400 | 87.55% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | **88.20%** |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 600 | 87.36% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | 87.83% |
| <tr><td colspan="11" align="center"> now, try 20% masked pretraining instead </td></tr> |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 400 | 87.73% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 500 | 88.67% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 700 | **89.04%** |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 88.30% |
| 40% | linear | 256 | 8 | 4 | no | 1e-4 | 500 | 1e-5, wd=1e-4 | 1000 | 88.39% |

## TLCA - Experiment (NTU RGB+D dataset)

![alt text](docs/NTU_comparison.png)

| masked pretraining | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | accuracy | 
| <tr><td colspan="11" align="center"> cross-subject evaluation </td></tr> |
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| <tr><td colspan="11" align="center"> let's start with regular pretraining </td></tr> |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 20 | 70.19% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 100 | 1e-5, wd=1e-4 | 50 | 70.46% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 10 | 63.56% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 50 | 71.33% |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 100 | **71.91%** |
| <tr><td colspan="11" align="center"> let's do 30% masked pretraining now </td></tr> |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 100 | 1e-5, wd=1e-4 | 50 | 70.08% |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 100 | 1e-5, wd=1e-4 | 100 | **70.45%** |
| 30% | linear | 256 | 8 | 4 | no | 1e-4, batch-cosine | 100 | 1e-5, wd=1e-4 | 300 | running now |
| <tr><td colspan="11" align="center"> cross-view evaluation </td></tr> |
| no | linear | 256 | 8 | 4 | no | 1e-4 | 300 | 1e-5, wd=1e-4 | 100 | TBD |



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
