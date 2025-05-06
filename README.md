# BPMT: Body Part as Modality Transformer for Efficient and Accurate Human Action and Gait Recognition

In this thesis project, I aim to design BPMT, Body Part as Modality Transformer, which can achieve efficient and accurate (i) Human Action Recognition and (ii) Human Gait Recognition, respectively.

# Human Action Recognition: a close-set classification problem

Dataset zoo: (I am currently using Penn Action)

| dataset | #videos | #actions | dimension | available? |
| ------- | ------- | -------- | --------- | ---------- |
| **Penn Action** (2013) | 2,326 | 15 | **2D** | downloaded (3GB) |
| NTU RGB+D  | 56,880 | 60 | 3D | under request |
| Skeletics-152 | 122,621 | 152 | 3D | downloadable (over 50GB!) |


## Existing State-of-the-art

3DA (best) with Pr-VIPE, UNIK, HDM-BG, 3D Deep, PoseMap, MultitaskCNN, STAR: 
![alt text](docs/3D_deformable_transformer.png)

## Baseline Design (Action Recognition)

Pretraining:
![alt text](docs/baseline_pretraining_classification.png)

Cascading Finetuning:
![alt text](docs/baseline_finetuning_classification.png)

## BPMT 1.0 - Design (Action Recognition)

Baseline Transformer (T1 and T2):
![alt text](docs/baseline_transformer.png)

First-stage pretraining:
![alt text](docs/first_stage.png)

Second-stage pretraining:
![alt text](docs/second_stage.png)

Finetuning:
![alt text](docs/finetuning_classification.png)

## Baseline - Experiment (Action Recognition)

| #subject | decoder | d_model | n_head | num_layers | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | clf-acc | 
|------------------|------------|------------|------------|------------|------------|--------|-------------|-------------|--------|------------|
| <tr><td colspan="10" align="center">Complete Experiments, 15% held-out validation (n = 2326)</td></tr> |
| 2326 | linear | 64 | 4 | 2 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 84.93% |
| 2326 | linear | 64 | 4 | 2 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 85.96% |
| 2326 | linear | 64 | 4 | 2 | finetune layer #2 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 87.36% |
| <tr><td colspan="10" align="center">Complete Experiments. 5% held-out validation (n = 2326)</td></tr> |
| 2326 | linear | 64 | 4 | 2 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 83.71% |
| 2326 | linear | 64 | 4 | 2 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 86.89% |
| 2326 | linear | 64 | 4 | 2 | finetune layer #2 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 88.95% |
| 2326 | linear | 256 | 8 | 4 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 500 | 85.11% |
| 2326 | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 200 | 89.70% |
| 2326 | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | **400** | **91.10%** |
| 2326 | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | **500** | **91.01%** |
| 2326 | linear | 256 | 8 | 4 | no  | 1e-4 | 1000 | 1e-5, wd=1e-4 | 700 | 89.42% |
| 2326 | linear | 256 | 8 | 4 | finetune layer #4 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 400 | 86.89% |
| 2326 | linear | 256 | 8 | 4 | finetune layer #4 | 1e-4 | 1000 | 1e-5, wd=1e-4 | 490 | 88.39% |
| 2326 | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 600 | 89.51% |
| 2326 | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | **700** | **90.45%** |
| 2326 | MLP | 256 | 8 | 4 | no  | 1e-5 | 1000 | 1e-5, wd=1e-4 | 1000 | 89.89% |

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
