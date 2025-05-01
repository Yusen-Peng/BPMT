# BPMT: Body Part as Modality Transformer for Efficient and Accurate Human Action and Gait Recognition

In this thesis project, I aim to design BPMT, Body Part as Modality Transformer, which can achieve efficient and accurate (i) Human Action Recognition and (ii) Human Gait Recognition, respectively.

# Human Action Recognition: a close-set classification problem

Dataset zoo: (I am currently using Penn Action)

| dataset | #videos | #actions | dimension | available? |
| ------- | ------- | -------- | --------- | ---------- |
| **Penn Action** | 2,325 | 15 | **2D** | downloaded (3GB) |
| NTU RGB+D  | 56,880 | 60 | 3D | under request |
| Skeletics-152 | 122,621 | 152 | 3D | downloadable (over 50GB!) |


## Existing State-of-the-art

TBD

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

| #subject | decoder | freeze T1? | T1-lr | #epochs | T2-lr (ft-lr) | #epochs | clf-acc | 
|------------------|------------|------------|--------|-------------|-------------|--------|------------|
| <tr><td colspan="10" align="center">Mini Experiments (n = 50/300)</td></tr> |
| 50 | linear | yes | 1e-4 | 5000 | 1e-5, wd=1e-4 | 30 | TBD | 
| 50 | linear | no  | 1e-4 | 5000 | 1e-5, wd=1e-4 | 30 | TBD |
| 50 | linear | finetune layer #2 | 1e-4 | 5000 | 1e-5, wd=1e-4 | 30 | TBD |
| 300 | linear | yes | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | TBD |
| 300 | linear | no | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | TBD |
| 300 | linear | finetune layer #2 | 1e-4 | 5000 | 1e-5, wd=1e-4 | 100 | TBD |
| <tr><td colspan="10" align="center">Complete Experiments (n = 2325)</td></tr> |
| 3000 | linear | yes | 1e-5 | 500 | 1e-5, wd=1e-4 | 200 | TBD |

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

<!-- ## Baseline - Experiment (close-set classification) -->


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
| 3000 | 3000 | linear | yes | 1e-5 | 500 | 1e-5, wd=1e-4 | 200 | TBD |

<!-- ## BPMT 1.0 - Experiment (close-set classification)

| #subject scanned | #subject actual | decoder | freeze T1? | T1-lr | #epochs | freeze T2? | T1-lr | #epochs | ft-lr | ft-#epochs | clf-acc | 
|------------------|------------------|------------|------------|--------|-------------|-------------|--------|-------------|----------------|--------------------|--------------|
| 50 | 27 | linear | yes | 1e-4 | 1000 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 130 | 26.6% |
| 50 | 27 | linear | yes | 1e-4 | 1000 | no | 1e-4 | 1000 | 1e-5, wd=1e-4 | 130 | 25.9% |           
| 300 | 109 | linear | yes | 1e-4 | 1000 | yes | 1e-4 | 1000 | 1e-6, wd=1e-4 | 400 | 6% |
| 300 | 109 | linear | yes | 1e-4 | 1000 | yes | 1e-4 | 1000 | 1e-6, wd=1e-4 | 1000 | 7.35% |  -->

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
