# BPMT: Body Part as Modality Transformer for Efficient and Accurate Gait Recognition

## research motivation

Although a massive number of gait recognition models have been proposed, and multiple architectures including IIP-Transformer, STSA-Net, and IGFormer have experimented with dividing the body poses explicitly into multiple meaningful parts or segments, no existing work has attempted to integrate body-part-aware transformers into the two-phase masked pretraining framework proposed in OmniVec2, which
was originally designed to learn multi-modal representations to perform multiple tasks. In this work, we attempt to address the following problem: can we treat different human parts as different modalities to integrate body-part-aware transformers into the two-phase masked pretraining framework proposed in OmniVec2?


## BPMT pipeline

![alt text](docs/BPMT_pipeline.png)


## BPMT architecture

Our BPMT architecture design:
![alt text](docs/BPMT.png)

First-stage pretraining:
![alt text](docs/first_stage.png)

Second-stage pretraining:
![alt text](docs/second_stage.png)

Finetuning:
![alt text](docs/finetuning.png)


## Baseline Transformer

Baseline Transformer architecture:
![alt text](docs/baseline_transformer.png)



## Camera-View-Aware Data Preprocessing

Camera-View-Aware Data Preprocessing:
![alt text](docs/camera-view-aware.png)


## What can be the next step?

integrate IIP-Transformer and compare with the baseline!

My Other ideas (from my past time series experience + NLP class):

1. try efficient attention mechanisms like FlowAttention, FlashAttention
2. dual encoder (noisy encoder + clean encoder) like DEPICT
3. add contrastive learning objective on top of cross attention like CLIP
4. add auxiliary classification objective (mix in fake samples) like DTCR

## Conda environment setup

BPMT_env - stay tuned!

## Current issue: overfitting

✅ fixing the bug of not saving pretrained cross-attention module properly

✅ severe underfitting during pretraining & finetuning

✅ hyperparameter tuning - smaller learning rate

✅ hyperparameter tuning - weight decay

❌ hyperparameter tuning - cosine scheduler?

❌ no early stopping (save the best checkpoint needed)


## Existing Gait recognition evaluation results 

The following papers give a nice overview of POSE-ONLY gait recognition
"GaitPT: Skeletons Are All You Need For Gait Recognition"
"SkeletonGait: Gait Recognition Using Skeleton Maps"
"GaitRef: Gait Recognition with Refined Sequential Skeletons"



TODO: gather some comprehensive benchmark results on Gait3D

For example, in GaitPT, multiple evaluation metrics are used: rank-1 accuracy, rank-5 accuracy. it shows the following results:
![alt text](docs/existing_results.png)







## Our experiment tracker

| #subject scanned | #subject actual | freeze T1? | T1-lr | #epochs | freeze T2? | T1-lr | #epochs | ft-lr | ft-#epochs | R1-acc | R5-acc |
|------------------|------------------|------------|--------|-------------|-------------|--------|-------------|----------------|--------------------|--------------|--------------|
| 50 | 27 | yes | 1e-4 | 1000 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 130 | 26% | TBD |             
| 300 | 109 | yes | 1e-4 | 1000 | yes | 1e-4 | 1000 | 1e-5, wd=1e-4 | 130 | TBD | TBD |