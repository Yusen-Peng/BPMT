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

❌ no early stopping
