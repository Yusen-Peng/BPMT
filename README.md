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

### Training Loss & Validation Loss - 50 subjects

#### first stage pretraining

![alt text](figures_50/Torso_train_val_loss.png)

![alt text](figures_50/Left_Arm_train_val_loss.png)

![alt text](figures_50/Right_Arm_train_val_loss.png)

![alt text](figures_50/Left_Leg_train_val_loss.png)

![alt text](figures_50/Right_Arm_train_val_loss.png)

#### second stage pretraining

![alt text](figures_50/Torso_Left_Arm_train_val_loss.png)

![alt text](figures_50/Torso_Right_Arm_train_val_loss.png)

![alt text](figures_50/Torso_Left_Leg_train_val_loss.png)

![alt text](figures_50/Torso_Right_Leg_train_val_loss.png)

![alt text](figures_50/Left_Arm_Right_Arm_train_val_loss.png)

![alt text](figures_50/Left_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_50/Left_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_50/Right_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_50/Right_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_50/Left_Leg_Right_Leg_train_val_loss.png)


#### finetuning

Right now I am facing overfitting issues:

![alt text](figures_50/finetuning_train_val_loss.png)


### Training Loss & Validation Loss - 300 subjects

#### first stage pretraining

![alt text](figures_300/Torso_train_val_loss.png)

![alt text](figures_300/Left_Arm_train_val_loss.png)

![alt text](figures_300/Right_Arm_train_val_loss.png)

![alt text](figures_300/Left_Leg_train_val_loss.png)

![alt text](figures_300/Right_Arm_train_val_loss.png)

#### second stage pretraining

![alt text](figures_300/Torso_Left_Arm_train_val_loss.png)

![alt text](figures_300/Torso_Right_Arm_train_val_loss.png)

![alt text](figures_300/Torso_Left_Leg_train_val_loss.png)

![alt text](figures_300/Torso_Right_Leg_train_val_loss.png)

![alt text](figures_300/Left_Arm_Right_Arm_train_val_loss.png)

![alt text](figures_300/Left_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_300/Left_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_300/Right_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_300/Right_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_300/Left_Leg_Right_Leg_train_val_loss.png)


#### finetuning

Right now I am facing the same overfitting issues:

![alt text](figures_300/finetuning_train_val_loss.png)


### Training Loss & Validation Loss - 50 subjects with class-specific split

#### first stage pretraining

![alt text](figures_50_class_specific/Torso_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Arm_train_val_loss.png)

![alt text](figures_50_class_specific/Right_Arm_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Right_Arm_train_val_loss.png)

#### second stage pretraining

![alt text](figures_50_class_specific/Torso_Left_Arm_train_val_loss.png)

![alt text](figures_50_class_specific/Torso_Right_Arm_train_val_loss.png)

![alt text](figures_50_class_specific/Torso_Left_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Torso_Right_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Arm_Right_Arm_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Right_Arm_Left_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Right_Arm_Right_Leg_train_val_loss.png)

![alt text](figures_50_class_specific/Left_Leg_Right_Leg_train_val_loss.png)


#### finetuning

Right now I am facing the same overfitting issues:

![alt text](figures_50_class_specific/finetuning_train_val_loss.png)


## Current issue: overfitting

❌ not seeing enough data (300 subjects still small)?

❌ no early stopping?

❌ no regularization (dropout, weight decay)?

❌ improper learning rate (and scheduler)?


## What can be the next step?

integrate IIP-Transformer and compare with the baseline!

My Other ideas (from my past time series experience + NLP class):

1. try efficient attention mechanisms like FlowAttention, FlashAttention
2. dual encoder (noisy encoder + clean encoder) like DEPICT
3. add contrastive learning objective on top of cross attention like CLIP
4. add auxiliary classification objective (mix in fake samples) like DTCR

## Conda environment setup

BPMT_env - stay tuned!
