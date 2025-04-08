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


## IIP-Transformer (Q. Wang et. al, CVPR 2022)

IIP-Transformer architecture:
![alt text](docs/IIP-Transformer.png)

## implementation pending tasks

- [ ] implement IIP-Transformer (T1)
  - [ ] individual transformer layer
    - [ ] Intra-Inter-Part attention
    - [ ] S-IIPA
    - [ ] T-IIPA
    - [ ] complete the transformer layer

  - [ ] finish up the entire transformer
    - [ ] add class token
    - [ ] add FC layer

  - [ ] add additional FC layers for dimension reduction

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


## Current issue: overfitting

❌ imbalanced split: random validation split instead of class-specific split (probably the root cause)

❌ no early stopping

❌ no regularization (dropout, weight decay)

❌ improper learning rate (and scheduler)

## Conda environment setup

BPMT_env - stay tuned!
