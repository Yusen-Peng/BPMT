# FR-Former: Fast, Real-time Transformer for Efficient and Accurate Gait Recognition

## Computer Vision Undergraduate Thesis Advised Dr. Alper Yilmaz

## proposed architecture

Our FR-Former pipeline design:
![alt text](docs/pipeline.png)


Our FR-Former architecture design:
![alt text](docs/FR-Former.png)

IIP-Transformer architecture:
![alt text](docs/IIP-Transformer.png)

## implementation roadmap

### First phase of masked pretraining (modality level):

- [ ] load Gait3D data

    - [ ] load pose/keypoints data from Gait3D (may start with a subset of data)

    - [ ] verify the correctness of the data loading process

- [ ] build the modality pool
    - [ ] partition original keypoints into 5 different parts/modalities
    - [ ] verify the correctness of:
        - [ ] torso
        - [ ] left leg
        - [ ] right leg
        - [ ] left arm
        - [ ] right arm

- [ ] implement IIP-Transformer (T1)
    - [ ] individual transformer layer
        - [ ] Intra-Inter-Part attention
        - [ ] S-IIPA
        - [ ] T-IIPA
        - [ ] complete the transformer layer

    - [ ] finish up the entire transformer
        - [ ] add class token
        - [ ] add FC layer 


- [ ] add additional FC layers to reduce dimension


### Second phase of masked pretraining (modality pair level):

- [ ] construct pairs of features from modality i and modality j

- [ ] implement the cross attention layer

- [ ] implement another IIP-Transformer (T2), should be similar to T1
    - [ ] individual transformer layer
        - [ ] Intra-Inter-Part attention
        - [ ] S-IIPA
        - [ ] T-IIPA
        - [ ] complete the transformer layer

    - [ ] finish up the entire transformer
        - [ ] add class token
        - [ ] add FC layer

### Finetuning:

- [ ] cross attention between transformer T2 output (the second phase of pretraining) and modality features (the first phase of pretraining)

- [ ] gait recognition head
