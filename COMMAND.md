# how to run

## prototyping:

### if we want to use existing checkpoints from both stages for finetuning:

```bash
CUDA_VISIBLE_DEVICES=0 taskset 0,1,2,3,4 python3 src/main.py
```

### if we want to only train second stage and use checkpoints from the first stage for finetuning:

```bash
CUDA_VISIBLE_DEVICES=0 taskset 0,1,2,3,4 python3 src/main.py --second_stage
```

### if we want to train both stages and then do finetuning:

```bash
CUDA_VISIBLE_DEVICES=0 taskset 0,1,2,3,4 python3 src/main.py --first_stage --second_stage
```

## nohup:

```bash
CUDA_VISIBLE_DEVICES=0 nohup taskset 0,1,2,3,4 python3 src/main.py --first_stage False --second_stage False
```

## nohup + background:

```bash
CUDA_VISIBLE_DEVICES=0 nohup taskset 0,1,2,3,4 python3 src/main.py --first_stage False --second_stage False &
```