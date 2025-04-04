# how to run

## prototyping:

```bash
CUDA_VISIBLE_DEVICES=0 taskset 0,1,2,3,4 python3 src/main.py
```

## nohup:

```bash
CUDA_VISIBLE_DEVICES=0 nohup taskset 0,1,2,3,4 python3 src/main.py
```

## nohup + background:

```bash
CUDA_VISIBLE_DEVICES=0 nohup taskset 0,1,2,3,4 python3 src/main.py &
```