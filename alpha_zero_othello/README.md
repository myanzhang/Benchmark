## **Model**
- AlphaZero: https://github.com/suragnair/alpha-zero-general


## **TEST FOR OTHELLO 8*8 GAME**
- **Single node,  single gpus**
  > `python3 main.py`

  > `python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr='127.0.0.1' --master_port=3689 main_ddp.py`
- **Single node, multi gpus**
  > `1*4: python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --master_addr='127.0.0.1' --master_port=3689 main_ddp.py`

  > `1*8: python3 -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr='127.0.0.1' --master_port=3689 main_ddp.py`

- **Multi nodes, multi gpus**
  > `todo`

## **Throughput**
> When the bert training phase is stable, the log shows the throughput of toekens per second, and then sample multiple times (at least 10) to average.

## **Specify the GPU to use**
- On NVIDIA gpus, Use CUDA_VISIBLE_DEVICES to specify the GPU visible to the program.
- On AMD gpus, Use HIP_VISIBLE_DEVICES to specify the GPU visible to the program.
