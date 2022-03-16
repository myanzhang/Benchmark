## **TEST**

- **Single node,  single gpus**
  
  > `python3 torch_resnet50.py --train-dir '/path/to/imagenet'`
- **Single node, multi gpus**
  > `1*4: python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=23456 torch_resnet50_ddp.py --train-dir '/path/to/imagenet'`

  > `1*8: python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=23456 torch_resnet50_ddp.py --train-dir '/path/to/imagenet'`

- **Multi nodes, multi gpus**
  > `todo`

## **Throughput**
> During the training of resnet-50, the log outputs the number of batches trained per second (its/s). After the training phase is stable, sample multiple times (at least 10) to obtain the average value，which is called avg(its/s). Then get the throughput value: imgs/sec = avg(its/s)* batch_size

## **Specify the GPU to use**

- On NVIDIA gpus, Use CUDA_VISIBLE_DEVICES to specify the GPU visible to the program.
- On AMD gpus, Use HIP_VISIBLE_DEVICES to specify the GPU visible to the program.
