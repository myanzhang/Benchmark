## **Model**
- UER-py: https://github.com/dbiir/UER-py

## **DATASET**
- Download: https://share.weiyun.com/9SPPGUOK
- Pre-process: `python3 preprocess.py --corpus_path CLUECorpusSmall_shuf.txt --vocab_path models/google_zh_vocab.txt  --dataset_path cluecorpussmall_bert_zh_seq128_dataset.pt --processes_num 32 --data_processor bert --seq_length 128`

## **TEST**
- **Single node,  single gpus**
  > `python3 pretrain.py --dataset_path /path/to/cluecorpussmall_bert_zh_seq128_dataset.pt --vocab_path models/google_zh_vocab.txt  --config_path models/bert/base_config.json --output_model_path models/bert128_model.bin --world_size 1 --gpu_ranks 0 --total_steps 50000000 --save_checkpoint_steps 100000 --batch_size 128`
- **Single node, multi gpus**
  > `1*4: python3 pretrain.py --dataset_path /path/to/cluecorpussmall_bert_zh_seq128_dataset.pt --vocab_path models/google_zh_vocab.txt  --config_path models/bert/base_config.json --output_model_path models/bert128_model.bin --world_size 4 --gpu_ranks 0 1 2 3 --total_steps 50000000 --save_checkpoint_steps 100000 --batch_size 128`

  > `1*8: python3 pretrain.py --dataset_path /path/to/cluecorpussmall_bert_zh_seq128_dataset.pt --vocab_path models/google_zh_vocab.txt  --config_path models/bert/base_config.json --output_model_path models/bert128_model.bin --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 --total_steps 50000000 --save_checkpoint_steps 100000 --batch_size 128`

- **Multi nodes, multi gpus**
  > `todo`

## **Throughput**
> When the bert training phase is stable, the log shows the throughput of toekens per second, and then sample multiple times (at least 10) to average.

## **Specify the GPU to use**
- On NVIDIA gpus, Use CUDA_VISIBLE_DEVICES to specify the GPU visible to the program.
- On AMD gpus, Use HIP_VISIBLE_DEVICES to specify the GPU visible to the program.
