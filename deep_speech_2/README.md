## **Model**
- DeepSpeech2 for PyTorch: https://github.com/SeanNaren/deepspeech.pytorch

## **Configuration**
- Fllow: https://github.com/SeanNaren/deepspeech.pytorch#install

## **DATASET**
### train:
- wget --no-check-certificate  https://us.openslr.org/resources/12/train-clean-100.tar.gz
- wget --no-check-certificate  https://us.openslr.org/resources/12/train-clean-360.tar.gz
- wget --no-check-certificate  https://us.openslr.org/resources/12/train-other-500.tar.gz
### test
- wget --no-check-certificate  https://us.openslr.org/resources/12/test-clean.tar.gz
- wget --no-check-certificate  https://us.openslr.org/resources/12/test-other.tar.gz
### dev
- wget --no-check-certificate  https://us.openslr.org/resources/12/dev-clean.tar.gz
- wget --no-check-certificate  https://us.openslr.org/resources/12/dev-other.tar.gz

### Pre-process: 
`python3 preprocess.py --target-dir path_to_dataset`

## **TEST**
- **Single node,  single gpus**
  > `python3 train.py +configs=librispeech trainer.gpus=1`
- **Single node, multi gpus**
  > `1*4: python3 train.py +configs=librispeech trainer.gpus=4`

  > `1*8: python3 train.py +configs=librispeech trainer.gpus=8`

- **Multi nodes, multi gpus**
  > `todo`

## **Throughput**
> When the bert training phase is stable, the log shows the throughput per second, and then sample multiple times (at least 10) to average.

## **Specify the GPU to use**
- On NVIDIA gpus, Use CUDA_VISIBLE_DEVICES to specify the GPU visible to the program.
- On AMD gpus, Use HIP_VISIBLE_DEVICES to specify the GPU visible to the program.
