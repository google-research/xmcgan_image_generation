# Cross-Modal Contrastive Learning for Text-to-Image Generation

This repository hosts the open source [JAX](https://github.com/google/jax) implementation of [XMC-GAN](https://arxiv.org/abs/2101.04702).


## Setup instructions

### Environment
Set up virtualenv, and install required libraries:
```
virtualenv venv
source venv/bin/activate
```

Add the XMC-GAN library to PYTHONPATH:
```
export PYTHONPATH=$PYTHONPATH:/home/path/to/xmcgan/root/
```

### JAX Installation
Note: Please follow the [official JAX instructions](https://github.com/google/jax#pip-installation) for installing a GPU compatible version of JAX.

### Other Dependencies
After installing JAX, install the remaining dependencies with:
```
pip install -r requirements.txt
```

### Preprocess COCO-2014
To create the training and eval data, first start a directory. By default, the training scripts expect to save results in `data/` in the base directory.
```
mkdir data/
```

The TFRecords required for training and validation on COCO-2014 can be created by running a preprocessing script over the [TFDS coco_captions dataset](https://www.tensorflow.org/datasets/catalog/coco_captions):

```
python preprocess_data.py
```
This may take a while to complete, as it runs a pretrained BERT model over the captions and stores the embeddings. With a GPU, it runs in about 2.5 hours for train, and 1 hour for validation. Once it is done, the train and validation tfrecords files will be saved in the `data/` directory. The train files require around 58G of disk space, and the validation requires 29G.

Note: If you run into an error related to TensorFlow gfile, one workaround is to edit `site-packages/bert/tokenization.py` and change `tf.gfile.GFile` to `tf.io.gfile.GFile`. For more details, refer to the following [link](https://github.com/google-research/bert/issues/1133#issuecomment-703818257).

If you run into a `tensorflow.python.framework.errors_impl.ResourceExhaustedError` about having too many open files, you may have to increase the machine's open file limits. To do so, open the limit configuration file for editing:
```
vi /etc/security/limits.conf
```
and append the following lines to the end of the file:
```
*         hard    nofile      500000
*         soft    nofile      500000
root      hard    nofile      500000
root      soft    nofile      500000
```
You may have to adjust the limit values depending on your machine. You will need to logout and login to your machine for these values to take effect.


### Download Pretrained ResNet

To train XMC-GAN, we need a network pretrained on ImageNet to extract features. For our purposes, we train a ResNet-50 network for this. To download the weights, run:
```
gsutil cp gs://gresearch/xmcgan/resnet_pretrained.npy data/
```
If you would like to pretrain your own network on ImageNet, please refer to the [official Flax ImageNet example](https://github.com/google/flax/tree/master/examples/imagenet).


### Training

Start a training run, by first editing `train.sh` to specify an appropriate work directory. By default, the script assumes that 8 GPUs are available, and runs training on the first 7 GPUs, while `test.sh` assumes testing will run on the last GPU.
After configuring the training job, start an experiment by running it on bash:
```
mkdir exp
bash train.sh exp_name &> train.txt
```

Checkpoints and Tensorboard logs will be saved in `/path/to/exp/exp_name`. By default, the configs/coco_xmc.py config is used, which runs an experiment for 128px images. This is able to accommodate a batch size of 8 on each GPU, and achieves an FID of around 10.5 - 11.0 with the EMA weights. To reproduce the full results on 256px images in our paper, the full model needs to be run using a 32-core Pod slice of [Google Cloud TPU v3](https://cloud.google.com/tpu) devices.

### Evaluation

To run an evaluation job, update `test.sh` with the correct settings used in the training script. Then, execute
```
bash test.sh exp_name &> eval.txt
```
to start an evaluation job. All checkpoints in `workdir` will be evaluated for FID and Inception Score. If you can spare the GPUs, you can also run `train.sh` and `test.sh` in parallel, which will continuously evaluate new checkpoints saved into the work directory. Scores will be written to Tensorboard and output to eval.txt.

### Tensorboard

To start a Tensorboard for monitoring training progress, run:
```
tensorboard --logdir /path/to/exp/exp_name
```

## Citation

If you find this work useful, please consider citing:

```
@inproceedings{zhang2021cross,
  title={Cross-Modal Contrastive Learning for Text-to-Image Generation},
  author={Zhang, Han and Koh, Jing Yu and Baldridge, Jason and Lee, Honglak and Yang, Yinfei},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```


## Disclaimer

Not an official Google product.


