import tensorflow as tf
import tensorflow_addons as tfa
import keras

import lib_snn

from lib_snn.sim import glb

from absl import flags

conf = flags.FLAGS

from config import config

#
from models.models_segmentation import model_sel

import utils

import collections


def model_builder(
        num_class,
        train_steps_per_epoch,
        valid_ds
):
    print('Model Builder - {}'.format(conf.nn_mode))
    glb.model_compile_done_reset()

    eager_mode = config.eager_mode

    #
    model_name = config.model_name
    dataset_name = config.dataset_name

    data_batch = iter(valid_ds.take(1))
    images, labels = next(data_batch)

    batch_size = config.batch_size

    image_shape = images.shape[1:]
    # print(f"images_shape : {image_shape}")

    # train
    train_type = config.train_type
    train_epoch = config.train_epoch
