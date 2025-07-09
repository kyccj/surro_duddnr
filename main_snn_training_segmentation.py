
#
# configuration
from config_snn_training_segmentation import config


# snn library
import lib_snn

#
import datasets
import callbacks

########################################
# configuration
########################################
dist_strategy = lib_snn.utils.set_gpu()


################
# name set
################
#
filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path()

########################################
# load dataset
########################################
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
    datasets.datasets.load()

########################################
# confirm dataset
########################################
# def visualize_sample(image, label, title="Sample Image & Mask"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(image.astype(np.uint8))
#     plt.title("Input Image")
#     plt.axis("off")
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(label[..., 0], cmap="jet")
#     plt.title("Segmentation Mask")
#     plt.axis("off")
#
#     plt.suptitle(title)
#     plt.show()
#
# for image, label in train_ds.take(1):
#     print(f"Train Image shape: {image.shape}, Train Label shape: {label.shape}")
#     image = image.numpy()[0]
#     label = label.numpy()[0]
#     visualize_sample(image, label, title="Train Sample")
#
# for image, label in valid_ds.take(1):
#     print(f"Valid Image shape: {image.shape}, Valid Label shape: {label.shape}")
#     image = image.numpy()[0]
#     label = label.numpy()[0]
#     visualize_sample(image, label, title="Validation Sample")

#

with dist_strategy.scope():

    ########################################
    # build model
    ########################################
    model = lib_snn.model_builder_segmentation.model_builder(num_class, train_steps_per_epoch, valid_ds)



