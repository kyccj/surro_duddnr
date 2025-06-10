#
# #
# # configuration
# from config_snn_training_surro import config
#
#
# # snn library
# import lib_snn
#
# #
# import datasets
# import callbacks
#
# ########################################
# # configuration
# ########################################
# dist_strategy = lib_snn.utils.set_gpu()
#
#
# ################
# # name set
# ################
# #
# filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path()
#
# ########################################
# # load dataset
# ########################################
# train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
#     datasets.datasets.load()
#     #datasets.datasets_bck_eventdata.load()
#
#
# #
# with dist_strategy.scope():
#
#     ########################################
#     # build model
#     ########################################
#     #data_batch = valid_ds.take(1)
#     #model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)
#     model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch,valid_ds)
#
#     ########################################
#     # load model
#     ########################################
#     if config.load_model:
#         model.load_weights(config.load_weight)
#
#     ################
#     # Callbacks
#     ################
#     callbacks_train, callbacks_test = \
#         callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)
#
#     #
#     if config.train:
#         print('Train mode')
#
#         model.summary()
#         #train_steps_per_epoch = train_ds_num/batch_size
#         train_epoch = config.flags.train_epoch
#         init_epoch = config.init_epoch
#         train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
#                                     initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
#     else:
#         print('Test mode')
#
#         result = model.evaluate(test_ds, callbacks=callbacks_test)
#


#
# configuration
from config_snn_training_surro import config
conf = config.flags


# snn library
import lib_snn

#
import datasets
import callbacks
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd
import logging

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
    #datasets.datasets_bck_eventdata.load()


#
with dist_strategy.scope():

    ########################################
    # build model
    ########################################
    #data_batch = valid_ds.take(1)
    #model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)
    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch,valid_ds)

    ########################################
    # load model
    ########################################
    if config.load_model:
        model.load_weights(config.load_weight)

    ################
    # Callbacks
    ################
    callbacks_train, callbacks_test = \
        callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)

    #
    if config.train:
        print('Train mode')

        model.summary()
        #train_steps_per_epoch = train_ds_num/batch_size
        train_epoch = config.flags.train_epoch
        init_epoch = config.init_epoch
        train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                    initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
    else:
        print('Test mode')
        result = model.evaluate(test_ds, callbacks=callbacks_test)
        # # ####################loss landscape######################young
        # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        #
        # batch_size = 100
        #
        # for layer_idx, layer in enumerate(model.layers):
        #     kernel_idx = None
        #     for i, w in enumerate(layer.weights):
        #         print(f"{layer_idx}:{layer.name}")
        #         print(f"{i}:{w.name}")
        #
        #         print("a")
        #
        #     for i, w in enumerate(layer.weights):
        #         if "kernel" in w.name:
        #             idx = int(layer_idx)
        #             kernel_idx = i
        #             break
        #
        #     if kernel_idx is None:
        #         print(f"Skipping layer {layer.name}: No kernel weights found.")
        #         continue
        #
        #     if idx < 43:
        #         print(f"Skipping layer {layer.name}: fast check")
        #         continue
        #
        #     print(f"Processing Layer : {layer.name}")
        #
        #     x_test_list, y_test_list = [], []
        #     for x_batch, y_batch in test_ds.as_numpy_iterator():
        #         x_test_list.append(x_batch)
        #         y_test_list.append(y_batch)
        #
        #     x_test = np.concatenate(x_test_list, axis=0)
        #     y_test = np.concatenate(y_test_list, axis=0)
        #
        #
        #     weights = layer.get_weights()
        #     kernel_weights = weights[kernel_idx]
        #
        #     print(f"Layer: {layer.name}")
        #     print("  kernel shape   :", kernel_weights.shape)
        #     direction1 = np.random.randn(*kernel_weights.shape)
        #     print("  direction1 shape:", direction1.shape)
        #     direction2 = np.random.randn(*kernel_weights.shape)
        #
        #     direction1 /= np.linalg.norm(direction1)
        #     direction2 /= np.linalg.norm(direction2)
        #
        #     x_range = np.linspace(-1, 1, 25)
        #     y_range = np.linspace(-1, 1, 25)
        #     X, Y = np.meshgrid(x_range, y_range)
        #     Z = np.zeros(X.shape).flatten()
        #     perturbations = np.stack([X.flatten(), Y.flatten()], axis=1)
        #
        #     ########Calculate loss ########
        #     num_samples = x_test.shape[0]
        #     num_batches = num_samples // batch_size
        #
        #     perturbed_weights_list = [kernel_weights + p[0] * direction1 + p[1] * direction2 for p in perturbations]
        #
        #     for perturb_idx, perturbed_kernel in enumerate(tqdm(perturbed_weights_list, desc=f"Layer {layer_idx + 1}", leave=False)):
        #         perturbed_weights = weights.copy()
        #         perturbed_weights[kernel_idx] = perturbed_kernel
        #         layer.set_weights(perturbed_weights)
        #
        #         y_pred_batches = model.predict(x_test, batch_size=batch_size, verbose=0)
        #         batch_loss = loss_fn(y_test, y_pred_batches)
        #         batch_loss = tf.reduce_mean(batch_loss).numpy()
        #
        #         Z[perturb_idx] = batch_loss
        #
        #     Z = Z.reshape(X.shape)
        #
        #     #     ############################discrete_contour
        #     plt.figure(figsize=(6, 6))
        #     contour = plt.contour(X, Y, Z, levels=10)
        #     plt.clabel(contour, inline=True, fontsize=8)
        #     plt.xlabel("Direction 1")
        #     plt.ylabel("Direction 2")
        #     plt.title(f'Loss Landscape - {layer.name}')
        #     plt.grid()
        #     # ###################################################
        #     ############################continuous_colorbar
        #     # plt.figure(figsize=(6, 4))
        #     # contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
        #     # plt.colorbar(contour)
        #     # plt.xlabel("Direction 1")
        #     # plt.ylabel("Direction 2")
        #     # plt.title(f"Loss Landscape - {layer.name}")
        #     # plt.grid()
        #     ####################################################
        #     model_dir = os.path.basename(conf.name_model_load)
        #     save_dir = f'/home/dydwls6598/PycharmProjects/Surro/landscape/{model_dir}'
        #     os.makedirs(save_dir, exist_ok=True)
        #
        #     filename = f"loss_landscape_{layer.name}.png"
        #     plt.savefig(os.path.join(save_dir, filename), dpi=300)
        #     plt.close()
        #
        #     print(f"Saved: {os.path.join(save_dir, filename)}")
        # print('end')