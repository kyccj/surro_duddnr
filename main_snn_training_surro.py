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
        model.load_weights(config.load_weight, by_name=True, skip_mismatch=True)

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
        # model.summary()
        # result = model.evaluate(test_ds, callbacks=callbacks_test)

        # ####################loss landscape######################young
        if conf.loss_landscape:
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)

            batch_size = 100

            for layer_idx, layer in enumerate(model.layers):
                kernel_idx = None
                for i, w in enumerate(layer.weights):
                    print(f"{layer_idx}:{layer.name}")
                    print(f"{i}:{w.name}")

                    print("a")

                for i, w in enumerate(layer.weights):
                    if "kernel" in w.name:
                        idx = int(layer_idx)
                        kernel_idx = i
                        break

                if kernel_idx is None:
                    print(f"Skipping layer {layer.name}: No kernel weights found.")
                    continue

                if idx < 1:
                    print(f"Skipping layer {layer.name}: fast check")
                    continue

                print(f"Processing Layer : {layer.name}")

                x_test_list, y_test_list = [], []
                for x_batch, y_batch in test_ds.as_numpy_iterator():
                    x_test_list.append(x_batch)
                    y_test_list.append(y_batch)

                x_test = np.concatenate(x_test_list, axis=0)
                y_test = np.concatenate(y_test_list, axis=0)


                weights = layer.get_weights()
                kernel_weights = weights[kernel_idx]

                print(f"Layer: {layer.name}")
                print("  kernel shape   :", kernel_weights.shape)
                direction1 = np.random.randn(*kernel_weights.shape)
                print("  direction1 shape:", direction1.shape)
                direction2 = np.random.randn(*kernel_weights.shape)

                direction1 /= np.linalg.norm(direction1)
                direction2 /= np.linalg.norm(direction2)

                x_range = np.linspace(-1, 1, 25)
                y_range = np.linspace(-1, 1, 25)
                X, Y = np.meshgrid(x_range, y_range)
                Z = np.zeros(X.shape).flatten()
                perturbations = np.stack([X.flatten(), Y.flatten()], axis=1)

                ########Calculate loss ########
                num_samples = x_test.shape[0]
                num_batches = num_samples // batch_size

                perturbed_weights_list = [kernel_weights + p[0] * direction1 + p[1] * direction2 for p in perturbations]

                for perturb_idx, perturbed_kernel in enumerate(tqdm(perturbed_weights_list, desc=f"Layer {layer_idx + 1}", leave=False)):
                    perturbed_weights = weights.copy()
                    perturbed_weights[kernel_idx] = perturbed_kernel
                    layer.set_weights(perturbed_weights)

                    y_pred_batches = model.predict(x_test, batch_size=batch_size, verbose=0)
                    batch_loss = loss_fn(y_test, y_pred_batches)
                    batch_loss = tf.reduce_mean(batch_loss).numpy()

                    Z[perturb_idx] = batch_loss

                Z = Z.reshape(X.shape)

                #     ############################discrete_contour
                plt.figure(figsize=(6, 6))
                contour = plt.contour(X, Y, Z, levels=10)
                plt.clabel(contour, inline=True, fontsize=8)
                plt.xlabel("Direction 1")
                plt.ylabel("Direction 2")
                plt.title(f'Loss Landscape - {layer.name}')
                plt.grid()
                # ###################################################
                ############################continuous_colorbar
                # plt.figure(figsize=(6, 4))
                # contour = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
                # plt.colorbar(contour)
                # plt.xlabel("Direction 1")
                # plt.ylabel("Direction 2")
                # plt.title(f"Loss Landscape - {layer.name}")
                # plt.grid()
                ####################################################
                model_dir = os.path.basename(conf.name_model_load)
                save_dir = conf.LS_save+model_dir
                os.makedirs(save_dir, exist_ok=True)

                filename = f"loss_landscape_{layer.name}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                plt.close()

                print(f"Saved: {os.path.join(save_dir, filename)}")
            print('end')

        ####################### t-SNE ############################

        if conf.tSNE :
            from sklearn.manifold import TSNE

            all_time_step_outputs = []
            all_labels = []

            for x_batch, y_batch in test_ds:
                preds = model(x_batch, training=False)
                arr = model.get_layer('n_predictions').time_step_output.numpy()
                all_time_step_outputs.append(arr)
                all_labels.append(y_batch.numpy())

            all_time_step_outputs = np.concatenate(all_time_step_outputs, axis=1)
            all_labels = np.concatenate(all_labels, axis=0)

            print("all_time_step_outputs shape:", all_time_step_outputs.shape)
            print("all_labels shape:", all_labels.shape)

            if all_labels.ndim > 1:
                all_labels = np.argmax(all_labels, axis=1)

            num_time_steps = all_time_step_outputs.shape[0]

            ncols = min(4, num_time_steps)
            nrows = int(np.ceil(num_time_steps / ncols))
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5 * nrows), dpi=300)
            axes = axes.flatten() if num_time_steps > 1 else [axes]

            all_scatter = []

            for t in range(num_time_steps):
                features = all_time_step_outputs[t]  # (total_sample, num_classes)
                labels = all_labels

                if features.shape[0] > 2000:
                    idx = np.random.choice(features.shape[0], 2000, replace=False)
                    features = features[idx]
                    labels = labels[idx]


                tsne_res = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(features)
                ax = axes[t]
                sc = ax.scatter(tsne_res[:, 0], tsne_res[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
                all_scatter.append(sc)
                ax.set_title(f't-SNE (Time step {t + 1})')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.set_aspect('auto')
                ax.grid(True)
                ax.set_xlim(-80, 80)
                ax.set_ylim(-80, 80)

            fig.subplots_adjust(right=0.87)
            cbar_ax = fig.add_axes([0.89, 0.13, 0.02, 0.75])
            cbar = fig.colorbar(all_scatter[0], cax=cbar_ax, ticks=range(10))
            cbar.set_label('Class Label')
            cbar.set_ticks(range(10))
            cbar.set_ticklabels([str(i) for i in range(10)])

            fig.suptitle('t-SNE by Time Step', fontsize=22, y=0.98)
            fig.tight_layout(rect=[0, 0, 0.87, 1])

            plt.savefig("./tSNE/box_4.svg", format="svg", dpi=300, bbox_inches="tight")
            plt.close(fig)



