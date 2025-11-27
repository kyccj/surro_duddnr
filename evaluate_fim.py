from config_snn_training_surro import config
conf = config.flags

import pandas as pd
import tensorflow as tf

import lib_snn, datasets
import tensorflow as tf, re, pandas as pd
import numpy as np
from tqdm import tqdm

POWER_ITERS      = 50
CSV_OUT          = f'{conf.fire_surro_grad_func}_beta={conf.surro_grad_beth}fim_summary_yc_epoch_1.csv'
def set_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        except RuntimeError:
            pass
    return tf.distribute.get_strategy()

def top_eigval_power_iter(g_flat, iters=POWER_ITERS):
    g_flat = tf.expand_dims(g_flat, 0)  # [1, D]
    D = g_flat.shape[1]
    v = tf.random.normal(shape=(D,), dtype=g_flat.dtype)
    v = tf.math.l2_normalize(v)

    for _ in range(iters):
        # [1, D] @ [D] -> [1]
        v = tf.linalg.matvec(g_flat, v)
        # [D, 1] @ [1] -> [D]
        v = tf.linalg.matvec(tf.transpose(g_flat), v) / tf.cast(tf.shape(g_flat)[0], tf.float32)
        v = tf.math.l2_normalize(v)

    v2d = tf.expand_dims(v, -1)  # [D, 1]
    g_flat_T = tf.transpose(g_flat)  # [D, 1]
    eigvec = tf.matmul(g_flat_T, tf.matmul(g_flat, v2d)) / tf.cast(tf.shape(g_flat)[0], tf.float32)  # [D, 1]
    eigval = tf.matmul(tf.transpose(v2d), eigvec)  # [1, 1]
    return tf.squeeze(eigval)  # scalar

strategy = set_gpu()

fim_results = []

with strategy.scope():
    # dataset
    (train_ds, valid_ds, test_ds,
     *_ , num_class, train_steps_per_epoch) = datasets.datasets.load()

    # model
    model = lib_snn.model_builder.model_builder(num_class=num_class,
                                               train_steps_per_epoch=train_steps_per_epoch,
                                               valid_ds=valid_ds)
    if config.load_model:
        model.load_weights(config.load_weight, by_name=True, skip_mismatch=True)

    model.trainable = False

    target_layers = ["conv1","conv1_1", "conv2","conv2_1", "conv3","conv3_1","conv3_2", "conv4","conv4_1","conv4_2","conv5","conv5_1", "conv5_2", "fc1", "fc2"]
    BATCHES_TO_SCAN = 100  # How many batches to sample

    for target_layer in target_layers:
        layer = model.get_layer(target_layer)
        weight = layer.kernel
        fim_diag = tf.zeros_like(weight)

        eigvals_total = []

        for batch in tqdm(test_ds.take(BATCHES_TO_SCAN), desc=f'{target_layer}'):
            x, y, *rest = tf.keras.utils.unpack_x_y_sample_weight(batch)

            with tf.GradientTape(persistent=True) as tape:
                tape.watch(weight)

                y_pred = model(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, y_pred))

            grad = tape.gradient(loss, weight)
            fim_diag += tf.square(grad)

            g_flat = tf.reshape(grad, [-1])

            # Instead of calculating the full FIM, use power iteration to compute ¥ë©û
            lambda_1 = top_eigval_power_iter(g_flat, iters=POWER_ITERS)

            eigvals_total.append(lambda_1.numpy())  # Store the computed ¥ë©û

            del tape

        # Calculate mean and std of ¥ë©û across batches
        lambda_1_mean = np.mean(eigvals_total)
        lambda_1_std = np.std(eigvals_total)

        # Save the results for this layer
        fim_results.append({
            "Layer": target_layer,
            "lambda_1_mean": lambda_1_mean,
            "lambda_1_std": lambda_1_std,
        })

    # ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡ Save results to CSV ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    fim_df = pd.DataFrame(fim_results)
    fim_df.to_csv(CSV_OUT, index=False)

    print(f"Results saved to {CSV_OUT}")