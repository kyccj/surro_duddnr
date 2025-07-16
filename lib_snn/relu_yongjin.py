import tensorflow as tf
from tensorflow.keras.layers import ReLU as _BaseReLU
from config import config
import lib_snn

class ReLU_yongjin(_BaseReLU):
    def __init__(self, max_value=None, negative_slope=0.0, threshold=0.0, log_dir=config.path_tensorboard, **kwargs):
        super().__init__(max_value, negative_slope, threshold, **kwargs)
        self.writer = tf.summary.create_file_writer(log_dir)

    def call(self, inputs):
        writer = self.writer
        layer_name = self.name
        max_v, neg_s, th = self.max_value, self.negative_slope, self.threshold

        @tf.custom_gradient
        def _relu_tb(x):
            # forward
            y = tf.keras.activations.relu(
                x,
                alpha=neg_s,
                max_value=max_v,
                threshold=th,
            )

            def grad(dy):
                # upstream gradient
                g = tf.cast(x > th, dy.dtype) * dy

                # sparsity
                nonzero = tf.reduce_sum(tf.cast(tf.not_equal(g, 0), tf.float32))
                total = tf.cast(tf.size(g), tf.float32)
                sparsity = nonzero / total

                def gsnr(grad_ret_flatten):
                    mask = tf.not_equal(grad_ret_flatten, 0.0)
                    nonzero_grad = tf.boolean_mask(grad_ret_flatten, mask)
                    grad_mean = tf.reduce_mean(nonzero_grad, axis=0)
                    grad_variance = tf.math.reduce_variance(nonzero_grad, axis=0)
                    gsnr = tf.square(grad_mean) / (grad_variance)
                    return tf.reduce_mean(gsnr)

                def write_grad_ret_sparsity(layer_name, sparsity, gsnr):
                    with writer.as_default(step=lib_snn.model.train_counter):
                        tf.summary.scalar(f"{layer_name}_relu/grad_sparsity", sparsity)
                        tf.summary.scalar(f"{layer_name}_relu/gsnr", gsnr)
                        self.writer.flush()

                log_common_cond = tf.logical_and(
                    tf.equal(tf.math.floormod(lib_snn.model.train_counter - 1, 1), 0),
                    tf.greater(lib_snn.model.train_counter, 1)
                )

                grad_gsnr = gsnr(tf.reshape(g, [-1]))

                tf.cond(log_common_cond,
                        lambda: tf.py_function(write_grad_ret_sparsity,
                                               [layer_name, sparsity, grad_gsnr],
                                               []),
                        lambda: tf.no_op())

                return g

            return y, grad

        return _relu_tb(inputs)