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
                g_flatten = tf.reshape(g, [-1])

                target_names = ['n_conv1', 'n_conv2', 'n_conv3', 'n_conv4', 'n_conv5_2', 'n_fc1', 'n_fc2']

                name_cond = tf.reduce_any([tf.equal(self.name, n) for n in target_names])

                log_common_cond = tf.logical_and(
                    tf.equal(tf.math.floormod(lib_snn.model.train_counter - 1, 1), 0),
                    tf.greater(lib_snn.model.train_counter, 1)
                )

                condition = tf.logical_and(log_common_cond, name_cond)

                def log_gradient_tensorboard(grad_ret_flatten, upstream, t):
                    nonzero = tf.reduce_sum(tf.cast(tf.not_equal(grad_ret_flatten, 0), tf.float32))
                    total = tf.cast(tf.size(grad_ret_flatten), tf.float32)
                    effective_grad_rate = nonzero / total

                    def gsnr(grad_ret_flatten):
                        grad_mean = tf.reduce_mean(grad_ret_flatten)
                        grad_variance = tf.math.reduce_variance(grad_ret_flatten)
                        gsnr = tf.square(grad_mean) / (grad_variance + 1e-8)
                        return tf.reduce_mean(gsnr)

                    firing_rate = nonzero / total

                    grad_gsnr = gsnr(grad_ret_flatten)

                    upstream_mean = tf.reduce_mean(upstream)
                    upstream_abs_mean = tf.reduce_mean(tf.abs(upstream))
                    upstream_variance = tf.math.reduce_variance(upstream)
                    upstream_max = tf.reduce_max(upstream)
                    upstream_min = tf.reduce_min(upstream)

                    grad_mean = tf.reduce_mean(grad_ret_flatten)
                    grad_abs_mean = tf.reduce_mean(tf.abs(grad_ret_flatten))
                    grad_variance = tf.math.reduce_variance(grad_ret_flatten)
                    grad_max = tf.reduce_max(grad_ret_flatten)
                    grad_min = tf.reduce_min(grad_ret_flatten)

                    with self.writer.as_default(step=lib_snn.model.train_counter):
                        tf.summary.scalar(f'{self.name}_firing_rate/{t}', firing_rate)
                        tf.summary.scalar(f'{self.name}_effective_grad_rate/{t}', effective_grad_rate)

                        tf.summary.scalar(f'{self.name}_activation_error/mean_{t}', upstream_mean)
                        tf.summary.scalar(f'{self.name}_activation_error/abs_mean_{t}', upstream_abs_mean)
                        tf.summary.scalar(f'{self.name}_activation_error/variance_{t}', upstream_variance)
                        tf.summary.scalar(f'{self.name}_activation_error/max_{t}', upstream_max)
                        tf.summary.scalar(f'{self.name}_activation_error/min_{t}', upstream_min)

                        tf.summary.scalar(f'{self.name}_gradient/mean_{t}', grad_mean)
                        tf.summary.scalar(f'{self.name}_gradient/abs_mean_{t}', grad_abs_mean)
                        tf.summary.scalar(f'{self.name}_gradient/variance_{t}', grad_variance)
                        tf.summary.scalar(f'{self.name}_gradient/max_{t}', grad_max)
                        tf.summary.scalar(f'{self.name}_gradient/min_{t}', grad_min)

                        tf.summary.scalar(f'{self.name}_grad_gsnr/{t}', grad_gsnr)
                        self.writer.flush()
                    return tf.no_op()

                for t in range(1, 5):
                    tf.cond(
                        condition,
                        lambda: log_gradient_tensorboard(g_flatten, dy, t),
                        lambda: tf.no_op()
                    )

                return g

            return y, grad

        return _relu_tb(inputs)