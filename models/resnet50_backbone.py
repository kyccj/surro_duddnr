import tensorflow as tf
import lib_snn
from absl import flags

conf = flags.FLAGS
tdbn = conf.nn_mode == 'SNN' and conf.tdbn

# Imagenet Mean RGB values
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
    """Subtract Imagenet mean RGB value."""
    mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
    num_channels = tf.shape(inputs)[-1]
    mean_rgb_extended = tf.concat([mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
    return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
    """Map image values from [0, 255] to [-1, 1]."""
    preprocessed_inputs = (2.0 / 255.0) * tf.cast(inputs, dtype=tf.float32) - 1.0
    return tf.cast(preprocessed_inputs, dtype=dtype)


def ResNet50_Backbone(input_shape, conf, weights=None, output_stride=16, is_training=True, global_pool=False,
                      normalize=False, reuse=None, scope='resnet50_backbone'):
    """
    ResNet50 Backbone with DeepLab-style feature extraction and preprocessing.

    Args:
        input_shape (tuple): (H, W, C)
        conf (object): Configuration object.
        weights (str or None): "imagenet" for pretrained weights, None for random init.
        output_stride (int): Feature map downsampling rate (8 or 16).
        is_training (bool): Training mode.
        global_pool (bool): If True, apply global average pooling.
        normalize (bool): If True, normalize input to [-1,1].
        reuse (bool): Reuse variables.
        scope (str): TensorFlow variable scope.

    Returns:
        net (tf.Tensor): Final feature map.
        end_points (dict): Intermediate feature maps.
    """
    act_type = conf.n_type if conf.nn_mode == 'SNN' else 'relu'
    end_points = {}

    with tf.variable_scope(scope, [input_shape], reuse=reuse) as sc:
        inputs = tf.keras.layers.Input(shape=input_shape, name='input_layer')

        # Preprocessing based on weights
        if weights == "imagenet":
            x = _preprocess_subtract_imagenet_mean(inputs)
        elif normalize:
            x = _preprocess_zero_mean_unit_range(inputs)
        else:
            x = inputs  # No preprocessing

        end_points['input'] = x
        x = lib_snn.layers.InputGenLayer(name='input_gen')(x)

        if conf.nn_mode == 'SNN':
            x = lib_snn.activations.Activation(act_type=act_type, loc='IN', name='n_in')(x)

        # First convolution (7x7 kernel, stride 2)
        x = lib_snn.layers.Conv2D(64, (7, 7), strides=2, padding='same', name='conv1')(x)
        x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name='conv1_bn')(x)
        x = lib_snn.activations.Activation(act_type=act_type, name='conv1_n')(x)
        x = lib_snn.layers.AveragePooling2D(3, strides=2, name='pool1_pool')(x)
        end_points['conv1'] = x

        def bottleneck_block(x, filters, stride=1, rate=1, name=None, store_feature=False):
            """ResNet Bottleneck Block with optional feature storage"""
            shortcut = lib_snn.layers.Conv2D(4 * filters, (1, 1), strides=stride, name=f'{name}_shortcut')(x)
            shortcut = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{name}_shortcut_bn')(shortcut)

            x = lib_snn.layers.Conv2D(filters, (1, 1), strides=stride, name=f'{name}_conv1')(x)
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{name}_conv1_bn')(x)
            x = lib_snn.activations.Activation(act_type=act_type, name=f'{name}_conv1_n')(x)

            x = lib_snn.layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=rate, name=f'{name}_conv2')(x)
            x = lib_snn.layers.BatchNormalization(en_tdbn=tdbn, name=f'{name}_conv2_bn')(x)
            x = lib_snn.activations.Activation(act_type=act_type, name=f'{name}_conv2_n')(x)

            x = lib_snn.layers.Conv2D(4 * filters, (1, 1), name=f'{name}_conv3')(x)
            x = lib_snn.layers.BatchNormalization(name=f'{name}_conv3_bn')(x)

            x = lib_snn.layers.Add(name=f'{name}_out')([shortcut, x])
            x = lib_snn.activations.Activation(act_type=act_type, name=f'{name}_out_n')(x)

            if store_feature:
                end_points[f'{name}_conv3'] = x  # Store for DeepLab decoder
            return x

        block_filters = [64, 128, 256, 512]
        num_blocks = [3, 4, 6, 3]  # ResNet-50

        current_stride = 1
        rate = 1

        for i, (filters, repeats) in enumerate(zip(block_filters, num_blocks)):
            if current_stride >= output_stride:
                stride = 1
                rate *= 2  # Increase dilation rate for deeper layers
            else:
                stride = 2
                current_stride *= stride

            x = bottleneck_block(x, filters, stride=stride, rate=rate, name=f'block{i + 1}_unit1')

            for j in range(1, repeats):
                store_feature = (i == 0 and j == 1) or (i == 1 and j == 2) or (i == 2 and j == 4)
                x = bottleneck_block(x, filters, stride=1, rate=rate, name=f'block{i + 1}_unit{j + 1}',
                                     store_feature=store_feature)

            end_points[f'block{i + 1}'] = x

        if global_pool:
            x = lib_snn.layers.GlobalAveragePooling2D(name='avg_pool')(x)
            end_points['global_pool'] = x

        net = x

        # Load pretrained weights if specified
        if weights == "imagenet":
            resnet_model = tf.keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_tensor=inputs
            )
            x = resnet_model.output  # Overwrite existing x with pretrained features

    return net, end_points
