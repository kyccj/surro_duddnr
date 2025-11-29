import tensorflow_datasets as tfds
# import events_tfds.events.cifar10_dvs
#from events_tfds.vis.image import as_frames
#from events_tfds.vis.image import as_frame
from datasets.events.image import as_frames
from datasets.events.image import as_frames_for_nda
from datasets.events.image import as_frame
# from events_tfds.vis.anim import animate_frames
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import math




from datasets.augmentation_cifar import cutmix

import tensorflow as tf

import matplotlib.pyplot as plt

from config import config
conf = config.flags

def load():
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)


    batch_size = config.batch_size
    num_parallel = tf.data.AUTOTUNE

    if False:
        for events, labels in train_ds:
            print(events)
            print(labels)


        #train_ds = train_ds.map(lambda events, labels: )

    train_ratio = 0.9
    train_ratio_percent = int(train_ratio*100)
    #train_ds, train_ds_info = tfds.load("cifar10_dvs", split="train", as_supervised=True)
    #train_ds = tfds.load("cifar10_dvs", split="train", as_supervised=True)

    #train_ds = tfds.load("cifar10_dvs", split="train[:"+str(train_ratio_percent)+"%]", as_supervised=True, shuffle_files=True)
    #valid_ds = tfds.load("cifar10_dvs", split="train["+str(train_ratio_percent)+"%:]", as_supervised=True)

    train_ds = tfds.load("cifar10_dvs", split="train[10%:]", as_supervised=True, shuffle_files=True)
    valid_ds = tfds.load("cifar10_dvs", split="train[:10%]", as_supervised=True)


    #train_ds = train_ds.map(lambda events, labels: as_frame())


    num_frames = conf.time_step
    conf.time_dim_size = num_frames

    #image_shape = (128,128,3)
    image_shape = (128,128,2)
    #image_shape = (128,128,1)

    # test
    ##for events, labels in train_ds:
    #ds, = train_ds.take(1)
    #events = ds[0]
    #labels = ds[1]
    #as_frames(events,labels,shape=image_shape,num_frames=num_frames)
    #assert False

    #train_ds = train_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))

    #
    #cifa10_dvs_img_size = conf.cifar10_dvs_img_size
    #cifar10_dvs_crop_img_size = conf.cifkhh

    # tf tensor version test
    if False:
    #if True:
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords']
            polarity = events['polarity']
            frame = as_frames(events,labels,shape=image_shape,num_frames=num_frames,augmentation=True)
            #print(labels.numpy())

    if conf.data_aug_mix == 'nda' :
        @tf.function
        def nda_cutmix(images, labels, alpha=1.0):
            choice = tf.random.uniform([], 0, 3, dtype=tf.int32)

            def flatten_bt(x):
                shape = tf.shape(x)
                return tf.reshape(x, [-1, shape[2], shape[3], shape[4]]), shape

            def unflatten_bt(x_flat, orig_shape):
                return tf.reshape(x_flat, orig_shape)

            def roll_all():
                dx = tf.random.uniform([], -3, 4, dtype=tf.int32)
                dy = tf.random.uniform([], -3, 4, dtype=tf.int32)
                return tf.roll(images, shift=[dx, dy], axis=[2, 3])

            def rotate_all():
                angle = tf.random.uniform([], -15., 15.) * math.pi / 180.0
                flat, shp = flatten_bt(images)
                flat = tfa.image.rotate(flat, angles=angle, interpolation='BILINEAR')
                return unflatten_bt(flat, shp)

            def shear_all():
                level = tf.random.uniform([], -15., 15.)
                rad = level * math.pi / 180.0
                transform = [1.0, tf.math.tan(rad), 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0]
                flat, shp = flatten_bt(images)
                flat = tfa.image.transform(flat, transform, interpolation='BILINEAR')
                return unflatten_bt(flat, shp)

            images = tf.case([(tf.equal(choice, 0), roll_all),
                              (tf.equal(choice, 1), rotate_all)],
                             default=shear_all, exclusive=True)

            B, T, H, W, C = tf.unstack(tf.shape(images))

            lam = tfp.distributions.Beta(alpha, alpha).sample([B])
            lam_img = tf.reshape(lam, [-1, 1, 1, 1, 1])
            lam_lbl = tf.reshape(lam, [-1, 1])

            idx = tf.random.shuffle(tf.range(B))
            img_shuf = tf.gather(images, idx)
            label_shuf = tf.gather(labels, idx)

            r_x = tf.random.uniform([B], 0, W, tf.int32)
            r_y = tf.random.uniform([B], 0, H, tf.int32)
            r_w = tf.cast(tf.sqrt(1. - lam) * tf.cast(W, tf.float32), tf.int32)
            r_h = tf.cast(tf.sqrt(1. - lam) * tf.cast(H, tf.float32), tf.int32)

            x1 = tf.clip_by_value(r_x - r_w // 2, 0, W)
            y1 = tf.clip_by_value(r_y - r_h // 2, 0, H)
            x2 = tf.clip_by_value(r_x + r_w // 2, 0, W)
            y2 = tf.clip_by_value(r_y + r_h // 2, 0, H)

            def _apply_cutmix(i, img, img_s):
                yy1, yy2, xx1, xx2 = y1[i], y2[i], x1[i], x2[i]
                paddings = [[0, 0],
                            [yy1, H - yy2],
                            [xx1, W - xx2],
                            [0, 0]]
                patch = tf.pad(img_s[:, yy1:yy2, xx1:xx2, :],
                               paddings, 'CONSTANT')
                return tf.where(patch != 0, patch, img)

            mix_imgs = tf.map_fn(
                lambda tup: _apply_cutmix(*tup),
                (tf.range(B), images, img_shuf),
                fn_output_signature=images.dtype)

            lam_area = 1. - (tf.cast((x2 - x1) * (y2 - y1), tf.float32) /
                             tf.cast(H * W, tf.float32))
            lam_area = tf.reshape(lam_area, [-1, 1])

            mix_labels = lam_area * labels + (1. - lam_area) * label_shuf
            return mix_imgs, mix_labels

        train_ds = train_ds.map(
            lambda events, labels: as_frames_for_nda(events, labels, shape=image_shape, num_frames=num_frames))
        # sample = next(iter(train_ds))
        # images, labels = sample
        # print(f"Shape of the first image: {images.shape}")
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda images, labels: nda_cutmix(images, labels))
    else :
        train_ds = train_ds.map(
            lambda events, labels: as_frames(events, labels, shape=image_shape, num_frames=num_frames,
                                                     augmentation=True))


        train_ds = train_ds.batch(batch_size,drop_remainder=True)
    train_ds = train_ds.prefetch(num_parallel)

    #valid_ds = valid_ds.map(lambda events,labels: as_frame(events,labels,shape=image_shape))
    valid_ds = valid_ds.map(lambda events,labels: as_frames(events,labels,shape=image_shape,num_frames=num_frames))
    valid_ds = valid_ds.batch(batch_size,drop_remainder=True)
    valid_ds = valid_ds.prefetch(num_parallel)


    if False: # tfds events code
    #if True: # tfds events code
        for events, labels in train_ds:
            #frames = as_frames(**{k: v.numpy() for k, v in events.items()}, num_frames=20)
            coords = events['coords'].numpy()
            polarity = events['polarity'].numpy()
            frame = as_frame(coords,polarity)
            #print(labels.numpy())
            #print(tf.reduce_max(events["coords"], axis=0).numpy())
            #anim = animate_frames(frames, fps=4)

            print(labels.numpy())
            plt.imshow(frame)

    #valid_ds = train_ds
    train_ds_num=10000*train_ratio
    valid_ds_num=10000*(1-train_ratio)

    return train_ds, valid_ds, valid_ds, train_ds_num, valid_ds_num, valid_ds_num