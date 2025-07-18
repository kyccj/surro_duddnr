import tensorflow_datasets as tfds
import events_tfds.events.cifar10_dvs
#from events_tfds.vis.image import as_frames
#from events_tfds.vis.image import as_frame
from datasets.events.image import as_frames
from datasets.events.image import as_frames_for_nda
from datasets.events.image import as_frame
from events_tfds.vis.anim import animate_frames
import tensorflow_addons as tfa



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
        def nda(images, labels):
            def augment_single_frame(img):
                # roll
                def roll_fn():
                    dx = tf.random.uniform([], -3, 4, dtype=tf.int32)
                    dy = tf.random.uniform([], -3, 4, dtype=tf.int32)
                    return tf.roll(img, shift=[dx, dy], axis=[0, 1])

                # rotate
                def rotate_fn():
                    angle = tf.random.uniform([], -15, 15) * 3.141592 / 180.0
                    return tfa.image.rotate(img, angles=angle, interpolation='BILINEAR')

                # shear
                def shear_fn():
                    level = tf.random.uniform([], -15.0, 15.0)
                    rad = level * 3.141592 / 180.0
                    transform = [1.0, tf.math.tan(rad), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                    return tfa.image.transform(img, transform, interpolation='BILINEAR')

                # cutout
                def cutout_fn():
                    return tfa.image.random_cutout(img[tf.newaxis, ...], mask_size=(16, 16))[0]

                choice = tf.random.uniform([], 0, 4, dtype=tf.int32)
                img = tf.cond(tf.equal(choice, 0), roll_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 1), rotate_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 2), shear_fn, lambda: img)
                img = tf.cond(tf.equal(choice, 3), cutout_fn, lambda: img)
                return img

            images = tf.map_fn(lambda sample: tf.map_fn(augment_single_frame, sample), images)
            return images, labels


        train_ds = train_ds.map(
            lambda events, labels: as_frames_for_nda(events, labels, shape=image_shape, num_frames=num_frames))

        sample = next(iter(train_ds))
        images, labels = sample
        print(f"Shape of the first image: {images.shape}")
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
        train_ds = train_ds.map(lambda images, labels: nda(images, labels))

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