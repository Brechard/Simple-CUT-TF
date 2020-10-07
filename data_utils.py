import tensorflow as tf


def decode_image(image, use_augmentation, image_size):
    image = tf.image.decode_jpeg(image, channels=3)
    if use_augmentation:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_crop(image, size=image_size)
    # Normalize the pixel values in the range [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, image_size)
    return image


def read_tfrecord(example, use_augmentation, image_size, tfrecord_format):
    if tfrecord_format is None:
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string)
        }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'], use_augmentation, image_size)
    return image


def load_dataset(filenames, image_size, batch_size, use_augmentation, tfrecord_format=None):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: read_tfrecord(x, use_augmentation, image_size, tfrecord_format),
                          num_parallel_calls=batch_size)
    dataset = dataset.cache().shuffle(2020).batch(batch_size, drop_remainder=True)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
