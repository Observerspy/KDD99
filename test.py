import tensorflow as tf
import numpy as np
import load_data as ld
import os

def generate_data():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    test_path = os.path.join(PROJECT_ROOT, "data/test3.data")
    images, label = ld.load(test_path)#debug
    # num = 261401
    # label = np.asarray(range(0, num))
    # images = np.random.random([num, 5, 5, 3])
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images

def get_batch_data():
    label, images = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    label = tf.one_hot(label, depth=2)
    #input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, label_batch = tf.train.shuffle_batch([images, label], batch_size=4096, capacity=10000,
                                                      min_after_dequeue=5000, enqueue_many=True)
    return image_batch, label_batch

image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            for j in range(10):
                print(i, image_batch_v[j], label_batch_v[j])
            i += 1
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)