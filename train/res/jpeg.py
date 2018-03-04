import os, random, math
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile


class Processor:

    def __init__(self):
        self.mean, self.std = 128, 128
        self.depth = 3
        self.height_out, self.width_out = 299, 299
        self.validation_ratio = 0.2
        self.max_class_index = 0
        self.class_indices = {}

        self.i_train_datas = 0
        self.train_images, self.train_labels = [], []

    # collect file pathes in 1st rank of sub dirs of dir_path
    def collect_in_dir(self, dir_path):
        file_pathes = {}
        sub_dir_names = []
        for d in gfile.Walk(dir_path):
            sub_dir_names = d[1]
            break
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(dir_path, sub_dir_name)
            for d in gfile.Walk(sub_dir_path):
                dir_file_pathes = [os.path.join(sub_dir_path, file_name) for file_name in d[2]]
                random.shuffle(dir_file_pathes)
                i = int(math.ceil(len(dir_file_pathes) * self.validation_ratio))
                file_pathes[sub_dir_name] = (dir_file_pathes[:i], dir_file_pathes[i:])
                break
        return file_pathes

    def gen_process_jpeg_tensors(self):
        self.tensor_in = tf.placeholder(tf.string)
        t = tf.expand_dims(tf.image.decode_jpeg(self.tensor_in, self.depth), 0)
        t = tf.image.resize_bilinear(t, tf.constant([self.height_out, self.width_out], dtype = tf.int32))
        t = tf.squeeze(t, [0])
        self.tensor_out = tf.multiply(tf.subtract(t, self.mean), 1.0 / self.std)

    def process(self, dir_path):
        train_datas, validation_images, validation_labels = [], [], []
        file_pathes = self.collect_in_dir(dir_path)
        graph = tf.Graph()
        with graph.as_default():
            self.gen_process_jpeg_tensors()
        with tf.Session(graph = graph) as session:
            for clazz, pathes in file_pathes.items():
                y = [0.] * self.max_class_index
                y[self.class_indices[clazz]] = 1.
                for path in pathes[0]:
                    file_data = gfile.FastGFile(path, 'rb').read()
                    image = session.run(self.tensor_out, feed_dict = {self.tensor_in: file_data})
                    train_datas.append((image, y))
                for path in pathes[1]:
                    file_data = gfile.FastGFile(path, 'rb').read()
                    image = session.run(self.tensor_out, feed_dict = {self.tensor_in: file_data})
                    validation_images.append(image)
                    validation_labels.append(y)
        random.shuffle(train_datas)
        for image, label in train_datas:
            self.train_images.append(image)
            self.train_labels.append(label)
        self.validations = (np.asarray(validation_images), np.asarray(validation_labels))

    def next_train_batch(self, count):
        l = len(self.train_images)
        i_end = self.i_train_datas + count
        if i_end <= l:
            out = (np.asarray(self.train_images[self.i_train_datas : i_end]),
                np.asarray(self.train_labels[self.i_train_datas : i_end]))
        else:
            out = (np.asarray(self.train_images[self.i_train_datas:] + self.train_images[:(i_end - l)]),
                np.asarray(self.train_labels[self.i_train_datas:] + self.train_labels[:(i_end - l)]))
        if i_end >= l:
            i_end -= l
        self.i_train_datas = i_end
        return out
