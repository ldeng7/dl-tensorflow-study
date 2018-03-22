import os, random, math
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

FNAMES = ["train_images", "train_labels", "validate_images", "validate_labels"]

class JpegSetFactory:

    def __init__(self):
        self.depth = 3
        self.mean, self.std = 128, 128
        self.height_out, self.width_out = 299, 299
        self.validate_ratio = 0.05
        self.n_label_index = 0
        self.label_indices = {}

    # collect file paths in 1st rank of sub dirs
    def collect_in_dir(self, dir_path):
        file_paths = {}
        sub_dir_names = []
        for d in gfile.Walk(dir_path):
            sub_dir_names = d[1]
            break
        for sub_dir_name in sub_dir_names:
            sub_dir_path = os.path.join(dir_path, sub_dir_name)
            for d in gfile.Walk(sub_dir_path):
                dir_file_paths = [os.path.join(sub_dir_path, file_name) for file_name in d[2]]
                random.shuffle(dir_file_paths)
                i = int(math.ceil(len(dir_file_paths) * (1 - self.validate_ratio)))
                file_paths[sub_dir_name] = (dir_file_paths[:i], dir_file_paths[i:])
                break
        return file_paths

    def gen_process_jpeg_tensors(self):
        tensor_in = tf.placeholder(tf.string)
        t = tf.expand_dims(tf.image.decode_jpeg(tensor_in, self.depth), 0)
        t = tf.image.resize_bilinear(t, tf.constant([self.height_out, self.width_out], dtype = tf.int32))
        t = tf.squeeze(t, [0])
        tensor_out = tf.multiply(tf.subtract(t, self.mean), 1.0 / self.std)
        return tensor_in, tensor_out

    def Preprocess(self, dir_path):
        files, arrs, train_datas = [], [], []
        for fname in FNAMES:
            files.append(open(os.path.join(dir_path, fname), "wb"))
            arrs.append([])

        file_paths = self.collect_in_dir(dir_path)
        graph = tf.Graph()
        with graph.as_default():
            tensor_in, tensor_out = self.gen_process_jpeg_tensors()
        with tf.Session(graph = graph) as session:
            for label, paths in file_paths.items():
                print("processing dir: %s" % (label))
                y = np.asarray([0.] * self.n_label_index, dtype = np.float32)
                y[self.label_indices[label]] = 1.
                for path in paths[0]:
                    file_data = gfile.FastGFile(path, "rb").read()
                    image = session.run(tensor_out, feed_dict = {tensor_in: file_data})
                    train_datas.append((image, y))
                for path in paths[1]:
                    file_data = gfile.FastGFile(path, "rb").read()
                    image = session.run(tensor_out, feed_dict = {tensor_in: file_data})
                    arrs[2].append(image)
                    arrs[3].append(y)

        print("shuffling train datas")
        random.shuffle(train_datas)
        for image, y in train_datas:
            arrs[0].append(image)
            arrs[1].append(y)
        for i in range(len(FNAMES)):
            print("writing file: %s" % FNAMES[i])
            files[i].write(np.asarray(arrs[i]).data)
            files[i].flush()
            files[i].close()

    def Load(self, dir_path):
        js = JpegSet()
        shapes = [
            (-1, self.height_out, self.width_out, self.depth),
            (-1, self.n_label_index),
        ] * 2
        for i in range(len(FNAMES)):
            print("reading file: %s" % FNAMES[i])
            file = open(os.path.join(dir_path, FNAMES[i]), "rb")
            arr = np.frombuffer(file.read(), dtype = np.float32)
            file.close()
            arr = np.reshape(arr, shapes[i])
            js.arrs.append(arr)
        return js

    def PreprocessFile(self, file_path):
        graph = tf.Graph()
        with graph.as_default():
            tensor_in, tensor_out = self.gen_process_jpeg_tensors()
        with tf.Session(graph = graph) as session:
            file_data = gfile.FastGFile(file_path, "rb").read()
        image = session.run(tensor_out, feed_dict = {tensor_in: file_data})
        return np.expand_dims(image, axis = 0)


class JpegSet:

    def __init__(self):
        self.i_train_datas = 0
        self.arrs = []

    def Next_train_batch(self, count):
        l = len(self.arrs[0])
        i_end = self.i_train_datas + count
        out = (self.arrs[0][self.i_train_datas : i_end], self.arrs[1][self.i_train_datas : i_end])
        if i_end >= l: i_end = 0
        self.i_train_datas = i_end
        return out

    def Validations(self):
        return (self.arrs[2], self.arrs[3])
