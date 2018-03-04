import sys, os
import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
INPUT_SIZE = [28, 28, 1]
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = BASE_DIR + r"/record/mnist-cnn-lenet/"


class Eval:

    def __init__(self):
        cp = tf.train.get_checkpoint_state(SAVE_PATH)
        saver = tf.train.import_meta_graph(cp.model_checkpoint_path + ".meta")
        self.session = tf.Session()
        saver.restore(self.session, cp.model_checkpoint_path)
        self.graph = tf.get_default_graph()

    def run(self, x):
        val = self.session.run(self.graph.get_tensor_by_name("y:0"), feed_dict = {
            self.graph.get_tensor_by_name("x:0"): np.reshape(x, [1] + INPUT_SIZE),
        })
        return val.tolist()[0]

    def close(self):
        self.session.close()


def train():
    import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
    sys.path.append(BASE_DIR)
    from network import cnn_lenet

    mnist = mnist_input_data.read_data_sets(BASE_DIR + r"/res/mnist", one_hot = True)
    def next_batch(mnist):
        x, y = mnist.train.next_batch(BATCH_SIZE)
        return np.reshape(x, [-1] + INPUT_SIZE), y
    validations_x = np.reshape(mnist.validation.images, [-1] + INPUT_SIZE)

    net = cnn_lenet.Network()
    net.conf_save_path = SAVE_PATH
    net.conf_save_name = "mnist-cnn-lenet"
    net.layout_input_size = INPUT_SIZE
    net.layout_cp_size = [[5, 32, 1, 2, 2], [5, 64, 1, 2, 2]]
    net.layout_fc_size = [512]
    net.layout_output_size = 10
    net.data_next_batch = next_batch
    net.data_next_batch_arg = mnist
    net.data_validations = lambda: (validations_x, mnist.validation.labels)

    net.train()

if __name__ == "__main__":
    train()
