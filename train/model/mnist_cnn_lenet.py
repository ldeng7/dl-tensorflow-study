import sys, os
import tensorflow as tf
import numpy as np

BATCH_SIZE = 100
INPUT_SIZE = [28, 28, 1]
file_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = file_dir + r"/../record/mnist-cnn-lenet/"


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
    sys.path.append(os.path.dirname(sys.path[0]))
    from network import cnn_lenet

    mnist = mnist_input_data.read_data_sets(file_dir + r"/../res/mnist", one_hot = True)
    conf = cnn_lenet.TrainConf()
    conf.save_path = SAVE_PATH
    conf.save_name = "mnist-cnn-lenet"
    conf.input_size = INPUT_SIZE
    conf.cp_layers_size = [[5, 32, 1, 2, 2], [5, 64, 1, 2, 2]]
    conf.fc_layers_input_size = [512]
    conf.output_size = 10

    def next_batch(mnist):
        x, y = mnist.train.next_batch(BATCH_SIZE)
        return np.reshape(x, [-1] + INPUT_SIZE), y
    validations_x = np.reshape(mnist.validation.images, [-1] + INPUT_SIZE)
    cnn_lenet.train(conf,
        next_batch, mnist,
        lambda: (validations_x, mnist.validation.labels))

if __name__ == "__main__":
    train()
