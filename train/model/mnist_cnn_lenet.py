import os
import numpy as np
import model.models as models

DL_BASE_DIR = os.environ["DL_BASE_DIR"]
SAVE_PATH = DL_BASE_DIR + r"/record/mnist-cnn-lenet/"
BATCH_SIZE = 100
INPUT_SIZE = [28, 28, 1]


class Eval:

    def __init__(self):
        models.Init_eval(self, SAVE_PATH)

    def Run(self, x):
        val = self.session.run(self.graph.get_tensor_by_name("y:0"), feed_dict = {
            self.graph.get_tensor_by_name("x:0"): np.reshape(x, [1] + INPUT_SIZE),
        })
        return val.tolist()[0]

    def Close(self):
        self.session.close()


def Train():
    import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
    import network.cnn_lenet as cnn_lenet

    mnist = mnist_input_data.read_data_sets(DL_BASE_DIR + r"/train/res/mnist", one_hot = True)
    def next_batch():
        x, y = mnist.train.next_batch(BATCH_SIZE)
        return np.reshape(x, [-1] + INPUT_SIZE), y
    validations_x = np.reshape(mnist.validation.images, [-1] + INPUT_SIZE)

    net = cnn_lenet.Lenet()
    net.conf_save_path = SAVE_PATH
    net.conf_save_name = "mnist-cnn-lenet"
    net.layout_input_size = INPUT_SIZE
    net.layout_cp_size = [[5, 32, 1, 2, 2], [5, 64, 1, 2, 2]]
    net.layout_fc_size = [512]
    net.layout_output_size = 10
    net.data_next_batch = next_batch
    net.data_validations = lambda: (validations_x, mnist.validation.labels)

    net.Train()
