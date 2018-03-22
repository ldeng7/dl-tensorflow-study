import os
import model.models as models

DL_BASE_DIR = os.environ["DL_BASE_DIR"]
SAVE_PATH = DL_BASE_DIR + r"/record/mnist-dnn/"
BATCH_SIZE = 100


class Eval:

    def __init__(self):
        models.Init_eval(self, SAVE_PATH)

    def Run(self, x):
        val = self.session.run(self.graph.get_tensor_by_name("y:0"), feed_dict = {
            self.graph.get_tensor_by_name("x:0"): [x],
        })
        return val.tolist()[0]

    def Close(self):
        self.session.close()


def Train():
    import network.dnn as dnn
    import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data

    mnist = mnist_input_data.read_data_sets(DL_BASE_DIR + r"/train/res/mnist", one_hot = True)
    net = dnn.DNN()
    net.conf_save_path = SAVE_PATH
    net.conf_save_name = "mnist-dnn"
    net.layout_layers_input_size = [784, 500]
    net.layout_output_size = 10
    net.arg_learning_rate_decay_steps = mnist.train.num_examples / BATCH_SIZE
    net.data_next_batch = lambda: mnist.train.next_batch(BATCH_SIZE)
    net.data_validations = lambda: (mnist.validation.images, mnist.validation.labels)

    net.Train()
