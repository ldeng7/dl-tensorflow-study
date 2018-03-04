import sys, os
import tensorflow as tf

BATCH_SIZE = 100
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_PATH = BASE_DIR + r"/record/mnist-dnn/"


class Eval:

    def __init__(self):
        cp = tf.train.get_checkpoint_state(SAVE_PATH)
        saver = tf.train.import_meta_graph(cp.model_checkpoint_path + ".meta")
        self.session = tf.Session()
        saver.restore(self.session, cp.model_checkpoint_path)
        self.graph = tf.get_default_graph()

    def run(self, x):
        val = self.session.run(self.graph.get_tensor_by_name("y:0"), feed_dict = {
            self.graph.get_tensor_by_name("x:0"): [x],
        })
        return val.tolist()[0]

    def close(self):
        self.session.close()


def train():
    import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
    sys.path.append(BASE_DIR)
    from network import dnn

    mnist = mnist_input_data.read_data_sets(BASE_DIR + r"/res/mnist", one_hot = True)
    net = dnn.Network()
    net.conf_save_path = SAVE_PATH
    net.conf_save_name = "mnist-dnn"
    net.layout_layers_input_size = [784, 500]
    net.layout_output_size = 10
    net.arg_learning_rate_decay_steps = mnist.train.num_examples / BATCH_SIZE
    net.data_next_batch = lambda mnist: mnist.train.next_batch(BATCH_SIZE)
    net.data_next_batch_arg = mnist
    net.data_validations = lambda: (mnist.validation.images, mnist.validation.labels)

    net.train()

if __name__ == "__main__":
    train()
