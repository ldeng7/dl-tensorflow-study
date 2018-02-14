import sys, os
import tensorflow as tf

BATCH_SIZE = 100
file_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = file_dir + r"/../record/mnist-dnn/"


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


if __name__ == "__main__":
    import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
    sys.path.append(os.path.dirname(sys.path[0]))
    from network import dnn

    mnist = mnist_input_data.read_data_sets(file_dir + r"/../res/mnist", one_hot = True)
    conf = dnn.TrainConf()
    conf.save_path = SAVE_PATH
    conf.save_name = "mnist-dnn"
    conf.layers_input_size = [784, 500]
    conf.output_size = 10
    conf.learning_rate_decay_steps = mnist.train.num_examples / BATCH_SIZE

    dnn.batch_train(conf,
        lambda mnist: mnist.train.next_batch(BATCH_SIZE), mnist,
        lambda: (mnist.validation.images, mnist.validation.labels))
