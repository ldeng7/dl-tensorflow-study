import tensorflow as tf
import network.networks as networks


class DNN(networks.Network):

    def __init__(self):
        super().__init__()
        self.conf_steps = 30000
        self.conf_info_per_steps = 1000

        self.arg_learning_rate_base = 0.8
        self.arg_learning_rate_decay = 0.99
        self.arg_learning_rate_decay_steps = 100
        self.arg_regularization_rate = 0.0001
        self.arg_weights_stddev = 0.1

    def infer(self, tensor):
        for i in range(1, len(self.layout_layers_input_size)):
            tensor = self.lf.Fc_layer(tensor, self.layout_layers_input_size[i], True, "fc_%d_"%(i + 1))
        return self.lf.Fc_layer(tensor, self.layout_output_size, False, "fc_out_", name = "y")

    def train(self):
        self.lf = networks.LayerFactory(self.arg_weights_stddev,
            tf.contrib.layers.l2_regularizer(self.arg_regularization_rate))

        x = tf.placeholder(tf.float32, [None, self.layout_layers_input_size[0]], name = "x")
        y_ = tf.placeholder(tf.float32, [None, self.layout_output_size], name = "y_")
        i_step = tf.Variable(0, trainable = False)

        y = self.infer(x)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection(networks.L2_COLLECTION_NAME))
        learning_rate = tf.train.exponential_decay(
            self.arg_learning_rate_base,
            i_step,
            self.arg_learning_rate_decay_steps,
            self.arg_learning_rate_decay,
            staircase = True
        )
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = i_step)
        return x, y, y_, train_op, loss, i_step
