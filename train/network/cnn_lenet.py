import tensorflow as tf
import network.networks as networks


class Lenet(networks.Network):

    def __init__(self):
        super().__init__()
        self.conf_steps = 30000
        self.conf_info_per_steps = 1000

        self.arg_adam_learning_rate_base = 0.0001
        self.arg_regularization_rate = 0.0001
        self.arg_weights_stddev = 0.1

    def infer(self, tensor):
        for i in range(len(self.layout_cp_size)):
            sz = self.layout_cp_size[i]
            tensor = self.lf.Conv_layer(tensor, sz[:3], True, "cv_%d_"%(i + 1))
            if len(sz) >= 5: tensor = self.lf.Pool_layer(tensor, True, sz[3:], True)

        shape = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, [-1, shape[1] * shape[2] * shape[3]])
        for i in range(len(self.layout_fc_size)):
            tensor = self.lf.Fc_layer(tensor, self.layout_fc_size[i], True, "fc_%d_"%(i + 1))
        return self.lf.Fc_layer(tensor, self.layout_output_size, False, "fc_out_", name = "y")

    def train(self):
        self.lf = networks.LayerFactory(self.arg_weights_stddev,
            tf.contrib.layers.l2_regularizer(self.arg_regularization_rate))

        x = tf.placeholder(tf.float32, [None] + self.layout_input_size, name = "x")
        y_ = tf.placeholder(tf.float32, [None, self.layout_output_size], name = "y_")
        i_step = tf.Variable(0, trainable = False)

        y = self.infer(x)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection(networks.L2_COLLECTION_NAME))
        train_op = tf.train.AdamOptimizer(self.arg_adam_learning_rate_base).minimize(loss, global_step = i_step)
        return x, y, y_, train_op, loss, i_step
