import tensorflow as tf
import network.networks as networks


class Network:

    def __init__(self):
        self.conf_steps = 30001
        self.conf_info_per_steps = 1000
        self.conf_save_per_steps = 5000
        
        self.arg_adam_learning_rate_base = 0.0001
        self.arg_regularization_rate = 0.0001
        self.arg_weights_stddev = 0.1

    def infer(self, tensor):
        cp_sizes = [[None, self.layout_input_size[2]]] + self.layout_cp_size
        for i in range(len(self.layout_cp_size)):
            sz = cp_sizes[i + 1]
            tensor = self.lf.conv_layer(tensor, [sz[0], cp_sizes[i][1], sz[1], sz[2]], True, "cv_%d_"%(i + 1))
            if len(sz) >= 5: tensor = self.lf.pool_layer(tensor, True, [sz[3], sz[4]], True)

        shape = tensor.get_shape().as_list()
        fc_size0 = shape[1] * shape[2] * shape[3]
        tensor = tf.reshape(tensor, [-1, fc_size0])
        fc_sizes = [fc_size0] + self.layout_fc_size
        for i in range(len(fc_sizes) - 1):
            tensor = self.lf.fc_layer(tensor, [sizes_in[i], sizes_in[i + 1]], True, "fc_%d_"%(i + 1))
        return self.lf.fc_layer(tensor, [sizes_in[-1], self.layout_output_size], False, "fc_out_", name = "y")

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

        accs = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(accs, tf.float32), name = "accuracy")

        with tf.Session() as session: networks.train(self, session, x, y_, train_op, loss, accuracy, i_step)
