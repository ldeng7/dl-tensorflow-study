import signal
import tensorflow as tf

L2_COLLECTION_NAME = "l2reg"


class LayerFactory:

    def __init__(self, weights_stddev, regularizer):
        self.weights_stddev = weights_stddev
        self.regularizer = regularizer

    def Fc_layer(self, tensor, sz_out, do_relu, var_pref, name = None):
        weights = tf.get_variable(
            var_pref + "weights",
            shape = [tensor.shape[-1], sz_out],
            initializer = tf.truncated_normal_initializer(stddev = self.weights_stddev)
        )
        tf.add_to_collection(L2_COLLECTION_NAME, self.regularizer(weights))
        biases = tf.get_variable(
            var_pref + "biases",
            shape = [sz_out],
            initializer = tf.constant_initializer(0.)
        )
        if do_relu:
            return tf.nn.relu(tf.add(tf.matmul(tensor, weights), biases), name = name)
        else:
            return tf.add(tf.matmul(tensor, weights), biases, name = name)

    def Conv_layer_ex(self, tensor, sz, is_same_padding, var_pref, name = None):
        h, w = sz[0], sz[0]
        if list == type(sz[0]):
            h, w = sz[0][0], sz[0][1]
        weights = tf.get_variable(
            var_pref + "weights",
            shape = [h, w, tensor.shape[-1], sz[1]],
            initializer = tf.truncated_normal_initializer(stddev = self.weights_stddev)
        )
        tf.add_to_collection(L2_COLLECTION_NAME, self.regularizer(weights))
        conv = tf.nn.conv2d(
            tensor,
            weights,
            strides = [1, sz[2], sz[2], 1],
            padding = "SAME" if is_same_padding else "VALID",
            name = name
        )
        return conv

    def Conv_layer(self, tensor, sz, is_same_padding, var_pref, name = None):
        conv = self.Conv_layer_ex(tensor, sz, is_same_padding, var_pref, name = name)
        biases = tf.get_variable(
            var_pref + "biases",
            shape = [sz[1]],
            initializer = tf.constant_initializer(0.)
        )
        return tf.nn.relu(tf.nn.bias_add(conv, biases))

    def Pool_layer(self, tensor, is_max, sz, is_same_padding, name = None):
        fn = tf.nn.max_pool if is_max else tf.nn.avg_pool
        return fn(
            tensor,
            ksize = [1, sz[0], sz[0], 1],
            strides = [1, sz[1], sz[1], 1],
            padding = "SAME" if is_same_padding else "VALID",
            name = name
        )


class Network:

    def __init__(self):
        self.sig_int_recv = False

    def train_sig_int(self):
        self.sig_int_recv = True
        print("will save and quit at next info step")

    def Train(self):
        graph = tf.Graph()
        session = tf.Session(graph = graph)
        cp = tf.train.get_checkpoint_state(self.conf_save_path)

        with graph.as_default():
            x, y, y_, train_op, loss, i_step = self.train()
            accs = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            acc = tf.reduce_mean(tf.cast(accs, tf.float32), name = "accuracy")
            saver = tf.train.Saver()
            if cp and cp.model_checkpoint_path:
                saver.restore(session, cp.model_checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())

        sig_int = signal.signal(signal.SIGINT, lambda s, f: self.train_sig_int())
        xv, yv = self.data_validations()
        for i in range(1, self.conf_steps + 1):
            xb, yb = self.data_next_batch()
            _, loss_val, i_step_val = session.run([train_op, loss, i_step], feed_dict = {x: xb, y_: yb})

            if i % self.conf_info_per_steps == 0:
                print("step: %d, loss: %g" % (i_step_val, loss_val))
                print("accuracy: %g" % session.run(acc, feed_dict = {x: xv, y_: yv}))
                if self.sig_int_recv:
                    saver.save(session, self.conf_save_path + self.conf_save_name, global_step = i_step)
                    signal.signal(signal.SIGINT, sig_int)
                    break
