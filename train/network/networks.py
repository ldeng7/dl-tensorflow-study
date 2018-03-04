import tensorflow as tf

L2_COLLECTION_NAME = "l2reg"

class LayerFactory:

    def __init__(self, weights_stddev, regularizer):
        self.weights_stddev = weights_stddev
        self.regularizer = regularizer

    def fc_layer(self, tensor, sz, var_pref, name = None):
        weights = tf.get_variable(
            var_pref + "weights",
            shape = sz,
            initializer = tf.truncated_normal_initializer(stddev = self.weights_stddev)
        )
        tf.add_to_collection(L2_COLLECTION_NAME, self.regularizer(weights))
        biases = tf.get_variable(
            var_pref + "biases",
            shape = [sz[1]],
            initializer = tf.constant_initializer(0.)
        )
        return tf.add(tf.matmul(tensor, weights), biases, name = name)

    def fc_layers(self, tensor, sizes_in, sz_out, var_pref, name = None):
        for i in range(len(sizes_in) - 1):
            tensor = self.fc_layer(tensor, [sizes_in[i], sizes_in[i + 1]], "%s%d_"%(var_pref, i + 1))
            tensor = tf.nn.relu(tensor)
        return self.fc_layer(tensor, [sizes_in[-1], sz_out], var_pref + "out_", name = name)

    def conv_layer_ex(self, tensor, sz, is_same_padding, var_pref, name = None):
        h, w = sz[0], sz[0]
        if list == type(sz[0]):
            h, w = sz[0][0], sz[0][1]
        weights = tf.get_variable(
            var_pref + "weights",
            shape = [h, w, sz[1], sz[2]],
            initializer = tf.truncated_normal_initializer(stddev = self.weights_stddev)
        )
        tf.add_to_collection(L2_COLLECTION_NAME, self.regularizer(weights))
        conv = tf.nn.conv2d(
            tensor,
            weights,
            strides = [1, sz[3], sz[3], 1],
            padding = "SAME" if is_same_padding else "VALID",
            name = name
        )
        return conv

    def conv_layer(self, tensor, sz, is_same_padding, var_pref, name = None):
        conv = self.conv_layer_ex(tensor, sz, is_same_padding, var_pref, name = name)
        biases = tf.get_variable(
            var_pref + "biases",
            shape = [sz[2]],
            initializer = tf.constant_initializer(0.)
        )
        return tf.nn.relu(tf.nn.bias_add(conv, biases))

    def pool_layer(self, tensor, is_max, sz, is_same_padding, name = None):
        fn = tf.nn.max_pool if is_max else tf.nn.avg_pool
        return fn(
            tensor,
            ksize = [1, sz[0], sz[0], 1],
            strides = [1, sz[1], sz[1], 1],
            padding = "SAME" if is_same_padding else "VALID",
            name = name
        )


def train(this, session, x, y_, train_op, loss, accuracy, i_step):
    saver = tf.train.Saver()
    cp = tf.train.get_checkpoint_state(this.conf_save_path)
    if cp and cp.model_checkpoint_path:
        saver.restore(session, cp.model_checkpoint_path)
    else:
        session.run(tf.global_variables_initializer())

    xv, yv = this.data_validations()
    for i in range(1, this.conf_steps):
        xb, yb = this.data_next_batch(this.data_next_batch_arg)
        _, loss_val, i_step_val = session.run([train_op, loss, i_step], feed_dict = {x: xb, y_: yb})

        if i % this.conf_info_per_steps == 0:
            print("batch train step: %d, loss: %g" % (i_step_val, loss_val))
            print("accuracy: %g" % session.run(accuracy, feed_dict = {x: xv, y_: yv}))

        if i % this.conf_save_per_steps == 0:
            saver.save(session, this.conf_save_path + this.conf_save_name, global_step = i_step)
