import tensorflow as tf
from tensorflow.python.ops import array_ops


class Network:

    def conv_layer_ex(self, tensor, sz, is_same_padding, weights_stddev, var_name_pref, name = None):
        h, w = sz[0], sz[0]
        if list == type(sz[0]):
            h, w = sz[0][0], sz[0][1]
        weights = tf.get_variable(var_name_pref + "weights",
            shape = [h, w, sz[1], sz[2]],
            initializer = tf.truncated_normal_initializer(stddev = weights_stddev)
        )
        tf.add_to_collection("l2reg", self.regularizer(weights))
        conv = tf.nn.conv2d(
            tensor,
            weights,
            strides = [1, sz[3], sz[3], 1],
            padding = "SAME" if is_same_padding else "VALID",
            name = name
        )
        return conv

    def conv_layer(self, tensor, sz, is_same_padding, var_name_pref, name = None):
        conv = self.conv_layer_ex(tensor, sz, is_same_padding, self.weights_stddev, var_name_pref, name = name)
        biases = tf.get_variable(var_name_pref + "biases",
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


    def incep_block_a(self, tensor, input_depth, x_depth, name_pref):
        tensor1 = self.conv_layer(tensor, [1, input_depth, 64, 1], True, name_pref + "br1_cv1_")
        tensor2 = self.conv_layer(tensor, [1, input_depth, 48, 1], True, name_pref + "br2_cv1_")
        tensor2 = self.conv_layer(tensor2, [5, 48, 64, 1], True, name_pref + "br2_cv2_")
        tensor3 = self.conv_layer(tensor, [1, input_depth, 64, 1], True, name_pref + "br3_cv1_")
        tensor3 = self.conv_layer(tensor3, [3, 64, 96, 1], True, name_pref + "br3_cv2_")
        tensor3 = self.conv_layer(tensor3, [3, 96, 96, 1], True, name_pref + "br3_cv3_")
        tensor4 = self.pool_layer(tensor, False, [3, 1], True)
        tensor4 = self.conv_layer(tensor4, [1, input_depth, x_depth, 1], True, name_pref + "br4_cv1_")
        return array_ops.concat([tensor1, tensor2, tensor3, tensor4], 3)

    def incep_block_4(self, tensor):
        tensor1 = self.conv_layer(tensor, [3, 288, 384, 2], False, "ic4_br1_cv1_")
        tensor2 = self.conv_layer(tensor, [1, 288, 64, 1], True, "ic4_br2_cv1_")
        tensor2 = self.conv_layer(tensor2, [3, 64, 96, 1], True, "ic4_br2_cv2_")
        tensor2 = self.conv_layer(tensor2, [3, 96, 96, 2], False, "ic4_br2_cv3_")
        tensor3 = self.pool_layer(tensor, True, [3, 2], True)
        return array_ops.concat([tensor1, tensor2, tensor3], 3)

    def incep_block_b(self, tensor, x_depth, name_pref):
        tensor1 = self.conv_layer(tensor, [1, 768, 192, 1], True, name_pref + "br1_cv1_")
        tensor2 = self.conv_layer(tensor, [1, 768, x_depth, 1], True, name_pref + "br2_cv1_")
        tensor2 = self.conv_layer(tensor2, [[1, 7], x_depth, x_depth, 1], True, name_pref + "br2_cv2_")
        tensor2 = self.conv_layer(tensor2, [[7, 1], x_depth, 192, 1], True, name_pref + "br2_cv3_")
        tensor3 = self.conv_layer(tensor, [1, 768, x_depth, 1], True, name_pref + "br3_cv1_")
        tensor3 = self.conv_layer(tensor3, [[7, 1], x_depth, x_depth, 1], True, name_pref + "br3_cv2_")
        tensor3 = self.conv_layer(tensor3, [[1, 7], x_depth, x_depth, 1], True, name_pref + "br3_cv3_")
        tensor3 = self.conv_layer(tensor3, [[7, 1], x_depth, x_depth, 1], True, name_pref + "br3_cv4_")
        tensor3 = self.conv_layer(tensor3, [[1, 7], x_depth, 192, 1], True, name_pref + "br3_cv5_")
        tensor4 = self.pool_layer(tensor, False, [3, 1], True)
        tensor4 = self.conv_layer(tensor4, [1, 768, 192, 1], True, name_pref + "br4_cv1_")
        return array_ops.concat([tensor1, tensor2, tensor3, tensor4], 3)

    def incep_block_9(self, tensor):
        tensor1 = self.conv_layer(tensor, [1, 768, 192, 1], True, "ic9_br1_cv1_")
        tensor1 = self.conv_layer(tensor1, [3, 192, 320, 2], False, "ic9_br1_cv2_")
        tensor2 = self.conv_layer(tensor, [1, 768, 192, 1], True, "ic9_br2_cv1_")
        tensor2 = self.conv_layer(tensor2, [[1, 7], 192, 192, 1], True, "ic9_br2_cv2_")
        tensor2 = self.conv_layer(tensor2, [[7, 1], 192, 192, 1], True, "ic9_br2_cv3_")
        tensor2 = self.conv_layer(tensor2, [3, 192, 192, 2], False, "ic9_br2_cv4_")
        tensor3 = self.pool_layer(tensor, True, [3, 2], False)
        return array_ops.concat([tensor1, tensor2, tensor3], 3)

    def incep_block_c(self, tensor, input_depth, name_pref):
        tensor1 = self.conv_layer(tensor, [1, input_depth, 320, 1], True, name_pref + "br1_cv1_")
        tensor2 = self.conv_layer(tensor, [1, input_depth, 384, 1], True, name_pref + "br2_cv1_")
        tensor2a = self.conv_layer(tensor2, [[1, 3], 384, 384, 1], True, name_pref + "br2_cv2a_")
        tensor2b = self.conv_layer(tensor2, [[3, 1], 384, 384, 1], True, name_pref + "br2_cv2b_")
        tensor2 = array_ops.concat([tensor2a, tensor2b], 3)
        tensor3 = self.conv_layer(tensor, [1, input_depth, 448, 1], True, name_pref + "br3_cv1_")
        tensor3 = self.conv_layer(tensor3, [3, 448, 384, 1], True, name_pref + "br3_cv2_")
        tensor3a = self.conv_layer(tensor3, [[1, 3], 384, 384, 1], True, name_pref + "br3_cv3a_")
        tensor3b = self.conv_layer(tensor3, [[3, 1], 384, 384, 1], True, name_pref + "br3_cv3b_")
        tensor3 = array_ops.concat([tensor3a, tensor3b], 3)
        tensor4 = self.pool_layer(tensor, False, [3, 1], True)
        tensor4 = self.conv_layer(tensor4, [1, input_depth, 192, 1], True, name_pref + "br4_cv1_")
        return array_ops.concat([tensor1, tensor2, tensor3, tensor4], 3)


    def infer_bottleneck(self, tensor):
        # 299 x 299 x 3

        # front layers
        tensor = self.conv_layer(tensor, [3, 3, 32, 2], False, "fr_cv1_")
        # 149 x 149 x 32
        tensor = self.conv_layer(tensor, [3, 32, 32, 1], False, "fr_cv2_")
        # 147 x 147 x 32
        tensor = self.conv_layer(tensor, [3, 32, 64, 1], True, "fr_cv3_")
        # 147 x 147 x 64
        tensor = self.pool_layer(tensor, True, [3, 2], False)
        # 73 x 73 x 64
        tensor = self.conv_layer(tensor, [1, 64, 80, 1], False, "fr_cv4_")
        # 73 x 73 x 80
        tensor = self.conv_layer(tensor, [3, 80, 192, 1], False, "fr_cv5_")
        # 71 x 71 x 192
        tensor = self.pool_layer(tensor, True, [3, 2], False)
        # 35 x 35 x 192

        # inception layers
        tensor = self.incep_block_a(tensor, 192, 32, "ic1_")
        # 35 x 35 x 256
        tensor = self.incep_block_a(tensor, 256, 64, "ic2_")
        # 35 x 35 x 288
        tensor = self.incep_block_a(tensor, 288, 64, "ic3_")
        # 35 x 35 x 288
        tensor = self.incep_block_4(tensor)
        # 17 x 17 x 768
        tensor = self.incep_block_b(tensor, 128, "ic5_")
        # 17 x 17 x 768
        tensor = self.incep_block_b(tensor, 160, "ic6_")
        # 17 x 17 x 768
        tensor = self.incep_block_b(tensor, 160, "ic7_")
        # 17 x 17 x 768
        tensor = self.incep_block_b(tensor, 192, "ic8_")
        tensor_aux = tensor
        # 17 x 17 x 768
        tensor = self.incep_block_9(tensor)
        # 8 x 8 x 1280
        tensor = self.incep_block_c(tensor, 1280, "ic10_")
        # 8 x 8 x 2048
        tensor = self.incep_block_c(tensor, 2048, "ic11_")
        # 8 x 8 x 2048

        # aux layers
        tensor_aux = self.pool_layer(tensor_aux, False, [5, 3], True)
        # 6 x 6 x 768
        tensor_aux = self.conv_layer(tensor_aux, [1, 768, 128, 1], True, "au_cv1_")
        # 6 x 6 x 128
        tensor_aux = self.conv_layer(tensor_aux, [5, 128, 768, 1], False, "au_cv2_")
        # 2 x 2 x 768
        tensor_aux = self.conv_layer_ex(tensor_aux, [1, 768, self.output_depth, 1], True, 0.001, "au_cv3_")
        # 2 x 2 x output_depth

        # back layers
        tensor = self.pool_layer(tensor, False, [8, 1], False, name = "bottleneck")
        # 1 x 1 x 2048
        return tensor, tensor_aux

    def infer_back(self, tensor):
        tensor = self.conv_layer_ex(tensor, [1, 2048, self.output_depth, 1], True, self.weights_stddev, "ba_cv1_")
        # 1 x 1 x output_depth
        tensor = tf.reshape(tensor, [-1, self.output_depth], name = "y")
        return tensor

    def __init__(self):
        self.steps = 3001
        self.info_per_steps = 100
        self.save_per_steps = 500

        self.adam_learning_rate_base = 0.0001
        self.regularization_rate = 0.00004
        self.weights_stddev = 0.1

    def train(self, next_batch, next_batch_arg, validations):
        x = tf.placeholder(tf.float32, [None, 299, 299, 3], name = "x")
        y_ = tf.placeholder(tf.float32, [None, self.output_size], name = "y_")
        i_step = tf.Variable(0, trainable = False)
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularization_rate)

        y, _ = self.infer_bottleneck(x)
        y = self.infer_back(y)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
        loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('l2reg'))
        train_op = tf.train.AdamOptimizer(self.adam_learning_rate_base).minimize(loss, global_step = i_step)

        accs = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(accs, tf.float32), name = "accuracy")

        saver = tf.train.Saver()
        xv, yv = validations()
        with tf.Session() as s:
            cp = tf.train.get_checkpoint_state(self.save_path)
            if cp and cp.model_checkpoint_path:
                saver.restore(s, cp.model_checkpoint_path)
            else:
                s.run(tf.global_variables_initializer())

            for i in range(1, self.steps):
                xb, yb = next_batch(next_batch_arg)
                _, loss_val, i_step_val = s.run([train_op, loss, i_step], feed_dict = {x: xb, y_: yb})

                if i % self.info_per_steps == 0:
                    print("batch train step: %d, loss: %g" % (i_step_val, loss_val))
                    accuracy_val = s.run(accuracy, feed_dict = {x: xv, y_: yv})
                    print("accuracy: %g" % accuracy_val)

                if i % self.save_per_steps == 0:
                    saver.save(s, self.save_path + self.save_name, global_step = i_step)
