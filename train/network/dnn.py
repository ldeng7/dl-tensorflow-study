import tensorflow as tf


class TrainConf:

    def __init__(self):
        self.steps = 30001
        self.info_per_steps = 1000
        self.save_per_steps = 5000
        
        self.learning_rate_base = 0.8
        self.learning_rate_decay = 0.99
        self.learning_rate_decay_steps = 100
        self.regularization_rate = 0.0001
        self.weights_init_stddev = 0.1


def infer(conf, tensor, regularizer):
    sizes = conf.layers_input_size + [conf.output_size]
    for i in range(len(conf.layers_input_size)):
        weights = tf.Variable(tf.truncated_normal(
            [sizes[i], sizes[i + 1]],
            stddev = conf.weights_init_stddev,
            name = "weights_" + str(i + 1)
        ))
        tf.add_to_collection('losses', regularizer(weights))
        biases = tf.Variable(tf.constant(
            0.,
            shape = [sizes[i + 1]],
            name = "biases_" + str(i + 1)
        ))
        if i != len(conf.layers_input_size) - 1:
            tensor = tf.nn.relu(tf.matmul(tensor, weights) + biases)
        else:
            tensor = tf.add(tf.matmul(tensor, weights), biases, name = "y")
    return tensor


def batch_train(conf, next_batch, next_batch_arg, validations):
    x = tf.placeholder(tf.float32, [None, conf.layers_input_size[0]], name = "x")
    y_ = tf.placeholder(tf.float32, [None, conf.output_size], name = "y_")
    i_step = tf.Variable(0, trainable = False)

    y = infer(conf, x, tf.contrib.layers.l2_regularizer(conf.regularization_rate))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        conf.learning_rate_base,
        i_step,
        conf.learning_rate_decay_steps,
        conf.learning_rate_decay,
        staircase = True
    )
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = i_step)

    accs = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(accs, tf.float32), name = "accuracy")

    saver = tf.train.Saver()
    xv, yv = validations()
    with tf.Session() as s:
        cp = tf.train.get_checkpoint_state(conf.save_path)
        if cp and cp.model_checkpoint_path:
            saver.restore(s, cp.model_checkpoint_path)
        else:
            s.run(tf.global_variables_initializer())

        for i in range(1, conf.steps):
            xb, yb = next_batch(next_batch_arg)
            _, loss_val, i_step_val = s.run([train_op, loss, i_step], feed_dict = {x: xb, y_: yb})

            if i % conf.info_per_steps == 0:
                print("batch train step: %d, loss: %g" % (i_step_val, loss_val))
                accuracy_val = s.run(accuracy, feed_dict = {x: xv, y_: yv})
                print("accuracy: %g" % accuracy_val)

            if i % conf.save_per_steps == 0:
                saver.save(s, conf.save_path + conf.save_name, global_step = i_step)
