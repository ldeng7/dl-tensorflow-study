import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r"data/mnist", one_hot = True)
cp = tf.train.get_checkpoint_state(r"./save/dnn-mnist/")
if cp and cp.model_checkpoint_path:
    saver = tf.train.import_meta_graph(cp.model_checkpoint_path + ".meta")
    with tf.Session() as s:
        saver.restore(s, cp.model_checkpoint_path)
        g = tf.get_default_graph()
        accuracy_val = s.run(g.get_tensor_by_name("accuracy:0"), feed_dict = {
            g.get_tensor_by_name("x:0"): mnist.test.images,
            g.get_tensor_by_name("y_:0"): mnist.test.labels
        })
        print("accuracy: %g" % accuracy_val)
