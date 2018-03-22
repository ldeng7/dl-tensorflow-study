import tensorflow as tf


def Init_eval(eval, save_path):
    eval.graph = tf.Graph()
    eval.session = tf.Session(graph = eval.graph)
    cp = tf.train.get_checkpoint_state(save_path)
    with eval.graph.as_default():
        saver = tf.train.import_meta_graph(cp.model_checkpoint_path + ".meta")
    saver.restore(eval.session, cp.model_checkpoint_path)
