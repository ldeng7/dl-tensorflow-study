import sys, time
from tensorflow.examples.tutorials.mnist import input_data
import dnn

BATCH_SIZE = 100

mnist = input_data.read_data_sets(r"data/mnist", one_hot = True)
conf = dnn.TrainConf()
conf.save_path = r"./save/dnn-mnist/"
conf.save_name = "dnn-mnist"
conf.input_size = [784, 500]
conf.output_size = 10
conf.learning_rate_decay_steps = mnist.train.num_examples / BATCH_SIZE
    
dnn.batch_train(conf,
    lambda: mnist.train.next_batch(BATCH_SIZE),
    lambda: (mnist.validation.images, mnist.validation.labels))
