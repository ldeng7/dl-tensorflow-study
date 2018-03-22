import os
import model.models as models
import res.jpeg_set as jpeg_set

DL_BASE_DIR = os.environ["DL_BASE_DIR"]
SAVE_PATH = DL_BASE_DIR + r"/record/blossom7-cnn-inception3/"
BATCH_SIZE = 10

jsf = jpeg_set.JpegSetFactory()
jsf.n_label_index = 7
jsf.label_indices = {
    "anthurium": 0,
    "dandelion": 1,
    "lotus": 2,
    "morning glory": 3,
    "rose": 4,
    "sunflower": 5,
    "tulip": 6,
}


class Eval:

    def __init__(self):
        models.Init_eval(self, SAVE_PATH)

    def Run(self, file_path):
        val = self.session.run(self.graph.get_tensor_by_name("y:0"), feed_dict = {
            self.graph.get_tensor_by_name("x:0"): jsf.PreprocessFile(file_path),
        })
        return val.tolist()[0]

    def Close(self):
        self.session.close()


def Preprocess(dir_path):
    jsf.Preprocess(dir_path)

def Train(dir_path):
    import network.cnn_inception3 as cnn_inception3

    js = jsf.Load(dir_path)
    net = cnn_inception3.Inception3()
    net.conf_save_path = SAVE_PATH
    net.conf_save_name = "blossom7-cnn-inception3"
    net.layout_fc_size = [1024]
    net.layout_output_size = jsf.n_label_index
    net.data_next_batch = lambda: js.Next_train_batch(BATCH_SIZE)
    net.data_validations = lambda: js.Validations()

    net.Train()
