import tensorflow as tf
import numpy as np


class BaseModel(tf.keras.Model):
    def __init__(self, inputs, outputs, name) -> None:
        super().__init__(inputs, outputs, name=name)
        self.__input_shape = inputs[0].shape.as_list()
        #print(list(self.__input_shape))

    def save_pb(self, pbpath, logpath=None):
        from tensorflow.python.framework import graph_util
        session = tf.keras.backend.get_session()
        print('input is :', self.input.op.name)
        print('output is:', self.output.op.name)
        graph = session.graph
        with graph.as_default():
            output_names = [self.output.op.name]
            # graphdef_inf = graph_util.remove_training_nodes(session.graph.as_graph_def())
            constant_graph = graph_util.convert_variables_to_constants(session, session.graph.as_graph_def(), output_names)
            with tf.gfile.GFile(pbpath, mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            if logpath is not None:
                writer = tf.summary.FileWriter('./log', constant_graph)
                writer.close()

    def save_rknn(self, rknnpath, verbose=True, verbose_file=None, input_mean_value='0 0 0 1', input_channels='0 1 2', do_quantization=True, pre_compile=True):
        TMP_PB_PATH = './tmp.pb'
        from rknn.api import RKNN
        self.save_pb(TMP_PB_PATH)

        rknn = RKNN(verbose=verbose, verbose_file=verbose_file)

        print('--> config model')
        rknn.config(channel_mean_value=input_mean_value, reorder_channel=input_channels)
        print('done')

        print('--> Loading pb, input shape = ' + str([self.__input_shape]))
        ret = rknn.load_tensorflow(tf_pb=TMP_PB_PATH,
                                   inputs=[self.input.op.name],
                                   outputs=[self.output.op.name],
                                   input_size_list=[list(self.__input_shape)])
        if ret != 0:
            print('Load pb failed! Ret = {}'.format(ret))
            exit(ret)
        print('done')

        print('--> Building model')
        ret = rknn.build(do_quantization=do_quantization, dataset='./rknn_quantization.txt', pre_compile=pre_compile)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        print('--> Export RKNN model')
        ret = rknn.export_rknn(rknnpath)
        if ret != 0:
            print('Export rknn failed!')
            exit(ret)
        print('done')

        rknn.release()

