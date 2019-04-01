# coding=utf-8
import tensorflow as tf
import tensorflow.keras.backend as K
from models.base_model import BaseModel


class InceptionResnetV1(BaseModel):
    def __init__(self, input_shape=(149, 149, 3), classes=128, dropout_keep_prob=0.8):
        self.__classes = classes
        self.__dropout_keep_prob = dropout_keep_prob

        inputs = tf.keras.Input(shape=input_shape)

        # input stem
        x = self.InputStem(inputs)

        # 5 * inception-resnet-a
        for i in range(1, 6):
            x = self.InceptionResnetA(x, scale=0.17, idx=i)

        # reduction-a
        x = self.Reduction_A(x)

        # 10 * inception-resnet-b
        for i in range(1, 11):
            x = self.InceptionResnetB(x, scale=0.10, idx=i)

        # reduction-b
        x = self.Reduction_B(x)

        # 5 * inception-resnet-c
        for i in range(1, 6):
            x = self.InceptionResnetC(x, scale=0.20, idx=i)

        x = self.InceptionResnetC(x, scale=1, idx=6)

        # Classification
        x = tf.keras.layers.GlobalAveragePooling2D(name='AvgPool')(x)
        x = tf.keras.layers.Dropout(rate=1-self.__dropout_keep_prob, name='Dropout')(x)
        x = tf.keras.layers.Dense(self.__classes, use_bias=False, name='Bottleneck')(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_bn')(x)
        #x = tf.keras.layers.Dense(100, activation=tf.keras.activations.softmax)(x)

        super(InceptionResnetV1, self).__init__(inputs, x, name='InceptionResnetV1')

    def Conv2D_Bn_Relu(self, x, filters, kernel_size, strides, padding='same', name=''):
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_momentum = 0.995
        bn_epsilon = 0.001
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, name=name+'_conv2d')(x)
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, momentum=bn_momentum, epsilon=bn_epsilon, scale=False, name=name+'_bn')(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu, name=name+'_relu')(x)
        return x

    def InputStem(self, x):
        x = self.Conv2D_Bn_Relu(x, 32, 3, strides=2, name='input_stem1')            # 149 x 149 x 32
        x = self.Conv2D_Bn_Relu(x, 32, 3, strides=1, name='input_stem2')            # 147 x 147 x 32
        x = self.Conv2D_Bn_Relu(x, 64, 3, strides=1, name='input_stem3')            # 147 x 147 x 64
        x = tf.keras.layers.MaxPool2D(3, strides=2, name='input_stem4_maxpool')(x)  # 73 x 73 x 64
        x = self.Conv2D_Bn_Relu(x, 80, 1, strides=1, name='input_stem5')            # 73 x 73 x 80
        x = self.Conv2D_Bn_Relu(x, 192, 3, strides=1, name='input_stem6')           # 71 x 71 x 192
        x = self.Conv2D_Bn_Relu(x, 256, 3, strides=2, name='input_stem7')           # 35 x 35 x 256
        return x

    # Inception-resnet-A / Block35
    def InceptionResnetA(self, x, scale=0.17, idx=0):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        branch0 = self.Conv2D_Bn_Relu(x, 32, 1, strides=1, name='irA'+str(idx)+'-b0')
        branch1 = self.Conv2D_Bn_Relu(x, 32, 1, strides=1, name='irA'+str(idx)+'-b1a')
        branch1 = self.Conv2D_Bn_Relu(branch1, 32, 3, strides=1, name='irA'+str(idx)+'-b1b')
        branch2 = self.Conv2D_Bn_Relu(x, 32, 1, strides=1, name='irA'+str(idx)+'-b2a')
        branch2 = self.Conv2D_Bn_Relu(branch2, 32, 3, strides=1, name='irA'+str(idx)+'-b2b')
        branch2 = self.Conv2D_Bn_Relu(branch2, 32, 3, strides=1, name='irA'+str(idx)+'-b2c')
        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name='irA'+str(idx)+'-concat')([branch0, branch1, branch2])
        up = tf.keras.layers.Conv2D(K.int_shape(x)[channel_axis], 1, strides=1, use_bias=True, name='irA'+str(idx)+'-conv2d')(mixed)
        up = tf.keras.layers.Lambda(lambda x: x*scale)(up)
        x = tf.keras.layers.Add()([x, up])
        x = tf.keras.layers.Activation(tf.keras.activations.relu, name='irA'+str(idx)+'-relu')(x)
        return x

    # Inception-resnet-B / Block17
    def InceptionResnetB(self, x, scale=0.10, idx=0):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        branch0 = self.Conv2D_Bn_Relu(x, 128, 1, strides=1, name='irB'+str(idx)+'-b0')
        branch1 = self.Conv2D_Bn_Relu(x, 128, 1, strides=1, name='irB'+str(idx)+'-b1a')
        branch1 = self.Conv2D_Bn_Relu(branch1, 128, [1, 7], strides=1, name='irB'+str(idx)+'-b1b')
        branch1 = self.Conv2D_Bn_Relu(branch1, 128, [7, 1], strides=1, name='irB'+str(idx)+'-b1c')
        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name='irB'+str(idx)+'-concat')([branch0, branch1])
        up = tf.keras.layers.Conv2D(K.int_shape(x)[channel_axis], 1, strides=1, use_bias=True, name='irB'+str(idx)+'-conv2d')(mixed)
        up = tf.keras.layers.Lambda(lambda x: x * scale)(up)
        x = tf.keras.layers.Add()([x, up])
        x = tf.keras.layers.Activation(tf.keras.activations.relu, name='irB' + str(idx) + '-relu')(x)
        return x

    # Inception-resnet-C / Block8
    def InceptionResnetC(self, x, scale=0.20, idx=0):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        branch0 = self.Conv2D_Bn_Relu(x, 192, 1, strides=1, name='irC'+str(idx)+'-b0')
        branch1 = self.Conv2D_Bn_Relu(x, 192, 1, strides=1, name='irC'+str(idx)+'-b1a')
        branch1 = self.Conv2D_Bn_Relu(branch1, 192, [1, 3], strides=1, name='irC'+str(idx)+'-b1b')
        branch1 = self.Conv2D_Bn_Relu(branch1, 192, [3, 1], strides=1, name='irC'+str(idx)+'-b1c')
        mixed = tf.keras.layers.Concatenate(axis=channel_axis, name='irC' + str(idx) + '-concat')([branch0, branch1])
        up = tf.keras.layers.Conv2D(K.int_shape(x)[channel_axis], 1, strides=1, use_bias=True, name='irC'+str(idx)+'-conv2d')(mixed)
        up = tf.keras.layers.Lambda(lambda x: x * scale)(up)
        x = tf.keras.layers.Add()([x, up])
        x = tf.keras.layers.Activation(tf.keras.activations.relu, name='irC' + str(idx) + '-relu')(x)
        return x

    def Reduction_A(self, x):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        branch0 = self.Conv2D_Bn_Relu(x, 384, 3, strides=2, padding='valid', name='reduA_conv2d_0a')
        branch1 = self.Conv2D_Bn_Relu(x, 192, 1, strides=1, name='reduA_conv2d_1a')
        branch1 = self.Conv2D_Bn_Relu(branch1, 192, 3, strides=1, name='reduA_conv2d_1b')
        branch1 = self.Conv2D_Bn_Relu(branch1, 256, 3, strides=2, padding='valid', name='reduA_conv2d_1c')
        branch2 = tf.keras.layers.MaxPool2D(3, strides=2, padding='valid', name='reduA_maxpool_2a')(x)
        net = tf.keras.layers.Concatenate(axis=channel_axis, name='reduA-concat')([branch0, branch1, branch2])
        return net

    def Reduction_B(self, x):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
        branch0 = self.Conv2D_Bn_Relu(x, 256, 1, strides=1, name='reduB_conv2d_0a')
        branch0 = self.Conv2D_Bn_Relu(branch0, 384, 3, strides=2, padding='valid', name='reduB_conv2d_0b')
        branch1 = self.Conv2D_Bn_Relu(x, 256, 1, strides=1, name='reduB_conv2d_1a')
        branch1 = self.Conv2D_Bn_Relu(branch1, 256, 3, strides=2, padding='valid', name='reduB_conv2d_1b')
        branch2 = self.Conv2D_Bn_Relu(x, 256, 1, strides=1, name='reduB_conv2d_2a')
        branch2 = self.Conv2D_Bn_Relu(branch2, 256, 3, strides=1, name='reduB_conv2d_2b')
        branch2 = self.Conv2D_Bn_Relu(branch2, 256, 3, strides=2, padding='valid', name='reduB_conv2d_2c')
        branch3 = tf.keras.layers.MaxPool2D(3, strides=2, padding='valid', name='reduB_maxpool_3a')(x)
        net = tf.keras.layers.Concatenate(axis=channel_axis, name='reduB-concat')([branch0, branch1, branch2, branch3])
        return net
