# coding=utf-8
import math
import numpy as np
import tensorflow as tf
import os

# https://medium.com/@moritzkrger/speeding-up-keras-with-tfrecord-datasets-5464f9836c36
# https://medium.com/tensorflow/training-and-serving-ml-models-with-tf-keras-fd975cc0fa27

class CASIA_WebFace:
    __tfrecord_path = ''
    __origin_path = ''

    def __init__(self, tfrecord_path, origin_path):
        self.__tfrecord_path = tfrecord_path
        self.__origin_path = origin_path

    def create_tfrecords(self, dir_path):
        writer = tf.python_io.TFRecordWriter(self.__tfrecord_path)

        peoples_dir = os.listdir(dir_path)
        for people in peoples_dir:
            print("people id:" + people)
            if not os.path.isdir(dir_path + '/' + people):
                continue
            for image_name in os.listdir(dir_path + '/' + people):
                img = cv2.imread(dir_path + '/' + people + '/' + image_name)

                features = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(people)]))
                }

                tf_features = tf.train.Features(feature=features)
                tf_example = tf.train.Example(features=tf_features)
                tf_serialized = tf_example.SerializeToString()

                writer.write(tf_serialized)
        writer.close()

    def __parse_dataset(self, proto):
        keys_to_features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(proto, keys_to_features)
        parsed_features['image'] = tf.decode_raw(parsed_features['image'], tf.uint8)
        return parsed_features['image'], parsed_features["label"]

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.__tfrecord_path)
        dataset = dataset.map(self.__parse_dataset, num_parallel_calls=8)

        dataset.repeat()                # 重复
        dataset.shuffle(1000)           # 混洗
        dataset = dataset.batch(32)     # 分批

        iterator = dataset.make_one_shot_iterator()
        image, label = iterator.get_next()
        image = tf.reshape(image, [-1, 149, 149, 3]) #?
        label = tf.one_hot(label, 100)
        return image, label


class CASIAWebFaceSequence(tf.keras.utils.Sequence):
    class Sample:
        label = 0
        images = []

    def __init__(self, path, target_shape, batch_size=1, shuffle=True):
        self.__path = path
        self.__batch_size = batch_size
        self.__target_shape = target_shape
        self.__shuffle = shuffle
        self.__data = []
        self.__data_indexes = []
        self.__image_num = 0

        label = 0
        for clz in os.listdir(self.__path):
            smp1 = self.Sample()
            smp1.label = label
            label_path = os.path.join(self.__path, clz)
            if os.path.isdir(label_path):
                print('classes image number:' + str(len(os.listdir(label_path))))
                for imgfile in os.listdir(label_path):
                    smp1.images.append(os.path.join(label_path, imgfile))
                    self.__image_num += 1
                label += 1
                self.__data.append(smp1)
            else:
                print('> no dir:' + label_path)

        self.__data_indexes = np.arange(len(self.__data))
        print('Labels size:' + str(label))

    def __len__(self):  # 每个epoch的迭代次数
        print('Sequence: get len:' + str(len(self.__data)))
        return int(math.ceil(self.__image_num / float(self.__batch_size) / 3.))

    def __getitem__(self, index):
        batch_indexs_pos = self.__data_indexes[index]
        batch_indexs_neg = self.__data_indexes[index+1]

        batch_images = []
        batch_labels = [] # no use

        image_size = len(self.__data[batch_indexs_pos].images)
        anchor_index = np.random.randint(int(image_size/2), size=self.__batch_size)
        positi_index = np.random.randint(low=int(image_size/2), high=image_size, size=self.__batch_size)
        neviga_index = np.random.randint(int(len(self.__data[batch_indexs_neg].images)), size=self.__batch_size)

        for k in range(self.__batch_size):
            batch_images.append(tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(self.__data[batch_indexs_pos].images[anchor_index[k]], target_size=self.__target_shape)
            ))  # anchor
            batch_labels.append([1, 0, 0])

        for k in range(self.__batch_size):
            batch_images.append(tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(self.__data[batch_indexs_pos].images[positi_index[k]], target_size=self.__target_shape)
            ))  # positive
            batch_labels.append([0, 1, 0])

        for k in range(self.__batch_size):
            batch_images.append(tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(self.__data[batch_indexs_neg].images[neviga_index[k]], target_size=self.__target_shape)
            ))  # negative
            batch_labels.append([0, 0, 1])

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__data_indexes)
        pass

    def getitem_test(self):
        self.on_epoch_end()
        imgs, labels = self.__getitem__(1)
        print('img size:' + str(len(imgs)))
        print('label size:' + str(len(labels)))
        return imgs, labels



if __name__ == '__main__':
    #webface = CASIA_WebFace('./casia-webface.tfrecords', './CASIA-Webr
    #webface.create_tfrecords('CASIA-WebFace')
    #webface.get_dataset_from_origin_path()
    #webface = CASIAWebFaceSequence('./CASIA-WebFace', target_shape=[149, 149])
    #webface.test()

    # image_size = 9
    # anchor_index = np.random.randint(int(image_size / 2), size=7)
    # positi_index = np.random.randint(low=int(image_size / 2), high=image_size, size=7)
    # print('anchor index:' + str(anchor_index))
    # print('positive index:' + str(positi_index))
    pass