import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = sys.path[0]
class Cifar(object):
  
    def __init__(self):
        #初始化操作
        self.height = 32
        self.width = 32
        self.channels = 3

        #字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes
    
    def read_binary(self):

        file_name = os.listdir('cifar-10')
        #print('file_name:\n', file_name)
        file_list = [os.path.join('./cifar-10/', file) for file in file_name if file[-3:] == 'bin']
        #print('file_list:\n', file_list)

        #1.构造文件名队列
        file_queue = tf.train.string_input_producer(file_list)

        #2.读取与解码
        #读取阶段
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        #key文件名 value一个样本
        key, value = reader.read(file_queue)
        print('key:\n', key)
        print('value:\n', value)

        #解码阶段
        decoded = tf.decode_raw(value, tf.uint8)
        print('decoded:\n', decoded)

        #将目标值和特征值分开
        label = tf.slice(decoded, [0], [self.label_bytes])
        image = tf.slice(decoded, [self.label_bytes], [self.image_bytes])
        print('label:\n', label)
        print('image:\n', image)

        #调整图片形状
        image_reshaped = tf.reshape(image, shape = [self.channels, self.height, self.width])
        print('image_reshaped:\n', image_reshaped)

        #转置
        image_transposed = tf.transpose(image_reshaped, [1, 2, 0])
        print('image_transposed:\n', image_transposed)

        #调整图像类型
        image_cast = tf.cast(image_transposed, tf.float32)
        print('image_cast:\n', image_cast)

        #3.批处理
        label_batch, image_batch = tf.train.batch([label, image_cast], batch_size = 100, num_threads = 1, capacity = 100)
        print('label_batch:\n', label_batch)
        print('image_batch:\n', image_batch)

        #开启会话
        with tf.Session() as sess:
            #开启线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            key_new, value_new, decoded_new, label_new, image_new, image_reshaped_new, image_transposed_new, label_value, image_value = sess.run([key, value, decoded, label, image, image_reshaped, image_transposed, label_batch, image_batch])
            #print('key_new:\n', key_new)
            #print('value_new:\n', value_new)
            #print('decoded_new:\n', decoded_new)
            #print('label_new:\n', label_new)
            #print('image_new:\n', image_new)
            #print('image_reshaped_new:\n', image_reshaped_new)
            #print('image_transposed_new:\n', image_transposed_new)

            #回收线程
            coord.request_stop()
            coord.join(threads)

        return image_value, label_value
    
    def write_to_tfrecords(self, image_batch, label_batch):
        """
        将样本特征值和目标值一起写入tfrecords文件
        """
        with tf.python_io.TFRecordWriter(path + '\cifar10.tfrecords') as write:
            #循环构造example对象，并序列化写入文件
            for i in range(100):
                image = image_batch[i].tostring()
                label = label_batch[i][0]
                #print('tfrecords_image:\n', image)
                #print('tfrecords_label:\n', label)
                example = tf.train.Example(features = tf.train.Features(feature = {
                    'image':tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                    'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
                }))
                
                #将序列化后的example写入文件
                write.write(example.SerializeToString())

        return None

    def read_tfrecords(self):
        """
        读取tfrecords文件
        """
        #1、构造文件队列名
        file_queue = tf.train.string_input_producer([path + '\cifar10.tfrecords'])

        #2、读取与解码
        #读取
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)

        #解析example
        feature = tf.parse_single_example(value, features = {
            'image':tf.FixedLenFeature([], tf.string),
            'label':tf.FixedLenFeature([], tf.int64),
        })
        image = feature['image']
        label = feature['label']
        print('read_tf_image:\n', image)
        print('read_tf_label:\n', label)

        #解码
        image_decoded = tf.decode_raw(image, tf.uint8)
        print('image_decoded:\n', image_decoded)

        #图像形状调整
        image_reshaped = tf.reshape(image_decoded, [self.height, self.width, self.channels])
        print('image_reshaped:\n', image_reshaped)

        #3、构造批处理队列
        image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size = 100, num_threads = 1, capacity = 100)
        print('image_batch:\n', image_batch)
        print('label_batch:\n', label_batch)

        #开启会话
        with tf.Session() as sess:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)

            image_value, label_value = sess.run([image, label])
            #print('image_value:\n', image_value) 
            #print('label_value:\n', label_value) 

            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':

    #实例化Cifar
    cifar = Cifar()
    #image_value, label_value = cifar.read_binary()
    #cifar.write_to_tfrecords(image_value, label_value)
    cifar.read_tfrecords()