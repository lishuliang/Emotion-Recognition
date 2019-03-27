
import tensorflow as tf
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

path = sys.path[0]

def picture_read(file_list):
  """
  狗图片读取案例
  """
  #1.构造文件名队列
  file_queue = tf.train.string_input_producer(file_list)

  #2.读取与解码
  #读取阶段
  reader = tf.WholeFileReader()
  #key文件名 value一张图片的原始编码形式
  key, value = reader.read(file_queue)
  print('key:\n', key)
  print('value:\n', value)
  #解码阶段
  image = tf.image.decode_jpeg(value)
  #print('image:\n', image)

  #图像的形状类型修改
  image_resized = tf.image.resize_images(image, [100,100])
  print('image_resized:\n', image_resized)

  #静态形状修改
  image_resized.set_shape(shape = [100, 100, 3])
  print('image_resized:\n', image_resized)

  #3.批处理
  image_batch = tf.train.batch([image_resized], batch_size = 100, num_threads = 1, capacity = 100)
  print('image_batch:\n', image_batch)

  #开启会话
  with tf.Session() as sess:
    #开启线程
    #创建线程协调员
    coord = tf.train.Coordinator()
    theards = tf.train.start_queue_runners(sess = sess, coord = coord)
    
    key_new, value_new, image_new, image_resized_new, image_batch_new = sess.run([key, value, image, image_resized, image_batch_new])
    #print('key_new:\n', key_new)
    #print('value_new:\n', value_new)
    #print('image_new:\n', image_new)
    #print('image_resized_new:\n', image_resized_new)
    #print('image_batch_new:\n', image_batch_new)

    #回收线程
    coord.request_stop()
    coord.join(theards)

  return None

if __name__ == '__main__':
  #构造路径+文件名的列表
  filename = os.listdir(path +'\dog')
  #print(filename)
  #拼接路径+文件名
  file_list = [os.path.join(path + '\dog', file) for file in filename]
  #print(file_list)
  picture_read(file_list)