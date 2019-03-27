import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = sys.path[0]

""" tf.app.flags.DEFINE_integer('is_train', -1, '指定是否训练模型，还是拿数据去预测')
FLAGS = tf.app.flags.FLAGS """

def full_connection():
    """
    用全连接来对手写数字进行识别
    """
    #1、准备数据
    with tf.variable_scope('mnist_data'):
        mnist = input_data.read_data_sets(path + '\mnist_data', one_hot = True)
        x = tf.placeholder(dtype = tf.float32, shape = [None, 784])
        y_true = tf.placeholder(dtype = tf.float32, shape = [None, 10])

    
    y_predict = create_model(x)

    #3.构建损失函数
    with tf.variable_scope('softmax_crossentrop'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_predict))
    
    #4.优化损失函数
    with tf.variable_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

    #5、准确率计算
    with tf.variable_scope('accuracy'):
    #1)比较输出的结果最大值所在位置和真实值的最大值所在位置
        equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
        #2）求平均
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        image, label = mnist.train.next_batch(100)

        print('训练前，损失为%f:' % sess.run(loss, feed_dict = {x: image, y_true: label}))

        #开始训练
        for i in range(3000):
            _, loss_value, accuracy_value = sess.run([train_op, loss, accuracy], feed_dict = {x: image, y_true: label})
            print('第%d训练，损失为%f，准确率为%f' % (i + 1, loss_value, accuracy_value))

    return None

def create_model(x):
    """
    构建卷积神经网络
    """
    #1)第一个卷积大层
    #将x[None, 784]形状进行修改
    with tf.variable_scope('conv1'):
        input_x = tf.reshape(x, shape = [-1, 28, 28, 1])
        #卷积层
        #定义filter和偏置
        conv1_weights = create_weights(shape = [5, 5, 1, 32])
        conv1_bias = create_weights(shape = [32])
        conv1_x = tf.nn.conv2d(input = input_x, filter = conv1_weights, strides = [1, 1, 1, 1], padding = 'SAME') + conv1_bias

        #激活层
        relu1_x = tf.nn.relu(conv1_x)

        #池化层
        pool1_x = tf.nn.max_pool(value = relu1_x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    #2)第二个卷积大层
    with tf.variable_scope('conv2'):
        #卷积层
        #定义filter和偏置
        conv2_weights = create_weights(shape = [5, 5, 32, 64])
        conv2_bias = create_weights(shape = [64])
        conv2_x = tf.nn.conv2d(input = pool1_x, filter = conv2_weights, strides = [1, 1, 1, 1], padding = 'SAME') + conv2_bias

        #激活层
        relu2_x = tf.nn.relu(conv2_x)

        #池化层
        pool2_x = tf.nn.max_pool(value = relu2_x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    #3)全连接层
    with tf.variable_scope('full_connection'):
        x_fc = tf.reshape(pool2_x, shape = [-1, 7 * 7 * 64])
        weights_fc = create_weights(shape = [7 * 7 * 64,10])
        bais_fc = create_weights(shape = [10])
        y_predict = tf.matmul(x_fc, weights_fc) + bais_fc

    return y_predict

def create_weights(shape):
    return tf.Variable(initial_value = tf.random_normal(shape = shape, stddev = 0.01))

if __name__ == '__main__':
    full_connection()