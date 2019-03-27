import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
path = sys.path[0]

def full_connection():
    """
    用全连接来对手写数字进行识别
    """
    #1、准备数据
    mnist = input_data.read_data_sets(path + '\mnist_data', one_hot = True)
    x = tf.placeholder(dtype = tf.float32, shape = [None, 784])
    y_true = tf.placeholder(dtype = tf.float32, shape = [None, 10])

    #2.构建模型
    Weights = tf.Variable(initial_value = tf.random_normal(shape = [784, 10]))
    bias = tf.Variable(initial_value = tf.random_normal(shape = [10]))
    y_predict = tf.matmul(x, Weights) + bias

    #3.构建损失函数
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_predict))
    
    #4.优化损失函数
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(error)

    #5、准确率计算
    #1)比较输出的结果最大值所在位置和真实值的最大值所在位置
    equal_list = tf.equal(tf.arg_max(y_true, 1), tf.arg_max(y_predict, 1))
    #2）求平均
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        image, label = mnist.train.next_batch(100)

        print('训练前，损失为%f:' % sess.run(error, feed_dict = {x: image, y_true: label}))

        #开始训练
        for i in range(3000):
            _, loss, accuracy_value = sess.run([optimizer, error, accuracy], feed_dict = {x: image, y_true: label})
            print('第%d训练，损失为%f，准确率为%f' % (i + 1, loss, accuracy_value))

    return None

if __name__ == '__main__':
    full_connection()