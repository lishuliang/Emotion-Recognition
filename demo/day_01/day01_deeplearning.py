
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def tensorflow_demo():

    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：\n", c_t)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print('c_t_value：\n', c_t_value)

    return None

def graph_demo():
    """
    查看默认图：
    1.调用方法
    2.查看属性
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：\n",c_t)
    default_g = tf.get_default_graph()
    print('default_g:\n',default_g)
    
    print('c_t的图属性:\n',c_t.graph)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print('c_t_value：\n',c_t_value)
        print('sess的图属性:\n',sess.graph)

    #自定义图
    new_g = tf.Graph()
    #在自己的图中定义数据和操作
    with new_g.as_default():
        a_new = tf.constant(20)
        b_new = tf.constant(30)
        c_new = a_new + b_new
        print('c_new:\n',c_new)
        print('c_new的图属性:\n',c_new.graph)
    
    #开启new_g的会话
    with tf.Session(graph = new_g) as new_sess:
        c_new_value = new_sess.run(c_new)
        print('c_new_value:\n',c_new_value)
        print('new_sess的图属性:\n',new_sess.graph)
        #1)将图写入本地events文件
        tf.summary.FileWriter('./demo/day_01/summary',graph = new_sess.graph)

    return None

def session_demo():
    """ 
    会话的演示
    """
    # TensorFlow实现加法运算
    a_t = tf.constant(2)
    b_t = tf.constant(3)
    c_t = a_t + b_t
    print("TensorFlow加法运算的结果：\n",c_t)
    default_g = tf.get_default_graph()
    print('default_g:\n',default_g)
    
    print('c_t的图属性:\n',c_t.graph)

    # 开启会话
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True,
                    log_device_placement = True)) as sess:
        c_t_value = sess.run(c_t)
        print('c_t_value：\n',c_t_value)
        print('sess的图属性:\n',sess.graph)

    return None

def tensor_demo():
    """ 
    张量的演示
    """
    tensor1 = tf.constant(4.0)
    tensor2 = tf.constant([1,2,3,4])
    tensor3 = tf.constant([[4],[9],[16],[25]], dtype = tf.int32)

    print('tensor1:\n',tensor1)
    print('tensor2:\n',tensor2)
    print('tensor3:\n',tensor3)
        
    #张量类型修改
    tensor3_cast = tf.cast(tensor3, dtype = tf.float32)
    print('tensor3_cast:\n',tensor3_cast)

    #更新改变静态形状,没有完全固定下来的静态形状
    a_p = tf.placeholder(dtype = tf.float32, shape = [None, None])
    b_p = tf.placeholder(dtype = tf.float32, shape = [None, 10])
    c_p = tf.placeholder(dtype = tf.float32, shape = [3, 2])
    print('a_p:\n',a_p)    
    print('b_p:\n',b_p) 
    # a_p.set_shape([2, 1])
    # b_p.set_shape([2, 10])
    #动态改变形状
    a_p_reshape = tf.reshape(a_p, shape = [2, 3, 1])
    print('a_p_shape:\n',a_p_reshape)    
    print('a_p:\n',a_p)    
    print('b_p:\n',b_p)    
    print('c_p:\n',c_p)    

    return None

def variable_demo():
    """
    变量的演示
    """
    with tf.variable_scope('my_scope'):   #修改命名空间
        a = tf.Variable(initial_value = 50)
        b = tf.Variable(initial_value = 40)
        c = tf.add(a, b)
    print('a:\n',a)
    print('b:\n',b)
    print('c:\n',c)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        #运行初始化
        sess.run(init)
        a_value, b_value, c_value = sess.run([a, b, c])
        print('a_value:\n',a_value)
        print('b_value:\n',b_value)
        print('c_value:\n',c_value)

    return None

def liner_regression():
    """ 
    自实现一个线性回归
    """
    with tf.variable_scope('prepare_data'):
        #1）准备数据
        X = tf.random_normal(shape = [100, 1], name = 'feature')
        y_true = tf.matmul(X, [[0.8]]) + 0.7
    
    with tf.variable_scope('creat_model'):
        #2）构造模型
        weights =  tf.Variable(initial_value = tf.random_normal(shape = [1, 1]), name = 'Weights')
        bias = tf.Variable(initial_value = tf.random_normal(shape = [1, 1]), name = 'Bias')
        y_predict = tf.matmul(X, weights) + bias

    with tf.variable_scope('loss_function'):
        #3）构造损失函数
        error = tf.reduce_mean(tf.square(y_predict - y_true))

    with tf.variable_scope('optimizer'):
        #4）优化损失
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(error)

    #2_收集变量
    tf.summary.scalar('error', error)
    tf.summary.histogram('weights', weights)
    tf.summary.histogram('bias', bias)

    #3_合并变量
    merged = tf.summary.merge_all()

    #创建Saver对象
    saver = tf.train.Saver()

    #显示的初始化变量
    init = tf.global_variables_initializer()

    #开启会话
    with tf.Session() as sess:
        #初始化变量
        sess.run(init)

        #1_创建事件文件
        file_writer = tf.summary.FileWriter('./demo/day_01/linear', graph = sess.graph)

        #查看初始化模型参数之后的值
        print('训练前模型参数为：权重%f，偏置%f，损失为%f' % (weights.eval(), bias.eval(), error.eval()))

        #开始训练
        for i in range(1000):
            sess.run(optimizer)
            #print('第%d训练后模型参数为：权重%f，偏置%f，损失为%f' % (i + 1, weights.eval(), bias.eval(), error.eval()))

            #运行合并变量操作
            summary = sess.run(merged)
            #将每次迭代后的变量写入事件文件
            file_writer.add_summary(summary, i )
            #保存模型
            if i % 10 == 0:
                saver.save(sess, './demo/model/my_linear.ckpt')

        print('训练后模型参数为：权重%f，偏置%f，损失为%f' % (weights.eval(), bias.eval(), error.eval()))

        #加载模型
        """ if os.path.exists('./demo/day_01/model/checkpoint'):
            saver.restore(sess, './demo/day_01/model/my_linear.ckpt')
        
        print('训练后模型参数为：权重%f，偏置%f，损失为%f' % (weights.eval(), bias.eval(), error.eval())) """

    return None

if __name__ == '__main__':
    #tensorflow_demo()
    #graph_demo()
    #session_demo()
    #tensor_demo()
    #variable_demo()
    liner_regression()