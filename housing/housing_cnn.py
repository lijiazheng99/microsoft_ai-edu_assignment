#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
#波士顿房价数据

df=pd.read_csv(
    '/Users/cynthia/Downloads/housing-2.data',
    sep=" +",
    header=None,
    names=['CRIM', 'ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','A']
)
df=df[~df['A'].isin([50])]
#df=df[~df['RM'].isin([8.78])]
X=df.iloc[0:, 0:13]
X_3=df[['RM','LSTAT','PTRATIO']]
X_3=X_3.values
y=df['A']
y=y.values
X=np.column_stack([X,X_3])
 
print('##################################################################')
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(X, y,
                                                                    train_size=0.8, random_state=33)
#数据标准化
ss_x = preprocessing.StandardScaler()
train_x_disorder = ss_x.fit_transform(train_x_disorder)
test_x_disorder = ss_x.transform(test_x_disorder)
 
ss_y = preprocessing.StandardScaler()
train_y_disorder = ss_y.fit_transform(train_y_disorder.reshape(-1, 1))
test_y_disorder=ss_y.transform(test_y_disorder.reshape(-1, 1))

#变厚矩阵
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
#偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#卷积处理 变厚过程
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement就是步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#pool 长宽缩小一倍
def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 16]) #原始数据的维度：16
ys = tf.placeholder(tf.float32, [None, 1])#输出数据为维度：1
 
keep_prob = tf.placeholder(tf.float32)#dropout的比例
 
x_image = tf.reshape(xs, [-1, 4, 4, 1])#原始数据16变成二维图片4*4
## conv1 layer ##第一卷积层
W_conv1 = weight_variable([2,2, 1,32]) # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 2x2x32，长宽不变，高度为32的三维图像
#h_pool1 = max_pool_2x2(h_conv1)     # output size 2x2x32 长宽缩小一倍
 
## conv2 layer ##第二卷积层
W_conv2 = weight_variable([2,2, 32, 64]) # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) #输入第一层的处理结果 输出shape 4*4*64
 
## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([4*4*64, 512])#4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([512])
 
h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*64])#把4*4，高度为64的三维图片拉成一维数组 降维处理
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#把数组中扔掉比例为keep_prob的元素
## fc2 layer ## full connection
W_fc2 = weight_variable([512, 1])#512长的一维数组压缩为长度为1的数组
b_fc2 = bias_variable([1])#偏置
#最后的计算结果
prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#prediction = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# 计算 predition与y 差距 所用方法很简单就是用 suare()平方,sum()求和,mean()平均值
cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 0.01学习效率,minimize(loss)减小loss误差
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
 
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())
#训练500次
for i in range(200):
    sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.7})
    print(i,'误差=',sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1.0}))  # 输出loss值
 
# 可视化
prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})
###画图###########################################################################
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
axes = fig.add_subplot(1, 1, 1)
line1,=axes.plot(range(len(prediction_value)), prediction_value, 'b--',label='cnn',linewidth=2)
#line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
line3,=axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g',label='real')
 
axes.grid()
fig.tight_layout()
#plt.legend(handles=[line1, line2,line3])
plt.legend(handles=[line1,  line3])
plt.title('cnn')
plt.show()
