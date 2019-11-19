#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
#波士顿房价数据

df=pd.read_csv(
    './housing.data',
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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#pool 长宽缩小一倍
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
 
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 16]) #原始数据的维度：16
ys = tf.placeholder(tf.float32, [None, 1])#输出数据为维度：1
 
keep_prob = tf.placeholder(tf.float32)#dropout的比例
 
x_image = tf.reshape(xs, [-1, 4, 4, 1])
## conv1 layer ##第一卷积层
W_conv1 = weight_variable([2,2, 1,32]) 
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
 
## conv2 layer ##第二卷积层
W_conv2 = weight_variable([2,2, 32, 64]) 
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2) 

## fc1 layer ##  full connection 全连接层
W_fc1 = weight_variable([4*4*64, 512])
b_fc1 = bias_variable([512])
 
h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## fc2 layer ## full connection
W_fc2 = weight_variable([512, 1])
b_fc2 = bias_variable([1])#偏置
#最后的计算结果
prediction =  tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 0.01学习效率,minimize(loss)减小loss误差
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
 
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#训练500次
for i in range(200):
    sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.7})
    print(i,'误差=',sess.run(cross_entropy, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1.0}))  # 输出loss值
 
# 可视化
prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 3)) 
axes = fig.add_subplot(1, 1, 1)
line1,=axes.plot(range(len(prediction_value)), prediction_value, 'b--',label='cnn',linewidth=2)
line3,=axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g',label='real')
 
axes.grid()
fig.tight_layout()
plt.legend(handles=[line1,  line3])
plt.title('cnn')
plt.show()
