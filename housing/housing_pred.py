#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:43:02 2019

@author: cynthia
"""

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



df=pd.read_csv(
    '/Users/cynthia/Downloads/housing-2.data',
    sep=" +",
    header=None,
    names=['CRIM', 'ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','A']
)
df=df[~df['A'].isin([50])]
X=df.iloc[0:, 0:13]
X=X.values
y=df['A']
y=y.values

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.1,random_state=0)
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)

n_hidden_1 = 64 #隐藏层1的神经元个数
n_hidden_2 = 64 #隐藏层2的神经元个数
n_input = 13 #输入层的个数
n_classes = 1 #输出层的个数
training_epochs = 200 #训练次数，总体数据需要循环多少次
batch_size = 5  #每批次要取的数据的量，这里是提取10条数据

nb_lstm_outputs = 30  #神经元个数
nb_time_steps = 28  #时间序列长度
nb_input_vector = 28 #输入序列


model = Sequential()
model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input))
model.add(Dense(n_hidden_2, activation='relu'))
model.add(Dense(n_classes)) 


#自定义评价函数
import keras.backend as K
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])

history = model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs)

pred_test_y = model.predict(test_x)
print(pred_test_y)

from sklearn.metrics import r2_score
pred_acc = r2_score(test_y, pred_test_y)
print('pred_acc',pred_acc)

#绘图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图形大小
plt.figure(figsize=(8, 4), dpi=80)
plt.plot(range(len(test_y)), test_y, ls='-.',lw=2,c='r',label='ture')
plt.plot(range(len(pred_test_y)), pred_test_y, ls='-',lw=2,c='b',label='predict')

# 绘制网格
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.xlabel('number') #设置x轴的标签文本
plt.ylabel('price') #设置y轴的标签文本

# 展示
plt.show()
