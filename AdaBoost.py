# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:42:33 2022

@author: 86131
"""

import numpy as np
from sklearn import tree
import math

spambase=np.loadtxt('spambase.data',delimiter=',')
e_A=spambase[:,0:57]  #4601个样本各属性值
e_label=spambase.T[57]  #4601个样本标签（1垃圾，0非垃圾）
e_label=np.where(e_label==0,-1,1)#4601个样本标签（1垃圾，-1非垃圾）
n=len(e_A)#样本个数


#采样
p=0.75#训练集比例
m=int(p*n)#训练集个数
tmp=np.arange(0,n)
test=np.random.choice(tmp,n-m,False)#测试集样本编号
train=[]
for i in range(n):
    if i not in test:
        train.append(i)
train=np.array(train)#训练集编号

X_test=e_A[test]#测试集属性
Y_test=e_label[test]#测试集标签
X_train=e_A[train]#训练集属性
Y_train=e_label[train]#训练集标签


#训练
T=100#决策树个数
w=np.ones_like(train)/m#权重，是向量（与train对应）
err_rate_test_t=[]#各分类器错误率
pre_test=np.zeros((len(test)))#总分类器对测试集中各样本预测结果
d_=0#每个决策树max_depth均值
for t in range(T):
    #对训练集加权
    tmp=np.arange(0,m)
    train_t=np.random.choice(tmp,(m),True,w)#其中元素i是train中第i个元素
    X_train_t=X_train[train_t]#当前训练集特征
    Y_train_t=Y_train[train_t]#当前训练集标签
    w_t=w[train_t]#当前训练集各样本对应权重
    w_t=w_t/np.sum(w_t)
    
    #训练决策树
    err_t_0=1
    err_t_1=0
    d=0
    while(d<5):
        Tr_t_0=tree.DecisionTreeClassifier(max_depth=d+1,min_samples_leaf=int(0.1*m))
        Tr_t_0=Tr_t_0.fit(X_train_t,Y_train_t)
        S_test_d=Tr_t_0.predict(X_test)
        err_t_1=(np.sum(abs(S_test_d-Y_test))/2)/len(test)
        if(err_t_1<err_t_0):
            Tr_t=Tr_t_0
            d=d+1
            err_t_0=err_t_1
        else:
            break
    err_rate_test_t.append(err_t_0)
    d_=d_+d
    
    #计算加权错误率
    S_train_t=Tr_t_0.predict(X_train_t)#当前决策树对当前训练集预测结果
    err_rate_t=0
    for i in range(len(S_train_t)):
        if(S_train_t[i]!=Y_train_t[i]):
            err_rate_t=err_rate_t+w_t[i]
    
    #更新权重
    a_t=math.log((1-err_rate_t)/err_rate_t)/2
    bo=np.zeros_like(train)#样本权重是否更新（第i个元素为train中第i个样本在train_t中被采样次数）
    for i in range(len(train_t)):
        if(bo[train_t[i]]==0):
            w[train_t[i]]=w[train_t[i]]*math.exp(-a_t*S_train_t[i]*Y_train_t[i])
            bo[train_t[i]]=1  
    w=w/np.sum(w)
    
    #测试
    pre_test=pre_test+(Tr_t.predict(X_test))*a_t
pre_test=np.where(pre_test<0,-1,1)
d_=d_/T

err_rate=np.sum(abs(pre_test-Y_test)/2)/len(test)#总分类器错误率
err_rate_test_t_avg=sum(err_rate_test_t)/T#各分类器平均错误率
    
















