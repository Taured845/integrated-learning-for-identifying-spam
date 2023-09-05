# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:59:03 2022

@author: 86131
"""

import numpy as np
from sklearn import tree
import math

spambase=np.loadtxt('spambase.data',delimiter=',')
e_A=spambase[:,0:57]  #4601个样本各属性值
e_label=spambase.T[57]  #4601个样本标签（1垃圾，0非垃圾）
n=len(e_A)#样本个数


T=100#决策树个数(？)
num_test=np.zeros((n,1))#每个样本被测试的次数
pre_test=np.zeros((n,1))#总分类器对每个样本预测结果
err_rate_t=[]#各训练器错误率
d_=0#每个决策树max_depth均值
for t in range(T):
    #选样本
    tmp=np.arange(0,n)
    D_t=np.random.choice(tmp,n,True)#第t个决策树选择的样本序号
    #选特征
    m=int(math.log(57))#每个决策树选取特征个数
    tmp=np.arange(0,57)
    AT_t=np.random.choice(tmp,m,False)
    #第t个树的训练集
    X_t=((e_A[D_t].T)[AT_t]).T#第t个树的训练样本（属性）
    y_t=e_label[D_t]#第t个树的训练样本标签
    
    #第t个树测试集
    D_test_t=[]#第t个树未选择的样本编号
    for i in range(n):
        if i not in D_t:
            D_test_t.append(i)
    X_test_t=((e_A[D_test_t].T)[AT_t]).T#第t个决策树未选择样本（属性）
    Y_test_t=e_label[D_test_t]#第t个决策树未选择样本标签
    
    
    #训练决策树
    err_t_0=1
    err_t_1=0
    d=0
    while(d<5):
        Tr_t_0=tree.DecisionTreeClassifier(max_depth=d+1,min_samples_leaf=int(0.1*n))#(剪枝？)
        Tr_t_0=Tr_t_0.fit(X_t,y_t)
        
        S_test_t=Tr_t_0.predict(X_test_t)
        err_t_1=np.sum(abs(S_test_t-Y_test_t))/len(X_test_t)
        
        if(err_t_1<err_t_0):
            Tr_t=Tr_t_0
            d=d+1
            err_t_0=err_t_1
        else:
            break
    err_rate_t.append(err_t_0)
    d_=d_+d
    
    
    #测试
    S_test_t=Tr_t_0.predict(X_test_t)
    for i in range(len(D_test_t)):
        pre_test[D_test_t[i]]=pre_test[D_test_t[i]]+S_test_t[i]
        num_test[D_test_t[i]]=num_test[D_test_t[i]]+1
pre_test=pre_test/num_test
pre_test=(np.where(pre_test<0.5,0,1)).reshape(-1)
d_=d_/T

err_rate=np.sum(abs(pre_test-e_label))/n#总分类器错误率
err_rate_t_avg=sum(err_rate_t)/T#各分类器平均错误率
            















   
    