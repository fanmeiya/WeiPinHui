# -*- coding:utf-8 -*-
# @FileName  :demo1.py
# @Time      :2023/6/1 15:11
# @Author    :FMY
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder


train_goods = pd.concat([
    pd.read_csv('E:/训练集/训练集/traindata_goodsid/part-00000', header=None, names=['goods_id', 'cat_id', 'brandsn']),
    pd.read_csv('E:/训练集/训练集/traindata_goodsid/part-00001', header=None, names=['goods_id', 'cat_id', 'brandsn']),
    pd.read_csv('E:/训练集/训练集/traindata_goodsid/part-00002', header=None, names=['goods_id', 'cat_id', 'brandsn'])
], axis=0)

train_user = pd.concat([
    pd.read_csv(x, header=None, names=['user_id', 'goods_id', 'is_clk', 'is_like', 'is_addcart', 'is_order', 'expose_start_time', 'dt'], nrows=None)
    for x in glob.glob('E:/训练集/训练集/traindata_user/part*')
], axis=0)

testa_goods = pd.concat([
    pd.read_csv('E:/测试集a/测试集a/predict_goods_id/part-00000', header=None, names=['goods_id']),
    pd.read_csv('E:/测试集a/测试集a/predict_goods_id/part-00001', header=None, names=['goods_id']),
], axis=0)

testa_user = pd.read_excel('E:/测试集a/测试集a/a榜需要预测的uid_5000.xlsx',names=['user_id'])

print(np.mean(testa_user['user_id'].isin(train_user['user_id'])),np.mean(testa_goods['goods_id'].isin(train_goods['goods_id'])))

train_goods = pd.merge(train_goods,testa_goods,on=['goods_id'])
all = pd.merge(train_user,train_goods,on=['goods_id'])

all.to_csv('allData.csv',index=False)