# -*- coding:utf-8 -*-
# @FileName  :demo3.py
# @Time      :2023/6/2 14:49
# @Author    :FMY
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rc = {'font.sans-serif': 'SimHei',
      'axes.unicode_minus': False}
train = pd.read_csv('data1.csv')
hasBuy = train[train['is_order']>0]
sns.countplot(data=train,x=hasBuy['dayofweek'],palette='Accent')
plt.show()
train = train[['is_order','goods_clk_sum','goods_like_sum','goods_addcart_sum','goods_order_sum','cat_clk_sum','cat_like_sum',
               'cat_addcart_sum','cat_order_sum','brandsn_clk_sum','brandsn_like_sum','brandsn_addcart_sum',
               'brandsn_order_sum','dayofweek','is_clk','is_like','is_addcart']]


sns.countplot(data=train,x=train['dayofweek'],hue='is_order',palette='Accent')
corr = train.corr()
plt.figure()
g1 = sns.heatmap(corr,cmap='cool',annot=True)
g1.set_title('Correlation coefficient matrix for features to features, features to labels')
plt.show()

