# -*- coding:utf-8 -*-
# @FileName  :demo2.py
# @Time      :2023/6/1 16:28
# @Author    :FMY

import pandas as pd


all = pd.read_csv('allData.csv')
# 把时间列标准化时间格式
all['time_slot1'] = pd.to_datetime(all['expose_start_time'])
# 输出这一天是周中的第几天，Monday=0, Sunday=6
all['dayofweek'] = all['time_slot1'].dt.dayofweek

all['is_order'] = all['is_order'].apply(lambda x: 1 if x > 0 else 0)
goods_feat = all.groupby(['goods_id']).agg({
    'is_clk': ['sum'],
    'is_like': ['sum'],
    'is_addcart': ['sum'],
    'is_order': ['sum']
})
goods_feat = goods_feat.reset_index()
goods_feat.columns = [
    'goods_id',
    'goods_clk_sum',
    'goods_like_sum',
    'goods_addcart_sum',
    'goods_order_sum'
]
cat_feat = all.groupby(['cat_id']).agg({
    'is_clk': ['sum'],
    'is_like': ['sum'],
    'is_addcart': ['sum'],
    'is_order': ['sum']
})
cat_feat = cat_feat.reset_index()
cat_feat.columns = [
    'cat_id',
    'cat_clk_sum',
    'cat_like_sum',
    'cat_addcart_sum',
    'cat_order_sum'
]
brandsn_feat = all.groupby(['brandsn']).agg({
    'is_clk': ['sum'],
    'is_like': ['sum'],
    'is_addcart': ['sum'],
    'is_order': ['sum']
})
brandsn_feat = brandsn_feat.reset_index()
brandsn_feat.columns = [
    'brandsn',
    'brandsn_clk_sum',
    'brandsn_like_sum',
    'brandsn_addcart_sum',
    'brandsn_order_sum'
]

train_data = pd.merge(all,goods_feat,on=['goods_id'])
train_data = pd.merge(train_data,cat_feat,on=['cat_id'])
train_data = pd.merge(train_data,brandsn_feat,on=['brandsn'])

train_data.to_csv('data1.csv')