import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv('data1.csv',index_col=0)
data = data.drop(['dt','expose_start_time','time_slot1','cat_id','brandsn','goods_clk_sum',
'goods_like_sum','brandsn_addcart_sum','brandsn_order_sum','is_like',
'cat_like_sum','cat_clk_sum','cat_order_sum','brandsn_like_sum','brandsn_clk_sum'],axis=1)
# print('原始数据',data.shape)
postive_user =  data.groupby('user_id').agg({
    'is_order': 'sum'
    })
postive_user = postive_user.reset_index()
postive_user.columns=[
    'user_id',
    'order_sum',
]

postive_user = postive_user[postive_user['order_sum']>0]
positive_user = data[data['user_id'].isin(postive_user['user_id'])]
negative_user = data[~data['user_id'].isin(postive_user['user_id'])]
# print(len(positive_user),len(negative_user))



test = pd.read_excel('E:/测试集a/测试集a/a榜需要预测的uid_5000.xlsx',names=['user_id'])
test = pd.merge(test,data,on=['user_id'])
#
#
train = pd.concat([
   positive_user,
   negative_user.sample(int(0.5 * len(negative_user)))
], axis=0)
#
train1 = train.drop(['is_order','user_id','goods_id'],axis=1)
test1 = test.drop(['is_order','user_id','goods_id'],axis=1)
# print(train1)
#
#
y = np.array(train['is_order'].values)

test1 = np.array(test1)
train1 = np.array(train1)
#
#
m1 = int(0.8*len(train1))
x_t = train1[:m1,:]
y_t = y[:m1]
x_t1 = train1[m1:,:]
y_t1 = y[m1:]
randomForest = RandomForestClassifier(n_jobs=-1,n_estimators=70)
model = randomForest.fit(x_t, y_t)
y_pre = model.predict(test1)

res = {'user_id':test['user_id'].values,'goods_id':test['goods_id'].values,'order':y_pre}
df = pd.DataFrame(res)
df = df[df['order']==1]
df = df.drop(columns=['order'])
df.to_csv('u2i.csv',index=False)