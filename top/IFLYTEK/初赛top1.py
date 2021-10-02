import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
def rmse(y_true,y_pred):
    return mean_squared_error(y_true=y_true,y_pred=y_pred) ** 0.5
train = pd.read_csv('./初赛_训练集/保定2016年.csv')
test  = pd.read_csv('./初赛_测试集/石家庄20160701-20170701.csv')

###########################################################################
# 分析数据 发现该数据更适合先用回归模型
data = pd.concat([train,test]).reset_index(drop=True)

###########################################################################
l1 =[]
for i in data['PM2.5']:
    if i == 0:
        l1.append(0)
    if i > 0 and i < 35:
        l1.append(50*i/35)
    if i >= 35 and i < 75:
        l1.append(50*(i-35)/40+50)
    if i >= 75 and i < 115:
        l1.append(50*(i-75)/40+100)
    if i >= 115 and i < 150:
        l1.append(50*(i-115)/35+150)
    if i >= 150 and i < 250:
        l1.append(100*(i-150)/100+200)
    if i >= 250 and i < 350:
        l1.append(100*(i-250)/100+300)
    if i >= 350 and i < 500:
        l1.append(100*(i-350)/150+400)
    if i >= 500:
        l1.append(500)
data['AQI_PM2.5'] = l1

l2 =[]
for i in data['PM10']:
    if i == 0:
        l2.append(0)
    if i > 0 and i < 50:
        l2.append(50*i/50)
    if i >= 50 and i < 150:
        l2.append(50*(i-50)/100+50)
    if i >= 150 and i < 250:
        l2.append(50*(i-150)/100+100)
    if i >= 250 and i < 350:
        l2.append(50*(i-250)/100+150)
    if i >= 350 and i < 420:
        l2.append(100*(i-350)/70+200)
    if i >= 420 and i < 500:
        l2.append(100*(i-420)/80+300)
    if i >= 500 and i < 600:
        l2.append(100*(i-500)/100+400)
    if i >= 600:
        l2.append(500)
data['AQI_PM10'] = l2

l3 =[]
for i in data['SO2']:
    if i == 0:
        l3.append(0)
    if i > 0 and i < 50:
        l3.append(50*i/50)
    if i >= 50 and i < 150:
        l3.append(50*(i-50)/100+50)
    if i >= 150 and i < 475:
        l3.append(50*(i-150)/325+100)
data['AQI_SO2'] = l3

l4 =[]
for i in data['CO']:
    if i == 0:
        l4.append(0)
    if i > 0 and i < 2:
        l4.append(50*i/2)
    if i >= 2 and i < 4:
        l4.append(50*(i-2)/2+50)
    if i >= 4 and i < 14:
        l4.append(50*(i-4)/10+100)
data['AQI_CO'] = l4

l5 =[]
for i in data['NO2']:
    if i == 0:
        l5.append(0)
    if i > 0 and i < 40:
        l5.append(50*i/40)
    if i >= 40 and i < 80:
        l5.append(50*(i-40)/40+50)
    if i >= 80 and i < 180:
        l5.append(50*(i-80)/100+100)
    if i >= 180 and i < 280:
        l5.append(50*(i-180)/100+150)
data['AQI_NO2'] = l5

l6 =[]
for i in data['O3_8h']:
    if i == 0:
        l6.append(0)
    if i > 0 and i < 100:
        l6.append(50*i/100)
    if i >= 100 and i < 160:
        l6.append(50*(i-100)/60+50)
    if i >= 160 and i < 215:
        l6.append(50*(i-160)/55+100)
    if i >= 215 and i < 265:
        l6.append(50*(i-215)/50+150)
    if i >= 265 and i < 800:
        l6.append(100*(i-265)/535+200)
data['AQI_O3'] = l6

###########################################################################
col = ['AQI_CO','AQI_PM2.5','AQI_PM10','AQI_NO2','AQI_SO2','AQI_O3']

data['AQI_L'] = data[col].min(axis=1)
data['AQI_H'] = data[col].max(axis=1)


l7 = []
l8 = []
l9 = []
l10 = []
for i in range(len(l1)):
    k = sorted([l1[i],l2[i],l3[i],l4[i],l5[i],l6[i]],reverse=True)
    l7.append(k[1])
    l8.append(k[2])
    l9.append(k[3])
    l10.append(k[4])

data['AQI_H2'] = l7
data['AQI_H3'] = l8
data['AQI_H4'] = l9
data['AQI_H5'] = l10

data['A_L'] = data['AQI_L'].apply(np.ceil)
data['A_H2'] = data['AQI_H2'].apply(np.ceil)
data['A_H3'] = data['AQI_H3'].apply(np.ceil)
data['A_H4'] = data['AQI_H4'].apply(np.ceil)
data['A_H5'] = data['AQI_H5'].apply(np.ceil)




data['sum1'] = data['AQI_H2']+data['AQI_H3']+data['AQI_H4']+data['AQI_H5']+data['AQI_L']
data['sum2'] = data['A_H2']+data['A_H3']+data['A_H4']+data['A_H5']+data['A_L']



train = data[:train.shape[0]]
test  = data[train.shape[0]:]

###########################################################################

###########################################################################

label = 'IPRC'
def get_l(train, test, feature):
    c = []
    oof_train = np.zeros((train.shape[0],))
    oof_test  = np.zeros((test.shape[0],))

    kf = KFold(n_splits=5,random_state=2020,shuffle=True)
    for index,(tr_index,vl_index) in enumerate(kf.split(train)):
        X_train,X_valid = train.iloc[tr_index][feature].values,train.iloc[vl_index][feature].values
        y_train,y_valid = train.iloc[tr_index][label],train.iloc[vl_index][label]

        lf = LinearRegression()
        lf.fit(X_train,y_train)

        oof_train[vl_index] = lf.predict(X_valid)
        oof_test = oof_test + lf.predict(test[feature].values) / kf.n_splits
        c1 = lf.coef_
        c.append(c1)
    r = rmse(train[label],oof_train)
    print(r)
    return  oof_test, r , c

###########################################################################
#优，良，轻度污染和中度污染预测很准确

sub = pd.DataFrame()
x = train[(train.质量等级 == '优')]
l = test[(test.质量等级 == '优')]

col = ['AQI','sum2','AQI_H','sum1']
answers0, score0, c0 = get_l(x,l,col)
s = pd.DataFrame()
s['date'] = l['日期']
s['IPRC'] = answers0
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 0)
print(c0)
print('评分为:', score0)
print("*" * 50)

###########################################################################
x = train[(train.质量等级 == '良')]
l = test[(test.质量等级 == '良')]

answers1, score1, c1 = get_l(x,l,col)
s = pd.DataFrame()
s['date'] = l['日期']
s['IPRC'] = answers1
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 1)
print(c1)
print('评分为:', score1)
print("*" * 50)

###########################################################################
x = train[(train.质量等级 == '轻度污染')]
l = test[(test.质量等级 == '轻度污染')]

answers2, score2, c2 = get_l(x,l,col)
s = pd.DataFrame()
s['date'] = l['日期']
s['IPRC'] = answers2
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 2)
print(c2)
print('评分为:', score2)
print("*" * 50)

###########################################################################
x = train[(train.质量等级 == '中度污染')]
l = test[(test.质量等级 == '中度污染')]

answers3, score3, c3 = get_l(x,l,col)
s = pd.DataFrame()
s['date'] = l['日期']
s['IPRC'] = answers3
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 3)
print(c3)
print('评分为:', score3)
print("*" * 50)

###########################################################################
#重度污染和严重污染因为数据较少预测不准，措意采用公式预测
#下面同时给出线性回归的系数预测，可以对比一下


x = train[(train.质量等级 == '重度污染')]
l = test[(test.质量等级 == '重度污染')]

answers4, score4, c4 = get_l(x,l,col)
s = pd.DataFrame()
l['I'] = 0.001962278*l['AQI']+0.000462278*l['sum2']+0.002517422*l['AQI_H']+0.001017422*l['sum1']
s['date'] = l['日期']
s['IPRC'] = l['I']
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 4)
print(c4)
print('评分为:', score4)
print("*" * 50)

###########################################################################
x = train[(train.质量等级 == '严重污染')]
l = test[(test.质量等级 == '严重污染')]
col5 = ['AQI','sum2','AQI_H','sum1']
answers5, score5, c5 = get_l(x,l,col5)
s = pd.DataFrame()
l['I'] = 0.002054733*l['AQI']+0.000554733*l['sum2']+0.002609878*l['AQI_H']+0.001109878*l['sum1']
s['date'] = l['日期']
s['IPRC'] = l['I']
sub = pd.concat([sub, s], axis=0)
print("*" * 50)
print('quality为:', 5)
print(c5)
print('评分为:', score5)
print("*" * 50)
sub.to_csv('0.csv',index=False)