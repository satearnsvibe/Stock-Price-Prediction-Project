#!/usr/bin/env python
# coding: utf-8

# # Stock Price Predction

# In[46]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV


# In[47]:


datas=pd.read_csv("stocks.csv",index_col="Date")
datas.head()


# # Feature Engineering

# In[76]:


def generate_feature(data):
    data_new=pd.DataFrame()
    data_new['Open']=data['Open']
    data_new['open_1']=data['Open'].shift(1)
    data_new['close_1']=data['Close'].shift(1)
    data_new['high_1']=data['High'].shift(1)
    data_new['low_1']=data['Low'].shift(1)
    data_new['volume_1']=data['Volume'].shift(1)
    
  #AVERAGE  
    
    data_new['avg_price_5']=data['Close'].rolling(5).mean().shift(1)
    data_new['avg_price_30']=data['Close'].rolling(21).mean().shift(1)
    data_new['avg_price_365']=data['Close'].rolling(252).mean().shift(1)
    
    
    data_new['avg_price_ratio_5_30']=data_new['avg_price_5']/data_new['avg_price_30']
    data_new['avg_price_ratio_5_365']=data_new['avg_price_5']/data_new['avg_price_365']
    data_new['avg_price_ratio_30_365']=data_new['avg_price_30']/data_new['avg_price_365']
    
    #VOLUME

    data_new['avg_volume_5']=data['Volume'].rolling(5).mean().shift(1)
    data_new['avg_volume_30']=data['Volume'].rolling(21).mean().shift(1)
    data_new['avg_volume_365']=data['Volume'].rolling(252).mean().shift(1)
    
    
    data_new['avg_volume_ratio_5_30']=data_new['avg_volume_5']/data_new['avg_volume_30']
    data_new['avg_volume_ratio_5_365']=data_new['avg_volume_5']/data_new['avg_volume_365']
    data_new['avg_volume_ratio_30_365']=data_new['avg_volume_30']/data_new['avg_volume_365']
    
    #Standard Derivative Price
    
    data_new['std_price_5']=data['Close'].rolling(5).std().shift(1)
    data_new['std_price_30']=data['Close'].rolling(21).std().shift(1)
    data_new['std_price_365']=data['Close'].rolling(365).std().shift(1)
    
    data_new['std_price_ratio_5_30']=data_new['std_price_5']/data_new['std_price_30']
    data_new['std_price_ratio_5_365']=data_new['std_price_5']/data_new['std_price_365']
    data_new['std_price_ratio_30_365']=data_new['std_price_30']/data_new['std_price_365']
    
    
    #Standard DErivative Volume
    
    data_new['std_volume_5']=data['Close'].rolling(5).std().shift(1)
    data_new['std_volume_30']=data['Close'].rolling(21).std().shift(1)
    data_new['std_volume_365']=data['Close'].rolling(365).std().shift(1)
    
    data_new['std_volume_ratio_5_30']=data_new['std_volume_5']/data_new['std_volume_30']
    data_new['std_volume_ratio_5_365']=data_new['std_volume_5']/data_new['std_volume_365']
    data_new['std_volume_ratio_30_365']=data_new['std_volume_30']/data_new['std_volume_365']
    
    data_new['Close']=data['Close']
    data_new=data_new.dropna(axis=0)
    return data_new


# In[49]:


mydata=generate_feature(datas)
mydata.head()


# # Train_Test_data

# In[50]:


start_train="1989-01-01"
end_train="2015-12-31"

start_test="2016-01-01"
end_test="2016-12-31"


# In[51]:


data_train=mydata[start_train:end_train]
x_train=data_train.drop('Close',axis=1).values
y_train=data_train['Close'].values


# In[52]:


print(x_train.shape)
print(y_train.shape)


# In[53]:


data_test=mydata[start_test:end_test]
x_test=data_test.drop('Close',axis=1).values
y_test=data_test['Close'].values


# In[54]:


print(x_test.shape)
print(y_test.shape)


# In[99]:


Scaler=StandardScaler()
x_scaled_train=scaler.fit_transform(x_scaled_train)
y_scaled_train=scaler.transform(y_scaled_train)

param={
    "alpha":[1e-5,3e-5,1e-4],
    "eta0":[0.03,0.01,0.3]
}


# In[100]:


from sklearn.linear_model import SGDRegressor
sgd_reg=SGDRegressor(penalty="l2",max_iter=100)
grid=GridSearchCV(sgd_reg,param,cv=5,scoring='r2')
grid.fit(x_scaled_train,y_train)
print(grid.best_params_)


# # To Check Accuracy

# In[101]:


sgd_reg_pre=grid.best_estimator_
predict_sgd=sgd_reg_pre.predict(x_scaled_train)


# In[102]:


print('MSE:{0:.3f}'.format(mean_squared_error(y_train,predict_sgd)))
print('MAE:{0:.3f}'.format(mean_absolute_error(y_train,predict_sgd)))
print('R2:{0:.3f}'.format(r2_score(y_train,predict_sgd)))


# In[ ]:




