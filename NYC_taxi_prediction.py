
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[3]:


data = pd.read_csv("./train.csv", sep=",")


# In[4]:


data.head()


# In[5]:


data.shape


# ## Data Cleaning Process
# * Remove all the rows with NaN's.
# * Remove all the rows with 0 values in each of the columns.
# * Compute the 99.9 percentile and 0.1 percentile values for each of the latitudes and longitudes.
# * Remove all the rows with values lying outside the above limits.
# * Remove all the rows with fare amount <= 0 and also fare amount >= 150USD (~99th percentile)
# * Remove all the rows with passengers >= 7 (~99th percentile)

# In[6]:


missing_data = data[data["dropoff_longitude"].isnull() & data["dropoff_latitude"].isnull()]


# In[7]:


missing_data.shape


# In[8]:


missing_data_inds = []
missing_data_inds += missing_data.index.tolist()


# In[9]:


zero_pickup_longitude = data[(data["pickup_longitude"]==0)]
zero_pickup_longitude.shape


# In[10]:


missing_data_inds += zero_pickup_longitude.index.tolist()


# In[11]:


zero_pickup_latitude = data[(data["pickup_latitude"]==0)]
zero_pickup_latitude.shape


# In[12]:


missing_data_inds += zero_pickup_latitude.index.tolist()


# In[13]:


zero_dropoff_longitude = data[(data["dropoff_longitude"]==0)]
zero_dropoff_longitude.shape


# In[14]:


missing_data_inds += zero_dropoff_longitude.index.tolist()


# In[15]:


zero_dropoff_latitude = data[(data["dropoff_latitude"]==0)]
zero_dropoff_latitude.shape


# In[16]:


missing_data_inds += zero_dropoff_latitude.index.tolist()


# In[17]:


zero_fare_amount = data[(data["fare_amount"]==0)]
zero_fare_amount.shape


# In[18]:


missing_data_inds += zero_fare_amount.index.tolist()


# In[19]:


zero_passenger_count = data[(data["passenger_count"]==0)]
zero_passenger_count.shape


# In[20]:


missing_data_inds += zero_passenger_count.index.tolist()


# In[21]:


len(missing_data_inds)


# In[22]:


missing_data_inds = list(set(missing_data_inds))


# In[23]:


len(missing_data_inds)


# In[24]:


data_without_na_zero = data.drop(missing_data_inds)
data_without_na_zero.shape


# In[25]:


data_without_na_zero["fare_amount"].min()


# In[26]:


data_without_na_zero["fare_amount"].max()


# In[27]:


data_without_na_zero = data_without_na_zero[data_without_na_zero["fare_amount"] >= 0]
data_without_na_zero.shape


# In[28]:


data_without_na_zero.describe()


# In[29]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["pickup_longitude"]>=np.percentile(data_without_na_zero["pickup_longitude"],0.1)))]
data_without_na_zero.shape


# In[30]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["pickup_longitude"]<=np.percentile(data_without_na_zero["pickup_longitude"],99.9)))]
data_without_na_zero.shape


# In[31]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["pickup_latitude"]<=np.percentile(data_without_na_zero["pickup_latitude"],99.9)))]
data_without_na_zero.shape


# In[32]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["pickup_latitude"]>=np.percentile(data_without_na_zero["pickup_latitude"],0.1)))]
data_without_na_zero.shape


# In[33]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["dropoff_longitude"]>=np.percentile(data_without_na_zero["dropoff_longitude"],0.1)))]
data_without_na_zero.shape


# In[34]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["dropoff_longitude"]<=np.percentile(data_without_na_zero["dropoff_longitude"],99.9)))]
data_without_na_zero.shape


# In[35]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["dropoff_latitude"]<=np.percentile(data_without_na_zero["dropoff_latitude"],99.9)))]
data_without_na_zero.shape


# In[36]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["dropoff_latitude"]>=np.percentile(data_without_na_zero["dropoff_latitude"],0.1)))]
data_without_na_zero.shape


# In[37]:


data_without_na_zero.describe()


# In[38]:


data_without_na_zero = data_without_na_zero[((data_without_na_zero["passenger_count"]<=7.0))]
data_without_na_zero.shape


# In[39]:


data_without_na_zero["distance"] = np.sqrt(((data_without_na_zero["dropoff_latitude"] - data_without_na_zero["pickup_latitude"]) ** 2) + ((data_without_na_zero["dropoff_longitude"] - data_without_na_zero["pickup_longitude"]) ** 2))


# In[40]:


data_without_na_zero.describe()


# In[41]:


data_without_na_zero = data_without_na_zero[data_without_na_zero["fare_amount"] <= 150]
data_without_na_zero.shape


# In[42]:


corr_distace_fare = data_without_na_zero["distance"].corr(data_without_na_zero["fare_amount"], method='pearson')


# In[43]:


corr_distace_fare


# In[44]:


data_without_na_zero['pickup_datetime'] = data_without_na_zero['pickup_datetime'].str.replace(" UTC", "")
data_without_na_zero.shape


# In[45]:


data_without_na_zero['pickup_datetime'] = pd.to_datetime(data_without_na_zero['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# In[46]:


data_without_na_zero["time_of_day"] = (data_without_na_zero.pickup_datetime.dt.hour*60) + data_without_na_zero.pickup_datetime.dt.minute
data_without_na_zero["time_of_day"].shape


# In[47]:


corr_time_distance = data_without_na_zero["time_of_day"].corr(data_without_na_zero["distance"], method='pearson')
corr_time_distance


# In[48]:


corr_time_fare = data_without_na_zero["time_of_day"].corr(data_without_na_zero["fare_amount"], method='pearson')
corr_time_fare


# In[49]:


plt.figure(figsize=(8,10))
plt.xlabel("Time of the Day in minutes starting 00:00")
plt.ylabel("Distance")
plt.title("Scatter Plot Between Time of the day and Distance")
plt.scatter(data_without_na_zero["time_of_day"][:1000], data_without_na_zero["distance"][:1000])
plt.show()


# In[50]:


plt.figure(figsize=(8,10))
plt.xlabel("Diatance")
plt.ylabel("Fare Amount")
plt.title("Scatter Plot Between Distance and Fare Amount")
plt.scatter(data_without_na_zero["distance"][:1000], data_without_na_zero["fare_amount"][:1000])
plt.show()


# In[51]:


plt.figure(figsize=(8,10))
plt.xlabel("Time of the Day in minutes starting 00:00")
plt.ylabel("Fare Amount")
plt.title("Scatter Plot Between Time of the day and Fare Amount")
plt.scatter(data_without_na_zero["time_of_day"][:1000], data_without_na_zero["fare_amount"][:1000])
plt.show()


# In[52]:


data_without_na_zero["hour"] = data_without_na_zero.pickup_datetime.dt.hour
data_without_na_zero["day"] = data_without_na_zero.pickup_datetime.dt.day
data_without_na_zero["day_of_week"] = data_without_na_zero.pickup_datetime.dt.weekday
data_without_na_zero["month"] = data_without_na_zero.pickup_datetime.dt.month


# In[53]:


data_without_na_zero.head()


# In[54]:


plt.figure(figsize=(10,8))
plt.xlabel("Hour of the day")
plt.ylabel("Fare Amount")
plt.title("Scatter Plot Between Hour of the day and Fare Amount")
plt.scatter(data_without_na_zero["hour"][:1000], data_without_na_zero["fare_amount"][:1000])
plt.show()


# * We observe that the prices of the cabs are high either during the late night and early mornings or during the peak office     hours.

# In[55]:


plt.figure(figsize=(10,8))
plt.xlabel("Day of week")
plt.ylabel("Fare Amount")
plt.title("Scatter Plot Between Day of Week and Fare Amount")
plt.scatter(data_without_na_zero["day_of_week"][:2000], data_without_na_zero["fare_amount"][:2000], alpha=0.1)
plt.show()


# In[56]:


data_without_na_zero["day_of_week"] = data_without_na_zero.pickup_datetime.dt.day_name()


# In[57]:


data_without_na_zero.head()


# In[58]:


plt.figure(figsize=(10,8))
plt.xlabel("Distance")
plt.ylabel("Day of week")
plt.title("Scatter Plot Between Distance and Day of Week")
plt.scatter(data_without_na_zero["distance"][:1000], data_without_na_zero["day_of_week"][:1000], alpha=0.1)
plt.show()


# In[125]:


plt.figure(figsize=(6,6))
plt.xlabel("Pickup Longitude")
plt.ylabel("Pickup Latitude")
plt.title("Pickup Locations")
plt.xlim(-74.06,-73.7)
plt.scatter(data_without_na_zero["pickup_longitude"], data_without_na_zero["pickup_latitude"], alpha=0.5, s=0.03)
plt.show()


# * We observe that most of the pickups are from Manhatten and Queens most of which includes airport pickups

# In[124]:


plt.figure(figsize=(6,6))
plt.xlabel("Dropoff Longitude")
plt.ylabel("Dropoff Latitude")
plt.title("Dropoff Locations")
plt.xlim(-74.06,-73.7)
plt.scatter(data_without_na_zero["dropoff_longitude"], data_without_na_zero["dropoff_latitude"], alpha=0.5, s=0.01, color='green')
plt.show()


# * We observe that the drop off locations are distributed throughout all the Boroughs of NYC

# In[61]:


plt.figure(figsize=(10,8))
plt.xlabel("Fare Amount")
plt.ylabel("Day of week")
plt.title("Scatter Plot Between Fare amount and Day of Week")
plt.scatter(data_without_na_zero["fare_amount"][:2000], data_without_na_zero["day_of_week"][:2000], alpha=0.2)
plt.show()


# In[62]:


test_data = pd.read_csv("./test.csv", sep=",")


# In[63]:


test_data.head()


# In[64]:


data_without_na_zero["day_of_week"] = data_without_na_zero.pickup_datetime.dt.weekday


# In[65]:


data_without_na_zero.head()


# In[66]:


# train_data = data_without_na_zero[:10000000]


# In[67]:


# target = train_data['fare_amount']
# features = train_data.drop(['fare_amount','pickup_datetime','key'], axis=1)


# In[68]:


# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.2)


# In[69]:


# model = LinearRegression()
# model.fit(features, target)


# In[70]:


# pred = model.predict(X_test)


# * Below are the co-efficients for different features in the model

# In[71]:


# model.coef_


# In[72]:


# np.sqrt(metrics.mean_squared_error(y_test, pred))


# In[73]:


# X_train.head()


# In[74]:


test_data.describe()


# In[75]:


test_data["distance"] = np.sqrt(((test_data["dropoff_latitude"] - test_data["pickup_latitude"]) ** 2) + ((test_data["dropoff_longitude"] - test_data["pickup_longitude"]) ** 2))


# In[76]:


test_data.head()


# In[77]:


test_data['pickup_datetime'] = test_data['pickup_datetime'].str.replace(" UTC", "")
test_data['pickup_datetime'] = pd.to_datetime(test_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
test_data["time_of_day"] = (test_data.pickup_datetime.dt.hour*60) + test_data.pickup_datetime.dt.minute
test_data["time_of_day"].shape


# In[78]:


test_data["hour"] = test_data.pickup_datetime.dt.hour
test_data["day"] = test_data.pickup_datetime.dt.day
test_data["day_of_week"] = test_data.pickup_datetime.dt.weekday
test_data["month"] = test_data.pickup_datetime.dt.month


# In[79]:


test_data.shape


# In[80]:


test_data.head()


# In[81]:


# x_test = test_data.drop(['pickup_datetime','key'], axis=1)


# In[82]:


# submission_pred = model.predict(x_test)


# In[83]:


# submission_pred


# In[84]:


# df = pd.DataFrame()


# In[85]:


# df["key"] = test_data["key"]


# In[86]:


# df["fare_amount"] = submission_pred


# In[87]:


# df.head()


# In[88]:


# df.to_csv("vr_linear_regression.csv", sep=",", index=False)


# In[122]:


train_data = data_without_na_zero[:10000000]
target = train_data['fare_amount']
features = train_data.drop(['fare_amount','pickup_datetime','key','time_of_day','day'], axis=1)
model = LinearRegression()
model.fit(features, target)
model.coef_


# In[127]:


features.columns


# In[90]:


x_test = test_data.drop(['pickup_datetime','key','time_of_day','day'], axis=1)
submission_pred = model.predict(x_test)
submission_pred


# In[91]:


df = pd.DataFrame()
df["key"] = test_data["key"]
df["fare_amount"] = submission_pred
df.to_csv("vr_linear_regression.csv", sep=",", index=False)


# In[92]:


# from sklearn.ensemble import RandomForestRegressor
# rf = RandomForestRegressor(n_estimators = 15)
# rf.fit(features, target)


# In[93]:


# rf.feature_importances_


# In[94]:


# rf_5_preds = rf.predict(x_test)


# In[95]:


# df = pd.DataFrame()
# df["key"] = test_data["key"]
# df["fare_amount"] = rf_5_preds
# df.to_csv("vr_Randomforest_regression.csv", sep=",", index=False)


# In[96]:


# from sklearn.svm import SVR
# clf = SVR()
# clf.fit(features, target)


# In[97]:


# from sklearn.neural_network import MLPRegressor
# mlp = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam', learning_rate='constant', learning_rate_init=0.01,  max_iter=1000)
# mlp.fit(features, target)


# In[98]:


# nn_preds = mlp.predict(x_test)


# In[99]:


# df = pd.DataFrame()
# df["key"] = test_data["key"]
# df["fare_amount"] = nn_preds
# df.to_csv("vr_mlp_regressor.csv", sep=",", index=False)


# In[100]:


# target = (target - target.mean())/target.std()


# In[101]:


# features = (features - features.mean())/features.std()


# In[102]:


# rf = RandomForestRegressor(n_estimators = 10)
# rf.fit(features, target)


# In[103]:


# x_test = (x_test - x_test.mean())/x_test.std()


# In[104]:


# x_test.head()


# In[105]:


# preds = rf.predict(x_test)


# In[106]:


# preds


# In[107]:


# preds = (preds*train_data["fare_amount"].std()) + train_data["fare_amount"].mean()


# In[108]:


# preds


# In[109]:


# df = pd.DataFrame()
# df["key"] = test_data["key"]
# df["fare_amount"] = preds
# df.to_csv("vr_Randomforest_regression_v2.csv", sep=",", index=False)


# In[115]:


train_data = data_without_na_zero[:10000000]
target = train_data['fare_amount']
features = train_data.drop(['fare_amount','pickup_datetime','key','time_of_day','day'], axis=1)


# In[116]:


target = (target - target.mean())/target.std()
features = (features - features.mean())/features.std()


# In[117]:


import lightgbm as lgbm
lgbm_train_data = lgbm.Dataset(features, target, silent=True)
params = {
        'boosting_type': 'gbdt', 'objective': 'regression','learning_rate': 0.005,  
        'reg_alpha': 1, 'reg_lambda': 0.001, 'metric': 'rmse'}
model = lgbm.train(params, train_set=lgbm_train_data, num_boost_round=1000)


# In[120]:


x_test = test_data.drop(['pickup_datetime','key','time_of_day','day'], axis=1)
x_test = (x_test - x_test.mean())/x_test.std()
lgbm_preds = model.predict(x_test)
lgbm_preds = (lgbm_preds*train_data["fare_amount"].std()) + train_data["fare_amount"].mean()
print(lgbm_preds)


# In[121]:


df = pd.DataFrame()
df["key"] = test_data["key"]
df["fare_amount"] = lgbm_preds
df.to_csv("vr_lgbm_v2.csv", sep=",", index=False)

