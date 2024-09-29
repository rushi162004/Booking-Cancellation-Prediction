#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None


# In[152]:


dataset = pd.read_csv("C:/Users/Rushi/Downloads/YourCabs (1).csv")


# In[153]:


dataset.head()


# In[154]:


dataset.drop(["id","user_id"],axis=1,inplace=True)


# In[155]:


dataset


# In[156]:


#adding a boolean column that says if the car model is 12 or not.
dataset['vehicleid_12ornot']=pd.get_dummies(dataset.vehicle_model_id.astype('object'))[12].astype('int')
dataset.head()


# # Spliting the data based on travel type IDs

# In[157]:


long_dist_df = dataset.loc[dataset["travel_type_id"] == 1]


# In[158]:


long_dist_df.head()


# In[159]:


point_to_point_df = dataset.loc[dataset["travel_type_id"] == 2]


# In[160]:


point_to_point_df.head()


# In[446]:


hourly_rental_df = dataset.loc[dataset["travel_type_id"] == 3]


# In[447]:


hourly_rental_df.head()


# # Long distance Machine Learning Model

# In[163]:


long_dist_df.head()


# ## Dropping the irrelevent columns

# In[164]:


long_dist_df.columns


# In[165]:


long_dist_df.drop(['vehicle_model_id','package_id', 'travel_type_id','to_area_id','from_lat',
       'from_long', 'to_lat', 'to_long'], axis = 1,inplace=True)


# In[166]:


long_dist_df.info()


# In[167]:


cancellation_percentage = long_dist_df.groupby('from_area_id')["Car_Cancellation"].mean()*100


# In[ ]:





# In[ ]:





# In[168]:


long_dist_df.from_area_id = long_dist_df.from_area_id.astype('object')
long_dist_df.from_city_id = long_dist_df.from_city_id.astype('object')
long_dist_df.to_city_id = long_dist_df.to_city_id.astype('object')

long_dist_df.from_date = pd.to_datetime(long_dist_df.from_date)
long_dist_df.booking_created = pd.to_datetime(long_dist_df.booking_created)
long_dist_df.info()


# In[169]:


long_dist_df.isna().sum() / long_dist_df.value_counts().sum()*100


# ### Imputing the missing values with most frequent one

# In[ ]:


long_dist_df.describe(include=['object'])


# In[171]:


long_dist_df.from_area_id = long_dist_df.from_area_id.fillna("393")
long_dist_df.from_city_id = long_dist_df.from_city_id.fillna("15")
long_dist_df.to_city_id = long_dist_df.to_city_id.fillna("32")


# In[172]:


long_dist_df.isnull().sum()


# In[173]:


long_dist_df.head()


# In[174]:


long_dist_df


# In[175]:


# calculating area wise cancellation percentage
cancellation_percentage_area = long_dist_df[["from_area_id","Car_Cancellation"]].groupby("from_area_id").mean()["Car_Cancellation"]


# In[176]:


long_dist_df['cancellation_percentage_area'] = long_dist_df.from_area_id.map(cancellation_percentage_area) *100


# In[177]:


long_dist_df


# In[178]:


# dividing the dataset into four categories that is Zero,Low,Medium and high based on the area wise cancellation percentage
long_dist_df["area_category"] = np.where(long_dist_df.cancellation_percentage_area == 0.0,"Zero",
                                        np.where(long_dist_df.cancellation_percentage_area <= 34,"Low",
                                                np.where(long_dist_df.cancellation_percentage_area >= 66,"High","Medium")))
long_dist_df.head()


# In[179]:


long_dist_df["route_city"] = long_dist_df['from_city_id'].astype('str') + ' >> ' + long_dist_df['to_city_id'].astype('str')


# In[180]:


long_dist_df.head()


# In[181]:


route_cancellation_percentage = long_dist_df[['route_city','Car_Cancellation']].groupby('route_city').mean()['Car_Cancellation']


# In[182]:


long_dist_df['route_cancellation_percentage'] =  long_dist_df.route_city.map(route_cancellation_percentage) *100


# In[183]:


long_dist_df.head()


# In[184]:


# dividing the dataset into four categories that is Zero,Low,Medium and high based on the route wise cancellation percentage
long_dist_df["route__category"] = np.where(long_dist_df.route_cancellation_percentage == 0.0,"Zero",
                                        np.where(long_dist_df.route_cancellation_percentage <= 34,"Low",
                                                np.where(long_dist_df.route_cancellation_percentage >= 66,"High","Medium")))
long_dist_df.head()


# In[185]:


long_dist_df['month'] = long_dist_df.from_date.dt.month
long_dist_df['day_of_week'] = long_dist_df.from_date.dt.day_of_week
long_dist_df['weekend'] = (long_dist_df['day_of_week']>=5).astype('int')


# In[186]:


long_dist_df['hour'] = long_dist_df.from_date.dt.hour


# In[187]:


long_dist_df.head()


# In[188]:


long_dist_df['time_of_day'] = pd.cut(long_dist_df['hour'],bins=[-1,6,12,18,24],labels=["Night","Morning","Afternoon","Evening"])


# In[189]:


long_dist_df.head()


# In[190]:


long_dist_df["time_diff"] = (long_dist_df.from_date - long_dist_df.booking_created).dt.total_seconds()/3600


# In[191]:


long_dist_df["time_diff"].describe()


# In[192]:


long_dist_df['booking_type'] = pd.cut(long_dist_df['time_diff'], bins=[-8,2,17,45,1500],labels=['Urgent','Sameday','Normal','Advance'])
long_dist_df.head()


# In[193]:


long_dist_df = long_dist_df.drop(["from_area_id","from_city_id","to_city_id","from_date","booking_created","route_city"],axis=1)


# In[194]:


long_dist_df = long_dist_df.drop(["cancellation_percentage_area","route_cancellation_percentage"],axis=1)


# In[195]:


long_dist_df = long_dist_df.drop(["time_diff"],axis=1)


# In[196]:


long_dist_df


# In[197]:


sns.countplot(x='weekend',data=long_dist_df)
plt.show()


# In[198]:


sns.countplot(x='weekend',hue="Car_Cancellation",data=long_dist_df)


# In[199]:


long_dist_df.drop('weekend',axis=1)


# In[200]:


sns.countplot(x='day_of_week',hue='route__category',data=long_dist_df)


# In[201]:


sns.countplot(x='day_of_week',hue='area_category',data=long_dist_df)


# In[202]:


sns.countplot(x='day_of_week',data=long_dist_df)


# In[203]:


sns.countplot(x='time_of_day',hue="area_category",data=long_dist_df)


# In[204]:


sns.countplot(x='time_of_day',hue="route__category",data=long_dist_df)


# In[205]:


sns.countplot(x='booking_type',hue="Car_Cancellation",data=long_dist_df)


# In[206]:


sns.countplot(x='day_of_week',hue="Car_Cancellation",data=long_dist_df)


# In[207]:


sns.countplot(x='day_of_week',hue="booking_type",data=long_dist_df)


# In[208]:


long_dist_df.area_category = long_dist_df.area_category.map({"Zero":1,"Low":2,"Medium":3,"High":4})
long_dist_df.route__category = long_dist_df.route__category.map({"Zero":1,"Low":2,"Medium":3,"High":4})
long_dist_df.booking_type = long_dist_df.booking_type.map({"Urgent":1,"Sameday":2,"Normal":3,"Advance":4})
long_dist_df.time_of_day = long_dist_df.time_of_day.map({"Morning":1,"Afternoon":2,"Evening":3,"Night":4})
long_dist_df.head()

long_dist_df.head()
# In[211]:


X = long_dist_df.drop("Car_Cancellation",axis=1)
Y = long_dist_df["Car_Cancellation"]


# In[212]:


from sklearn.model_selection import train_test_split


# In[213]:


xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)


# ## Decision Tree

# In[143]:


from sklearn.tree import DecisionTreeClassifier


# In[144]:


long_dist_model_dt = DecisionTreeClassifier()


# In[214]:


long_dist_model_dt.fit(xtrain,ytrain)


# In[215]:


ytrianpred = long_dist_model_dt.predict(xtrain)


# In[216]:


ytestnpred = long_dist_model_dt.predict(xtest)


# In[218]:


from sklearn import metrics


# In[220]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[221]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Naive_Bayes

# In[223]:


from sklearn.naive_bayes import GaussianNB


# In[224]:


long_dist_nb = GaussianNB()


# In[225]:


long_dist_nb.fit(xtrain,ytrain)


# In[226]:


ytrianpred = long_dist_nb.predict(xtrain)
ytestnpred = long_dist_nb.predict(xtest)


# In[227]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[228]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Random Forest

# In[229]:


from sklearn.ensemble import RandomForestClassifier


# In[230]:


long_dist_rf = RandomForestClassifier()


# In[231]:


long_dist_rf.fit(xtrain,ytrain)


# In[232]:


ytrianpred = long_dist_rf.predict(xtrain)
ytestnpred = long_dist_rf.predict(xtest)


# In[233]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[234]:


print(metrics.classification_report(ytest,ytestnpred))


# # Point to Point Distance

# In[236]:


point_to_point_df.head()


# ## Dropping the irrelevent columns

# In[250]:


point_to_point_df.columns


# In[244]:


point_to_point_df.drop(['vehicle_model_id','package_id','from_city_id','to_city_id'], axis = 1,inplace=True)


# In[251]:


point_to_point_df.info()


# In[252]:


cancellation_percentage = point_to_point_df.groupby('from_area_id')["Car_Cancellation"].mean()*100


# In[253]:


point_to_point_df.from_area_id = point_to_point_df.from_area_id.astype('object')

point_to_point_df.from_date = pd.to_datetime(point_to_point_df.from_date)
point_to_point_df.booking_created = pd.to_datetime(point_to_point_df.booking_created)
point_to_point_df.info()


# In[254]:


point_to_point_df.isna().sum()


# 

# In[257]:


point_to_point_df.isnull().sum()


# In[258]:


point_to_point_df.head()


# In[259]:


point_to_point_df.from_area_id = point_to_point_df.from_area_id.astype('int')
point_to_point_df.to_area_id = point_to_point_df.to_area_id.astype('int')
point_to_point_df.head()


# In[263]:


point_to_point_df["route_area"] = point_to_point_df['from_area_id'].astype('str') + ' >> ' + point_to_point_df['to_area_id'].astype('str')


# In[264]:


point_to_point_df.head()


# In[265]:


route_cancellation_percentage = point_to_point_df[['route_area','Car_Cancellation']].groupby('route_area').mean()['Car_Cancellation']


# In[267]:


point_to_point_df['route_cancellation_percentage'] =  point_to_point_df.route_area.map(route_cancellation_percentage) *100


# In[268]:


point_to_point_df.head()


# In[269]:


# dividing the dataset into four categories that is Zero,Low,Medium and high based on the route wise cancellation percentage
point_to_point_df["route__category"] = np.where(point_to_point_df.route_cancellation_percentage == 0.0,"Zero",
                                        np.where(point_to_point_df.route_cancellation_percentage <= 34,"Low",
                                                np.where(point_to_point_df.route_cancellation_percentage >= 66,"High","Medium")))
point_to_point_df.head()


# In[270]:


point_to_point_df['month'] = point_to_point_df.from_date.dt.month
point_to_point_df['day_of_week'] = point_to_point_df.from_date.dt.day_of_week
point_to_point_df['weekend'] = (point_to_point_df['day_of_week']>=5).astype('int')


# In[271]:


point_to_point_df['hour'] = point_to_point_df.from_date.dt.hour


# In[272]:


point_to_point_df.head()


# In[273]:


point_to_point_df['time_of_day'] = pd.cut(point_to_point_df['hour'],bins=[-1,6,12,18,24],labels=["Night","Morning","Afternoon","Evening"])


# In[274]:


point_to_point_df.head()


# In[275]:


point_to_point_df["time_diff"] = (point_to_point_df.from_date - point_to_point_df.booking_created).dt.total_seconds()/3600


# In[276]:


point_to_point_df["time_diff"].describe()


# In[277]:


point_to_point_df['booking_type'] = pd.cut(point_to_point_df['time_diff'], bins=[-23,3.18,9,20,1920],labels=['Urgent','Sameday','Normal','Advance'])
point_to_point_df.head()


# In[278]:


point_to_point_df = point_to_point_df.drop(["from_area_id","to_area_id","from_date","booking_created","route_area","route_cancellation_percentage","time_diff"],axis=1)


# In[280]:


get_ipython().system('pip install geopy')


# In[283]:


from geopy.distance import geodesic
def calc_distance(row):
    from_cord = (row["from_lat"],row["from_long"])
    to_cord = (row["to_lat"],row["to_long"])
    return geodesic(from_cord,to_cord).kilometers


# In[285]:


point_to_point_df["distance"] = point_to_point_df.apply(calc_distance,axis=1)


# In[292]:


point_to_point_df.drop(["from_lat","from_long","to_lat","to_long"],axis=1,inplace=True)


# In[293]:


point_to_point_df.head()


# In[294]:


sns.countplot(x='weekend',data=point_to_point_df)
plt.show()


# In[295]:


sns.countplot(x='weekend',hue="Car_Cancellation",data=point_to_point_df)


# In[296]:


sns.countplot(x='day_of_week',hue='route__category',data=point_to_point_df)


# In[298]:


sns.countplot(x='day_of_week',data=point_to_point_df)


# In[299]:


sns.countplot(x='time_of_day',hue="route__category",data=point_to_point_df)


# In[300]:


sns.countplot(x='booking_type',hue="Car_Cancellation",data=point_to_point_df)


# In[301]:


sns.countplot(x='day_of_week',hue="Car_Cancellation",data=point_to_point_df)


# In[302]:


sns.countplot(x='day_of_week',hue="booking_type",data=point_to_point_df)


# In[304]:


point_to_point_df.route__category = point_to_point_df.route__category.map({"Zero":1,"Low":2,"Medium":3,"High":4})
point_to_point_df.booking_type = point_to_point_df.booking_type.map({"Urgent":1,"Sameday":2,"Normal":3,"Advance":4})
point_to_point_df.time_of_day = point_to_point_df.time_of_day.map({"Morning":1,"Afternoon":2,"Evening":3,"Night":4})
point_to_point_df.head()


# In[305]:


X = point_to_point_df.drop("Car_Cancellation",axis=1)
Y = point_to_point_df["Car_Cancellation"]


# In[306]:


from sklearn.model_selection import train_test_split


# In[307]:


xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)


# ## Decision Tree

# In[143]:


from sklearn.tree import DecisionTreeClassifier


# In[309]:


point_to_point_model_dt = DecisionTreeClassifier()


# In[310]:


point_to_point_model_dt.fit(xtrain,ytrain)


# In[311]:


ytrianpred = point_to_point_model_dt.predict(xtrain)


# In[312]:


ytestnpred = point_to_point_model_dt.predict(xtest)


# In[313]:


from sklearn import metrics


# In[314]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[315]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Naive Bayes

# In[316]:


from sklearn.naive_bayes import GaussianNB


# In[317]:


point_to_point_nb = GaussianNB()


# In[318]:


point_to_point_nb.fit(xtrain,ytrain)


# In[319]:


ytrianpred = point_to_point_nb.predict(xtrain)
ytestnpred = point_to_point_nb.predict(xtest)


# In[320]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[321]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Random Forest

# In[229]:


from sklearn.ensemble import RandomForestClassifier


# In[322]:


point_to_point_rf = RandomForestClassifier()


# In[323]:


point_to_point_rf.fit(xtrain,ytrain)


# In[324]:


ytrianpred = point_to_point_rf.predict(xtrain)
ytestnpred = point_to_point_rf.predict(xtest)


# In[325]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[326]:


print(metrics.classification_report(ytest,ytestnpred))


# # Hourly Rental Database

# In[448]:


hourly_rental_df.head()


# In[449]:


hourly_rental_df = hourly_rental_df.drop(["vehicle_model_id",'travel_type_id','to_area_id','from_city_id','to_city_id','from_lat','from_long','to_lat','to_long'],axis=1)


# In[450]:


hourly_rental_df.head()


# In[451]:


cancellation_percentage_area = hourly_rental_df[["from_area_id","Car_Cancellation"]].groupby("from_area_id").mean()["Car_Cancellation"]


# In[452]:


hourly_rental_df['cancellation_percentage_area'] = hourly_rental_df.from_area_id.map(cancellation_percentage_area) *100


# In[453]:


hourly_rental_df["area_category"] = np.where(hourly_rental_df.cancellation_percentage_area == 0.0,"Zero",
                                        np.where(hourly_rental_df.cancellation_percentage_area <= 34,"Low",
                                                np.where(hourly_rental_df.cancellation_percentage_area >= 66,"High","Medium")))
hourly_rental_df.head()


# In[455]:


hourly_rental_df.from_date = pd.to_datetime(hourly_rental_df.from_date)
hourly_rental_df.booking_created = pd.to_datetime(hourly_rental_df.booking_created)


# In[456]:


hourly_rental_df.isna().sum()


# In[457]:


hourly_rental_df['month'] = hourly_rental_df.from_date.dt.month
hourly_rental_df['day_of_week'] = hourly_rental_df.from_date.dt.day_of_week
hourly_rental_df['weekend'] = (hourly_rental_df['day_of_week']>=5).astype('int')


# In[458]:


hourly_rental_df['hour'] = hourly_rental_df.from_date.dt.hour


# In[459]:


hourly_rental_df.head()


# In[460]:


hourly_rental_df['time_of_day'] = pd.cut(hourly_rental_df['hour'],bins=[-1,6,12,18,24],labels=["Night","Morning","Afternoon","Evening"])


# In[468]:


hourly_rental_df.isna().sum()


# In[469]:


hourly_rental_df["time_diff"] = (hourly_rental_df.from_date - hourly_rental_df.booking_created).dt.total_seconds()/3600


# In[466]:


hourly_rental_df["time_diff"].describe()


# In[467]:


hourly_rental_df['booking_type'] = pd.cut(hourly_rental_df['time_diff'], bins=[-4,2,6,13,700],labels=['Urgent','Sameday','Normal','Advance'])
hourly_rental_df.head()


# In[ ]:


hourly_rental_df.columns


# In[470]:


hourly_rental_df = hourly_rental_df.drop(['from_area_id', 'from_date', 'booking_created','cancellation_percentage_area', 'time_diff'],axis=1)


# In[471]:


hourly_rental_df.head()


# In[472]:


sns.countplot(x='weekend',data=hourly_rental_df)
plt.show()


# In[473]:


sns.countplot(x='day_of_week',hue='area_category',data=hourly_rental_df)


# In[474]:


sns.countplot(x='day_of_week',data=hourly_rental_df)


# In[475]:


sns.countplot(x='time_of_day',hue="area_category",data=hourly_rental_df)


# In[476]:


sns.countplot(x='booking_type',hue="Car_Cancellation",data=hourly_rental_df)


# In[477]:


sns.countplot(x='day_of_week',hue="Car_Cancellation",data=hourly_rental_df)


# In[478]:


sns.countplot(x='day_of_week',hue="booking_type",data=hourly_rental_df)


# In[479]:


hourly_rental_df.area_category = hourly_rental_df.area_category.map({"Zero":1,"Low":2,"Medium":3,"High":4})
hourly_rental_df.booking_type = hourly_rental_df.booking_type.map({"Urgent":1,"Sameday":2,"Normal":3,"Advance":4})
hourly_rental_df.time_of_day = hourly_rental_df.time_of_day.map({"Morning":1,"Afternoon":2,"Evening":3,"Night":4})
hourly_rental_df.head()


# In[480]:


X = hourly_rental_df.drop("Car_Cancellation",axis=1)
Y = hourly_rental_df["Car_Cancellation"]


# In[482]:


from sklearn.model_selection import train_test_split


# In[483]:


xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)


# # Decision Tree

# In[484]:


from sklearn.tree import DecisionTreeClassifier


# In[485]:


hourly_rental_model_dt = DecisionTreeClassifier()


# In[486]:


hourly_rental_model_dt.fit(xtrain,ytrain)


# In[487]:


ytrianpred = hourly_rental_model_dt.predict(xtrain)


# In[488]:


ytestnpred = hourly_rental_model_dt.predict(xtest)


# In[489]:


from sklearn import metrics


# In[490]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[491]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Naive_bayes

# In[492]:


from sklearn.naive_bayes import GaussianNB


# In[493]:


hourly_rental_nb = GaussianNB()


# In[494]:


hourly_rental_nb.fit(xtrain,ytrain)


# In[495]:


ytrianpred =hourly_rental_nb.predict(xtrain)
ytestnpred = hourly_rental_nb.predict(xtest)


# In[496]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[497]:


print(metrics.classification_report(ytest,ytestnpred))


# ## Random Forest

# In[498]:


from sklearn.ensemble import RandomForestClassifier


# In[499]:


hourly_rental_rf = RandomForestClassifier()


# In[500]:


hourly_rental_rf.fit(xtrain,ytrain)


# In[501]:


ytrianpred = hourly_rental_rf.predict(xtrain)
ytestnpred = hourly_rental_rf.predict(xtest)


# In[502]:


print(metrics.classification_report(ytrain,ytrianpred))


# In[503]:


print(metrics.classification_report(ytest,ytestnpred))


# For Long Distance - Disicion Tree
# For Poin to Point Distance - Random Forest
# For Hourly Rental - Random Forest

# In[ ]:




