
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score

housing = pd.DataFrame(pd.read_excel("housing price.py/dataset.xlsx"))
housing.pop('Id')
housing=housing[(~housing["SalePrice"].isnull())]



fig,axs =plt.subplots(2,4,figsize= (10,5))
plt1=sns.boxplot(housing["MSSubClass"],ax=axs[0,0])
plt2=sns.boxplot(y="BsmtFinSF2",ax= axs[0,1],data=housing)
plt3=sns.boxplot(housing['LotArea'],ax=axs[0,2])
plt4=sns.boxplot(housing["OverallCond"],ax=axs[0,3])
plt1=sns.boxplot(housing['YearBuilt'],ax=axs[1,0])
plt2=sns.boxplot(housing['YearRemodAdd'],ax=axs[1,1])
plt3=sns.boxplot(housing["TotalBsmtSF"],ax=axs[1,2])
plt4=sns.boxplot(y=housing["SalePrice"],ax= axs[1,3])
plt.tight_layout()
plt.show()



q1=housing["LotArea"].quantile(0.25)
q2=housing["LotArea"].quantile(0.75)
IQR= q2-q1
housing=housing[(housing["LotArea"]>=q1-1.5*IQR) & (housing["LotArea"]<=q2+ 1.5*IQR)]

q1=housing["TotalBsmtSF"].quantile(0.25)
q2=housing["TotalBsmtSF"].quantile(0.75)
IQR= q2-q1
housing=housing[(housing["TotalBsmtSF"]>=q1-1.5*IQR) & (housing["TotalBsmtSF"]<=q2+ 1.5*IQR)]

q1=housing["SalePrice"].quantile(0.25)
q2=housing["SalePrice"].quantile(0.75)
IQR= q2-q1
housing=housing[(housing["SalePrice"]>=q1-1.5*IQR) & (housing["SalePrice"]<=q2+ 1.5*IQR)]




fig,axs =plt.subplots(2,4,figsize= (10,5))
plt1=sns.boxplot(housing["MSSubClass"],ax=axs[0,0])
plt2=sns.boxplot(y="BsmtFinSF2",ax= axs[0,1],data=housing)
plt3=sns.boxplot(housing['LotArea'],ax=axs[0,2])
plt4=sns.boxplot(housing["OverallCond"],ax=axs[0,3])
plt1=sns.boxplot(housing['YearBuilt'],ax=axs[1,0])
plt2=sns.boxplot(housing['YearRemodAdd'],ax=axs[1,1])
plt3=sns.boxplot(housing["TotalBsmtSF"],ax=axs[1,2])
plt4=sns.boxplot(y=housing["SalePrice"],ax= axs[1,3])
plt.tight_layout()
plt.show()




plt.figure(figsize=(10,5))
sns.boxplot(x="MSZoning",y="SalePrice",data=housing)
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(x="LotConfig",y="SalePrice",data=housing)
plt.show()
plt.figure(figsize=(10,5))
sns.boxplot(x="BldgType",y="SalePrice",data=housing)
plt.show()
plt.figure(figsize=(20,5))
sns.boxplot(x="Exterior1st",y="SalePrice",data=housing)
plt.show()

sns.pairplot(housing)
plt.show()




check=pd.get_dummies(housing["MSZoning"], dtype=int)
housing=pd.concat([housing,check],axis=1)
housing.pop("MSZoning")
housing.pop('RM')

check=pd.get_dummies(housing["LotConfig"], dtype=int)
housing=pd.concat([housing,check],axis=1)
housing.pop("LotConfig")
housing.pop("FR3")

check=pd.get_dummies(housing["BldgType"], dtype=int)
housing=pd.concat([housing,check],axis=1)
housing.pop("BldgType")
housing.pop("Twnhs")

check=pd.get_dummies(housing["Exterior1st"], dtype=int)
housing=pd.concat([housing,check],axis=1)
housing.pop("Exterior1st")
housing.pop("ImStucc")

print(housing.info())


from sklearn.model_selection import train_test_split
df_train,df_test= train_test_split(housing,train_size=0.7,test_size=0.3,shuffle=True,random_state=100)
y_train=df_train.pop("SalePrice")
x_train=df_train 
dataset = x_train.copy()
print(x_train.info())
'''
x_train.pop("MSSubClass")
del x_train["YearBuilt"]
del x_train["1Fam"]
'''


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numvar=['MSSubClass',"LotArea","OverallCond","YearBuilt","YearRemodAdd","TotalBsmtSF","BsmtFinSF2"]
x_train[numvar]=  scaler.fit_transform(x_train[numvar])


plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()


# MODEL BUILDING
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train) 
rfe = RFE(estimator = lm , n_features_to_select= 12)         
rfe = rfe.fit(x_train, y_train) 
list(zip(x_train.columns,rfe.support_,rfe.ranking_))


col = x_train.columns[rfe.support_]
print('các features được chọn là:',col)
c = x_train.columns[~rfe.support_]

x_train_rfe = x_train[col]

# Adding a constant variable 
import statsmodels.api as sm  
x_train_rfe = sm.add_constant(x_train_rfe)
lm = sm.OLS(y_train,x_train_rfe).fit()
print(lm.summary())



from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X = x_train_rfe
vif['Features'] = X.columns 

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2) 
vif = vif.sort_values(by = "VIF", ascending = False) 
print('giá trị của VIF là:',vif) 







y_train_price = lm.predict(x_train_rfe)
res = (y_train_price - y_train)

import matplotlib.pyplot as plt
import seaborn as sns


fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20) 
fig.suptitle('Error Terms', fontsize = 20)                
plt.xlabel('Errors', fontsize = 18)                   
plt.show()

plt.scatter(y_train,res)
plt.show()


num_vars=['MSSubClass',"LotArea","OverallCond","YearBuilt","YearRemodAdd","TotalBsmtSF","BsmtFinSF2"]

y_test = df_test.pop('SalePrice')
x_test = df_test
x_test[num_vars] = scaler.fit_transform(x_test[num_vars]) 
x_test = sm.add_constant(x_test) 


x_test_rfe = x_test[x_train_rfe.columns] 

y_pred = lm.predict(x_test_rfe) 
from sklearn.metrics import r2_score 

fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)            
plt.xlabel('y_test', fontsize=18)                         
plt.ylabel('y_pred', fontsize=16)                       
plt.show()

print('r2 score của model là:',r2_score(y_test,y_pred))



