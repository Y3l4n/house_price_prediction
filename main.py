
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error,r2_score
import statsmodels.api as sm  

housing = pd.DataFrame(pd.read_excel("housing price.py/dataset.xlsx"))
housing.pop('Id')
housing=housing[(~housing["SalePrice"].isnull())]


print("Chào mừng bạn đến với phần mềm dự đoán giá nhà")
print("Mời nhập MSSubClass(int): ",end='')

def tryy():
    global MSSubClass
    try:
        MSSubClass=int(input())
        print("Hệ thống đã ghi nhận MSSubClass là:",MSSubClass)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy()
tryy()

print("Mời nhập MSZoning('RL','RM','C (all)','FV','RH'): ",end=' ')
MSZoning= input()
def tryy1():
    global MSZoning
    if MSZoning not in ['RL','RM','C (all)','FV','RH']:
        print("Sai rồi, mời nhập lại: ",end=' ')
        MSZoning=input()
        tryy1()
    else: 
        print("Hệ thống đã ghi nhận MSZoning là:",MSZoning)
tryy1()

print("Mời nhập LotArea(int): ",end=' ')
LotArea = None
def tryy2():
    global LotArea
    try:
        LotArea=int(input())
        print("Hệ thống đã ghi nhận LotArea là:",LotArea)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy2()
tryy2()

print("Mời nhập LotConfig('Inside','FR2','Corner','CulDSac','FR3'): ",end=' ')
LotConfig = input()
def tryy3():
    global LotConfig
    if LotConfig not in ['Inside','FR2','Corner','CulDSac','FR3']:
        print("Sai rồi, mời nhập lại: ",end=' ')
        LotConfig=input()
        tryy3()
    else: 
        print("Hệ thống đã ghi nhận LotConfig là:",LotConfig)
tryy3()


print("Mời nhập BldgType('1Fam','2fmCon','TwnhsE','Duplex','Twnhs'): ",end=' ')
BldgType = input()
def tryy4():
    global BldgType
    if BldgType not in ['1Fam','2fmCon','TwnhsE','Duplex','Twnhs']:
        print("Sai rồi, mời nhập lại: ",end=' ')
        BldgType=input()
        tryy4()
    else:
        print("Hệ thống đã ghi nhận BldgType là:",BldgType)
tryy4()


print("Mời nhập OverallCond(int): ",end=' ')
OverallCond = None
def tryy5():
    global OverallCond
    try:
        OverallCond=int(input())
        print("Hệ thống đã ghi nhận OverallCond là:",OverallCond)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy5()
tryy5()


print("Mời nhập YearBuilt(int): ",end=' ')
YearBuilt = None
def tryy6():
    global YearBuilt
    try:
        YearBuilt=int(input())
        print("Hệ thống đã ghi nhận YearBuilt là:",YearBuilt)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy6()
tryy6()


print("Mời nhập YearRemodAdd(int): ",end=' ')
YearRemodAdd = None
def tryy7():
    global YearRemodAdd
    try:
        YearRemodAdd=int(input())
        print("Hệ thống đã ghi nhận YearRemodAdd là:",YearRemodAdd)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy7()
tryy7()


print("Mời nhập Exterior1st('Stucco','AsbShng','WdShing','BrkComm','Stone','ImStucc','CBlock','VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','CemntBd','Plywood'): ",end=' ')
Exterior1st = input()
def tryy8():
    global Exterior1st
    if Exterior1st not in ["Stucco",'AsbShng','WdShing','BrkComm','Stone','ImStucc','CBlock','VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','CemntBd',"Plywood"]:
        print("Sai rồi, mời nhập lại: ",end=' ')
        Exterior1st=input()
        tryy8()
    else:
        print("Hệ thống đã ghi nhận Exterior1st là:",Exterior1st)
tryy8()


print("Mời nhập BsmtFinSF2(int): ",end=' ')
BsmtFinSF2 = None
def tryy9():
    global BsmtFinSF2
    try:
        BsmtFinSF2=int(input())
        print("Hệ thống đã ghi nhận BsmtFinSF2 là:",BsmtFinSF2)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy9()
tryy9()


print("Mời nhập TotalBsmtSF(int): ",end=' ')
TotalBsmtSF = None
def tryy10():
    global TotalBsmtSF
    try:
        TotalBsmtSF=int(input())
        print('Hệ thống đã ghi nhận TotallBsmtSF là:',TotalBsmtSF)
    except:
        print("Sai rồi, mời nhập lại: ",end=' ')
        tryy10()
tryy10()



X_input = [MSSubClass, MSZoning, LotArea, LotConfig, BldgType, OverallCond, YearBuilt, YearRemodAdd, Exterior1st, BsmtFinSF2, TotalBsmtSF]
df = pd.DataFrame([X_input], columns=['MSSubClass', 'MSZoning', 'LotArea', 'LotConfig', 'BldgType', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'Exterior1st', 'BsmtFinSF2', 'TotalBsmtSF'])



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



housing = pd.concat([housing,df], ignore_index=True, axis=0)


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


a,b = housing.shape
df = pd.DataFrame([housing.loc[a-1,:]]) 
df.pop('SalePrice')
housing = housing.drop(a-1)




from sklearn.model_selection import train_test_split
df_train,df_test= train_test_split(housing,train_size=0.7,test_size=0.3,shuffle=True,random_state=100)
y_train=df_train.pop("SalePrice")
x_train=df_train 


x_train=pd.concat([x_train,df],ignore_index=True,axis=0)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numvar=['MSSubClass',"LotArea","OverallCond","YearBuilt","YearRemodAdd","TotalBsmtSF","BsmtFinSF2"]
x_train[numvar]=  scaler.fit_transform(x_train[numvar])

c,d = x_train.shape
df = pd.DataFrame([x_train.loc[c-1,:]])

x_train = x_train.drop(c-1)



from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)

rfe = RFE(estimator = lm , n_features_to_select= 12)         
rfe = rfe.fit(x_train, y_train) 
list(zip(x_train.columns,rfe.support_,rfe.ranking_))

col = x_train.columns[rfe.support_] 
c = x_train.columns[~rfe.support_] 
x_train_rfe = x_train[col] 




import statsmodels.api as sm  
x_train_rfe = sm.add_constant(x_train_rfe) 
y_train = y_train.reset_index(drop=True)

lm = sm.OLS(y_train,x_train_rfe).fit() 
print(lm.summary())



df = df[col]
df.insert(0,'constant',1)
z=lm.predict(df)
print('giá nhà cần dự đoán là:',z[891])

