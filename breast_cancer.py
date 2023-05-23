



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,classification_report,r2_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
ds=pd.read_csv(r"C:\Users\HP\Desktop\INTERNSHIP 5 MARCH\ml project\breast cancer detection 08project ml\breast-cancer.csv")
print(ds)
print(ds.head(4))
print(ds.tail(4))
print(ds.shape)
print(ds.info)
print(ds.describe)


#mapping data 
ds.replace({"diagnosis":{"M":1,"B":0}})
print(ds)
ds["diagnosis"]=ds["diagnosis"].map({"B":0,"M":1})
ds.replace({"diagnosis":{"M":1,"B":0}})
print(ds.head(4))

#find x and y

x=ds.iloc[:,2:32].values
y=ds.iloc[:,1].values
#graph 
import matplotlib.pyplot as plt
import seaborn as sns 
sns.pairplot(ds.reset_index(),palette="husl",hue="diagnosis",height=2)
sns.set_style("darkgrid")
plt.show()

#train_test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
print(x_test)

## all models terain

models=[]

models.append(("LR",LogisticRegression()))
models.append(("DT",DecisionTreeClassifier()))
models.append(("SVM",SVC(gamma="auto")))
models.append(("NB",GaussianNB()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("LDA",LinearDiscriminantAnalysis()))
result=[]
names=[]
res=[]
for name,model in models:
    kfold=StratifiedKFold(n_splits=10,random_state=None)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring="accuracy")

result.append(cv_results)
names.append(name)
res.append(cv_results.mean())
print('%s:%f'%(name,cv_results.mean()))


#for one model train and predict

model=LogisticRegression()
model.fit(x_train,y_train)

pre=model.predict(x_test)

#output
print(classification_report(pre,y_test))
print(accuracy_score(pre,y_test)*100)
print(r2_score(pre,y_test)*100)