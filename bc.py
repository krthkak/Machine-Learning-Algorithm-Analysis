import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X = pd.DataFrame(bc.data)
Y = pd.DataFrame(bc.target)
dataset = pd.DataFrame(bc.data , columns = bc.feature_names)
dataset["target"] = pd.Series(bc.target)

from sklearn.preprocessing import StandardScaler
stad_X = StandardScaler()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = .2,random_state = 0)

#Scaled Values
x_train = stad_X.fit_transform(x_train)
x_test = stad_X.transform(x_test)

#K Nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors = 5)
knc.fit(x_train,y_train)

y_pred = knc.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_knn = confusion_matrix(y_test,y_pred)
ac_knn = accuracy_score(y_test,y_pred)
cr_knn = classification_report(y_test,y_pred)

# naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)

cm_nb = confusion_matrix(y_test,y_pred_nb)
ac_nb = accuracy_score(y_test,y_pred_nb)
cr_nb = classification_report(y_test,y_pred_nb)

#Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred_lr = lr.predict(x_test)

cm_lr = confusion_matrix(y_test,y_pred_lr)
ac_lr = accuracy_score(y_test,y_pred_lr)
cr_lr = classification_report(y_test,y_pred_lr)

#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 25)
rfc.fit(x_train,y_train)

y_pred_rfc = rfc.predict(x_test)

cm_rfc = confusion_matrix(y_test,y_pred_rfc)
ac_rfc = accuracy_score(y_test,y_pred_rfc)
cr_rfc = classification_report(y_test,y_pred_rfc)

rfc1 = RandomForestClassifier(n_estimators = 100)
rfc1.fit(x_train,y_train)

y_pred_rfc1 = rfc1.predict(x_test)

cm_rfc1 = confusion_matrix(y_test,y_pred_rfc1)

ac_rfc1 = accuracy_score(y_test,y_pred_rfc1)
cr_rfc1 = classification_report(y_test,y_pred_rfc1)


#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)

y_pred_dtc = dtc.predict(x_test)

cm_dtc = confusion_matrix(y_test,y_pred_dtc)
ac_dtc = accuracy_score(y_test,y_pred_dtc)
cr_dtc = classification_report(y_test,y_pred_dtc)

#Support Vcetor Machines
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)

y_pred_svc = svc.predict(x_test)

cm_svc = confusion_matrix(y_test,y_pred_svc)

ac_svc = accuracy_score(y_test,y_pred_svc)
cr_svc = classification_report(y_test,y_pred_svc)

#pedicting for training dataset 
y_train_pred = rfc.predict(x_train)
cm_rfc_train = confusion_matrix(y_train,y_train_pred)

#dtc - Decision tree classifier
#knn - k Nearest Neighbors 
#lr - Logistics Regression
#rfc - Randon forest Regressor
#svc - Support Vector Classifier
#nb - Naive Bayes
import matplotlib.pyplot as plt
accuracy_scores = [ac_dtc,ac_knn,ac_lr,ac_rfc,ac_rfc1,ac_svc,ac_nb]
a_s = [i*100 for i in accuracy_scores]
names = ["Decision tree classifier","k Nearest Neighbors","Logistics Regression","Randon forest Regressor","Randon forest Regressor 100","Support Vector Classifier","Naive Bayes"]
as_range = list(range(1,len(names)+1))
plt.bar(as_range,a_s,color = "green")
plt.xlabel("Classifiers")
plt.ylabel("Accuracy scores")

plt.xticks(as_range,names)
plt.show()

plt.barh(as_range,a_s,color = "green")
plt.ylabel("Classifiers")
plt.xlabel("Accuracy scores")

plt.yticks(as_range,names)
plt.show()