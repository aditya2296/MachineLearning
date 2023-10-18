import pandas as pd
import numpy as np
    
d = pd.read_csv(r'D:\a3\termplan_datas.csv')

d['age'] = np.where(d['age']<25.0,25.0,d['age'])
d['age'] = np.where(d['age']>64.0,64.0,d['age'])
d['balance'] = np.where(d['balance']<0.0,0.0,d['balance'])
d['balance'] = np.where(d['balance']>3574.0,3574.0,d['balance'])
d['duration'] = np.where(d['duration']<58.0,58.0,d['duration'])
d['duration'] = np.where(d['duration']>548.0,548.0,d['duration'])
d['campaign'] = np.where(d['campaign']<1.0,1.0,d['campaign'])
d['campaign'] = np.where(d['campaign']>5.0,5.0,d['campaign'])

d = d.drop('default',axis=1)
d = d.drop('loan',axis=1)
d = d.drop('pdays',axis=1)
d = d.drop('previous',axis=1)
d = d.drop('poutcome',axis=1)

a = {'month': {'jan': 1,'feb': 2,'mar': 3,'apr': 4,'may': 5,'jun': 6,'jul': 7,'aug': 8,'sep': 9,'oct': 10,'nov': 11,'dec': 12}}
d.replace(a,inplace=True)
from sklearn.preprocessing import OneHotEncoder
o = OneHotEncoder(handle_unknown='ignore')
o_d = pd.DataFrame(o.fit_transform(d[['job']]).toarray())
o_d.columns = o.get_feature_names(['job'])
d_f = d.join(o_d)
d_f = d_f.drop('job',axis=1)
o_d = pd.DataFrame(o.fit_transform(d_f[['marital']]).toarray())
o_d.columns = o.get_feature_names(['marital'])
d_f2 = d_f.join(o_d)
d_f2 = d_f2.drop('marital',axis=1)
from sklearn.preprocessing import LabelEncoder
e = LabelEncoder()
d_f2['education'] = e.fit_transform(d_f2['education'])
from sklearn.preprocessing import OrdinalEncoder
e = OrdinalEncoder()
d_f2[['housing','contact','y']] = e.fit_transform(d_f2[['housing','contact','y']])
a = {'month': {'jan': 1,'feb': 2,'mar': 3,'apr': 4,'may': 5,'jun': 6,'jul': 7,'aug': 8,'sep': 9,'oct': 10,'nov': 11,'dec': 12}}
d_f2.replace(a,inplace=True)
y = d_f2['y']
X = d_f2.drop('y',axis=1)

def scoring(y_test,y_predict):
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    print('accuracy_score: ',format(accuracy_score(y_test,y_predict),"0.2f"))
    print(classification_report(y_test,y_predict))
    print(confusion_matrix(y_test,y_predict))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
dt2 = DecisionTreeClassifier()
p = {'max_depth':[2,10]}
clf = GridSearchCV(dt2,p,cv=5,scoring='accuracy')
clf.fit(X_train,y_train)
print(clf.best_params_)
scoring(y_test,y_predict)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train,y_train)
y_predict = dt.predict(X_test)
print('Decision tree classifier')
scoring(y_test,y_predict)

from sklearn.naive_bayes import GaussianNB
g = GaussianNB()
g.fit(X_train,y_train)
y_predict = g.predict(X_test)
print('Naive Bayes Classifier')
scoring(y_test,y_predict)


from sklearn.ensemble import RandomForestClassifier
r = RandomForestClassifier()
r.fit(X_train,y_train)
y_predict=r.predict(X_test)
print('Random Forest Classifier')
scoring(y_test,y_predict)
print('.')
