import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Sample DataFrame with data
d = pd.read_csv(r'termplan_dataset.csv')

# Handling outliers and capping values
d['age'] = np.clip(d['age'], 18, 70)
d['balance'] = np.clip(d['balance'], 0, 4000).astype(int)
d['duration'] = np.clip(d['duration'], 10, 700)
d['campaign'] = np.clip(d['campaign'], 1, 5)

# Grouping 'job' categories
d['job'] = d['job'].replace(['unknown', 'student', 'entrepreneur', 'housemaid'], ['unemployed', 'unemployed', 'self-employed', 'self-employed'])

# Dropping unnecessary columns
d = d.drop(['default', 'loan', 'pdays', 'previous', 'poutcome'], axis=1)

# Binning 'age' column into two categories
d['age'] = (d['age'] > 40).astype(int)

# Binning 'day' column into weeks
bins = [0, 7, 14, 21, 28, 32]
bin_labels = [1, 2, 3, 4, 5]
d['day'] = pd.cut(d['day'], bins=bins, labels=bin_labels, right=False).astype(int)

# Mapping 'month' column to financial quarters
month_to_quarter = {'jan': 1, 'feb': 1, 'mar': 1,
                    'apr': 2, 'may': 2, 'jun': 2,
                    'jul': 3, 'aug': 3, 'sep': 3,
                    'oct': 4, 'nov': 4, 'dec': 4}
d['month'] = d['month'].map(month_to_quarter).astype(int)

# Binning balance column in 3 categories
bins = [0, 1001, 2001, 4001]
bin_labels = [1, 2, 3]
d['balance'] = pd.cut(d['balance'], bins=bins, labels=bin_labels, right=False).astype(int)

# Binning duration campaign in 2 categories
d['duration'] = (d['duration'] > 319).astype(int)

# Performing one hot encoding on nominal variables
columns_to_cast_as_int = ['job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_technician', 'job_unemployed', 'job_blue-collar', 'marital_divorced', 'marital_single', 'marital_married', 'job_admin.', 'contact_cellular', 'contact_telephone', 'contact_unknown']
df_encoded = pd.get_dummies(d, columns=['job', 'marital', 'contact'], prefix=['job', 'marital', 'contact'])
df_encoded[columns_to_cast_as_int] = df_encoded[columns_to_cast_as_int].astype(int)

# Performing ordinal encoding on ordinal variables
df_encoded['education'] = df_encoded['education'].replace({'unknown': -1,'primary': 1, 'secondary': 2, "tertiary": 3})
df_encoded[['housing', 'y']] = df_encoded[['housing', 'y']].replace({'yes': 1,'no': 0})

# MODELLING
y = df_encoded['y']
X = df_encoded.drop('y',axis=1)
def scoring(y_test,y_predict):
    print('accuracy_score: ',format(accuracy_score(y_test,y_predict),"0.2f"))
    print('classification_report', classification_report(y_test,y_predict))
    print('confusion_matrix', confusion_matrix(y_test,y_predict))

# Handling imbalanced dataset using SMOTE techniques
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train,X_test,y_train,y_test = train_test_split(X_resampled, y_resampled, test_size = 0.25, random_state=42)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_train,y_train)
print('LogisticRegression')
scoring(y_test, logisticRegression.predict(X_test))

randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(X_train, y_train)
print('Random Forest Classifer')
scoring(y_test, randomForestClassifier.predict(X_test))

