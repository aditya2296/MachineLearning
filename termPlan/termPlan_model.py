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
d['age'] = np.clip(d['age'], 18, 70) # age column has no null values
d['balance'] = np.clip(np.cbrt(d['balance']), -10.0, 25.0) # applied cube root transformation as column was positively skewed and was having negative and zero values
d['duration'] = np.clip(np.cbrt(d['duration']), 1.0, 10.2) # applied cube root transformation as column was positively skewed and was having negative and zero values
d['campaign'] = np.clip(np.cbrt(d['campaign']), 1.0, 2.5) # applied cube root transformation as column was positively skewed
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

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# Handling imbalanced dataset using SMOTE techniques
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

logisticRegression = LogisticRegression()
logisticRegression.fit(X_resampled,y_resampled)
print('LogisticRegression')
scoring(y_test, logisticRegression.predict(X_test))

randomForestClassifier = RandomForestClassifier(oob_score=True, random_state=42)
randomForestClassifier.fit(X_resampled,y_resampled)
print('Random Forest Classifer')
scoring(y_test, randomForestClassifier.predict(X_test))
