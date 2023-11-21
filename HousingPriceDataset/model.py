import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

df  = pd.read_csv(r'housing.csv')

# Handle missing values for total_bedrooms. As the column does not follow normal distribution fill it with median
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
df = df.drop('total_rooms', axis=1)

# Handling outliers in numerical columns
df['housing_median_age'] = np.clip(df['housing_median_age'], 18.0, 38.0)
df['total_bedrooms'] = np.clip(df['total_bedrooms'], 418.0, 643.25)
df['population'] = np.clip(df['population'], 787.0, 1750)
df['median_income'] = np.clip(df['median_income'], 2.5, 5.0)

# Grouping 'ocean_proximity' categories
df['ocean_proximity'] = df['ocean_proximity'].replace(['NEAR OCEAN', 'NEAR BAY', 'ISLAND'], 'NEAR WATER')
# Performing one hot encoding on nominal variables
df = pd.get_dummies(df, columns=['ocean_proximity'], prefix ='', prefix_sep ='').astype('float64')

# MODELLING
y = df['median_house_value']
x = df.drop('median_house_value',axis=1)

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# Scoring method
def scoring(model, X_test ,y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
print('Linear Regression Results', scoring(model, X_test, y_test))
# Here we are getting rmse to be around 78,000 so an error of 78,000 is in the housing price

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust n_estimators as needed
rf_regressor.fit(X_train, y_train)
print('Random Forest Regression results ', scoring(rf_regressor, X_test, y_test))
# Here we are getting rmse to be around 51,000 so an error of 51,000 is in the housing price

# Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, random_state=42)
gb_regressor.fit(X_train, y_train)
print('Gradient Boost Regressor results ', scoring(gb_regressor, X_test, y_test))
# Here we are getting rmse to be around 58,000 so an error of 58,000 is in the housing price.

randomForestPickle = pickle.dump(rf_regressor, open('regModel.pkl', 'wb'))
pickled_model = pickle.load(open('regModel.pkl', 'rb'))

