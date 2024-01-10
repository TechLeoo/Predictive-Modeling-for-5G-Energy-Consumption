# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from buildml.automate import SupervisedLearning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import joblib

# Deal with Warnings
warnings.filterwarnings("ignore")

# Get Dataset
data = pd.read_csv("5G_energy_consumption_dataset.csv")
dataset = pd.read_csv("5G_energy_consumption_dataset.csv")

# Exploratory Data Analysis
data.info()
data_head = data.head()
data_tail = data.tail()
data_descriptive_statistic = data.describe()
data_more_desc_statistic = data.describe(include = "all")
data_distinct_count = data.nunique()
data_correlation_matrix = data.corr() 
data_null_count = data.isnull().sum()
data_total_null_count = data.isnull().sum().sum()
data_hist = data.hist(figsize = (15, 10), bins = 10)

# Data Cleaning and Transformation
# data = data.drop(["Time", "BS"], axis = 1)
data["Time"] = pd.to_datetime(data["Time"])

        # Extracting Date Features for Time Series Analysis 
data["Year"] = data["Time"].dt.year
data["Month"] = data["Time"].dt.month
data["Day"] = data["Time"].dt.day
data["Hour"] = data["Time"].dt.hour

        # Drop Time Column
data = data.drop("Time", axis = 1)

        # Drop Duplicate Columns
# data = data.drop_duplicates()

        # Transform BS Column
data = pd.get_dummies(data, drop_first = True, dtype = np.int8)

# Further Data Preparation and Segregation
x = data.drop("Energy", axis = 1)
y = data.Energy

        # Feature Selection
feature_tool = SelectKBest(score_func = f_regression, k = 10)
new_x = feature_tool.fit_transform(x, y)

new_x = pd.DataFrame(new_x, columns = feature_tool.get_feature_names_out())
feature_importance = pd.DataFrame({"Columns": feature_tool.feature_names_in_, "Feature Score": feature_tool.scores_})

x_train, x_test, y_train, y_test = train_test_split(new_x, y, test_size = 0.2, random_state = 0)

# Model Training
regressor = XGBRegressor()
model = regressor.fit(x_train, y_train)

# Model Prediction
y_pred = model.predict(x_train)
y_pred1 = model.predict(x_test)

# Model Evaluation
training_r2 = r2_score(y_train, y_pred)
training_rmse = np.sqrt(mean_squared_error(y_train, y_pred))

test_r2 = r2_score(y_test, y_pred1)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred1))

        # Validation 
# cv_mean = cross_val_score(estimator = regressor, X = x, y = y, cv = 20)

