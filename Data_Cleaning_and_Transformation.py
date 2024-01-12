import pandas as pd
import numpy as np
from Initial_EDA import data

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
data = data.drop_duplicates()

        # Transform BS Column
data = pd.get_dummies(data, drop_first = True, dtype = np.int8)

#         #  Data Binning 
# data["load"] = pd.cut(x = data["load"], bins = 30, labels = False)
# data.ESMODE = pd.cut(x = data.ESMODE, bins = 30, labels = False)  