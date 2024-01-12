from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from Data_Cleaning_and_Transformation import data

# Further Data Preparation and Segregation
x = data.drop("Energy", axis = 1)
y = data.Energy

        # Feature Selection
# feature_tool = SelectKBest(score_func = f_regression, k = 10)
# new_x = feature_tool.fit_transform(x, y)

# new_x = pd.DataFrame(new_x, columns = feature_tool.get_feature_names_out())
# feature_importance = pd.DataFrame({"Columns": feature_tool.feature_names_in_, "Feature Score": feature_tool.scores_})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
