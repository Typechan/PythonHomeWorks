import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.pop('SalePrice')

test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# Create X (After completing the exercise, you can return to modify this line!)

numerics = ['int16', 'int32', 'int64', 'float16', 'float32']
features = list(test_data.select_dtypes(include = numerics).columns)
# Select columns corresponding to features, and preview the data
X = home_data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(n_estimators = 100, max_depth = 100, random_state=18, min_samples_split = 3, min_samples_leaf = 1, max_features = 10, max_leaf_nodes = 1000)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(n_estimators = 100, max_depth = 100, random_state=18, min_samples_split = 3, min_samples_leaf = 1, max_features = 10, max_leaf_nodes = 1000)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)
# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data.select_dtypes(include=numerics)

# Specify Model

# Fit Model
rf_model_on_full_data.fit(X, y)
# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)
# Check your answer (To get credit for completing the exercise, you must get a "Correct" result!)
step_1.check()
# step_1.solution()
# Run the code to save predictions in the format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
