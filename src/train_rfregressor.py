import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

"""
load in the strong correlation columns and the columns to
drop to reduce multicolinarity. Load in the X and y dataset
and then do a train_test_split and save those files.
"""

with open('data/processed_data/strong_correlations.pkl', 'rb') as f:
    strong = pickle.load(f)

with open('data/processed_data/drop_columns.pkl', 'rb') as f:
    drop_columns = pickle.load(f)

X = pd.read_csv('X_data.csv')
y = pd.read_csv('y_data.csv')

X = X[strong].drop(columns = drop_columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)

X_train.to_csv('data/processed_data/X_train.csv', index = False)
X_test.to_csv('data/processed_data/X_test.csv', index = False)
y_train.to_csv('data/processed_data/y_train.csv', index = False)
y_test.to_csv('data/processed_data/y_test.csv', index = False)

y_train_array = y_train.squeeze().to_numpy()

"""
generate the rfr object and create the parameters
for a grid search. Create the grid search object
and then fit the data. Extract and save the best
parameters.
"""

rfr = RandomForestRegressor(random_state = 42)

params = {'max_depth' : [None, 5, 10, 20], 
          'min_samples_split' : [2, 4, 8],
          'min_samples_leaf' : [1, 2, 5]}

rfr_grid = GridSearchCV(rfr, param_grid = params, cv = 4, scoring = 'r2', n_jobs = -1)
rfr_grid.fit(X_train, y_train_array)
best_params = rfr_grid.best_params_

with open('models/RandomForestRegressor_best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

"""
train a rfr with the best parameters, get the scores, save the
scores and the model. Extract feature importance and save those
results in a dataframe.
"""

rfr = RandomForestRegressor(**best_params, random_state = 42)

rfr.fit(X_train, y_train_array)

y_pred = rfr.predict(X_train)
rfr_train_score = r2_score(y_train_array, y_pred)

y_test_pred = rfr.predict(X_test)

rfr_test_score = r2_score(y_test, y_test_pred)

rfr_scores = {'Training Score' : [rfr_train_score],
              'Test Score' : [rfr_test_score]}

print(f"Training r2 score: {rfr_train_score}\nTest r2 score: {rfr_test_score}")

with open('models/RandomForestRegressor.pkl', 'wb') as f:
    pickle.dump(rfr, f)

with open('metrics/rfr_scores.pkl', 'wb') as f:
    pickle.dump(rfr_scores, f)

feature_importance_rfr = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance_rfr': rfr.feature_importances_
})

feature_importance_rfr.to_csv('metrics/feature_importance_rfr.csv', index = False)
