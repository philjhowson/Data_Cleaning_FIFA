import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle
import json

"""
load in the strong correlation columns and the columns to
drop to reduce multicolinarity. Load in the X and y dataset
and then do a train_test_split and save those files.
"""

X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

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

with open('metrics/rfr_scores.json', 'w') as f:
    json.dump(rfr_scores, f, indent = 4)

feature_importance_rfr = pd.Series(rfr.feature_importances_,
    index = X_train.columns,
)

feature_importance = feature_importance_rfr.to_dict()

with open('metrics/feature_importance_rfr.json', 'w') as f:
    json.dump(feature_importance, f, indent = 4)