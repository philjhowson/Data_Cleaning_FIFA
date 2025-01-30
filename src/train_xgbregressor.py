import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

"""
load in the train_test_split files created in
the rfr training, set the model params for a
grid search and fit the model. Extract the
best parameters and save them.
"""

X_train = pd.read_csv('data/processed_data/X_train.csv')
y_train = pd.read_csv('data/processed_data/y_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb = XGBRegressor()

xgb_grid = GridSearchCV(xgb, param_grid = params, scoring = 'r2', cv = 4, verbose = 1, n_jobs = -1)

xgb_grid.fit(X_train, y_train)

best_params = xgb_grid.best_params_

with open('models/XGBRegressor_best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

"""
train the model with the best parameters, extract scores
for training and test splits, save the model, and the scores.
"""

xgb = XGBRegressor(**best_params)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_train)
xgb_train_score = r2_score(y_train, y_pred)

y_test_pred = xgb.predict(X_test)

xgb_test_score = r2_score(y_test, y_test_pred)

print(f"Training r2 score: {xgb_train_score}\nTest r2 score: {xgb_test_score}")

with open('models/XBGRegressor.pkl', 'wb') as f:
    pickle.dump(xgb, f)

xgb_scores = {'Training Score' : [xgb_train_score],
              'Test Score' : [xgb_test_score]}

with open('metrics/xgb_scores.pkl', 'wb') as f:
    pickle.dump(xgb_scores, f)

"""
finally, calculate the feature importance and
save that as a dataframe.
"""

feature_importance_xgb = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance_xgb': xgb.feature_importances_
})


feature_importance_xgb.to_csv('metrics/feature_importance_xgb.csv', index = False)
