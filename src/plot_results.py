import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

"""
load in all the scores and combine them into a single
dictionary. Convert them to a df.
"""

with open('metrics/rfr_scores.json', 'r') as f:
    rfr_scores = json.load(f)

with open('metrics/xgb_scores.json', 'r') as f:
    xgb_scores = json.load(f)

with open('metrics/fnn_v7_scores.json', 'r') as f:
    fnn_scores = json.load(f)

scores = {'Training Score': [rfr_scores['Training Score'][0], xgb_scores['Training Score'][0], fnn_scores['Training Score'][0]],
          'Test Score': [rfr_scores['Test Score'][0], xgb_scores['Test Score'][0], fnn_scores['Test Score'][0]]}

scores = pd.DataFrame(scores, index = ['Random Forest Regressor', 'XGBRegressor', 'FNN'])

"""
create a barplot to present the training and test
scores for each model and save it.
"""

plt.figure(figsize = (15, 5))

x = np.arange(len(scores))
bar_width = 0.35

training_bars = plt.bar(x - bar_width / 2, scores['Training Score'], color = 'skyblue', width = bar_width, label = 'Training Score')
test_bars = plt.bar(x + bar_width / 2, scores['Test Score'], color = 'purple', width = bar_width, label = 'Test Score')

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks(x, scores.index)
plt.ylabel('Scores')
plt.title('Training and Test Scores')

for bar in training_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha = 'center', va = 'bottom', fontsize = 10)

for bar in test_bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.4f}', ha = 'center', va = 'bottom', fontsize = 10)

plt.legend();

plt.savefig('images/training_test_scores.png', dpi = 300, bbox_inches = 'tight')

"""
load in the feature importances and combine them into a single
df.
"""

with open('metrics/feature_importance_rfr.json', 'r') as f:
    rfr_features = json.load(f)

with open('metrics/feature_importance_xgb.json', 'r') as f:
    xgb_features = json.load(f)

with open('metrics/feature_importance_fnn_v7.json', 'r') as f:
    fnn_features = json.load(f)

feature_importance = pd.DataFrame(list(rfr_features.items()), columns=['Feature', 'Importance_rfr'])

feature_importance['Importance_xgb'] = xgb_features.values()
feature_importance['Importance_fnn'] = fnn_features.values()

"""
plot the feature imporantances for each model and save the
result.
"""

plt.figure(figsize = (15, 5))

y = np.arange(len(feature_importance))
bar_height = 0.25

rfr_bars = plt.barh(y - bar_height, feature_importance['Importance_rfr'], color = 'skyblue', height = bar_height, label = 'Random Forest Regressor')
xgb_bars = plt.barh(y, feature_importance['Importance_xgb'], color = 'blue', height = bar_height, label = 'XGBRegressor')
fnn_bars = plt.barh(y + bar_height, feature_importance['Importance_fnn'], color = 'purple', height = bar_height, label = 'FNN')

plt.ylabel('Features')
plt.title('Feature Importances')

for bar in rfr_bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', ha = 'left', va = 'center', fontsize = 10)

for bar in xgb_bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', ha = 'left', va = 'center', fontsize = 10)

for bar in fnn_bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f'{width:.4f}', ha = 'left', va = 'center', fontsize = 10)

plt.yticks(y, feature_importance['Feature'])
plt.legend();

plt.savefig('images/feature_importance.png', dpi = 300, bbox_inches = 'tight')
