import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

data = pd.read_csv('data/processed_data/cleaned_data.csv')

"""
there are a lot of columns, so for simplification sake, a number of columns will be dropped, especially features
that are necessarily unique, like player name, or perhaps not as informative as other features, such as Nationality,
or Month Joined.

while I did calculate features to drop, I actually decided to drop those features later, so that the whole set
of possible features was available and easy to access should further experimentation with features and models
be desired.
"""

data.drop(columns = ['ID', 'Name', 'LongName', 'photoUrl', 'playerUrl', 'Nationality', 'Club', 'Month Joined', 'Day Joined', 'Loan Date End', 'Positions', 'Preferred Foot'], inplace = True)

enc = OneHotEncoder(sparse_output = False)

positions = pd.DataFrame(enc.fit_transform(pd.DataFrame(data['Best Position'])), columns = enc.get_feature_names_out())
positions.columns = positions.columns.str.replace('Best Position_', '')

data = pd.concat([data, positions], axis = 1)
data.drop(columns = ['Best Position'], inplace = True)

"""
the variables 'A/W' and 'D/W' are both coded 'Low', 'Medium', 'High' and so can
be changed 1, 2, 3, here.
"""

mapping = {'Low': 0, 'Medium': 1, 'High': 2}

data['A/W'] = data['A/W'].map(mapping)
data['D/W'] = data['D/W'].map(mapping)

"""
the rest of the variables will be scaled with MinMaxScaler.
"""

y = data['Wage (€K)']
to_be_scaled = data.drop(columns = ['Wage (€K)', 'A/W', 'D/W'], axis = 1)
already_scaled = data[['A/W', 'D/W']]

enc = MinMaxScaler()
scaled = pd.DataFrame(enc.fit_transform(to_be_scaled), columns = to_be_scaled.columns)

X = pd.concat([scaled, already_scaled], axis = 1)

y.to_csv('data/processed_data/y_data.csv', index = False)
X.to_csv('data/processed_data/X_data.csv', index = False)