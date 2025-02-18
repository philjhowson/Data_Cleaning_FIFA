import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('data/processed_data/cleaned_data.csv')

"""
first, separate out all the numerical and categorical
columns for analysis.
"""

numerical = []
categorical = []

for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        numerical.append(column)
    else:
        categorical.append(column)

"""
generates a scatterplot for selected numerical feature to
compare it against wage and see the relationship. I only
do some features here for ease of plotting, but in full data
exploration, all features are initially compared with wage.
"""

features = ['Value (€M)', 'Hits', 'IR', 'Total Stats', 'POT', '↓OVA']

fig, ax = plt.subplots(2, 3, figsize = (15, 10))

for index, axes in enumerate(ax.flat):
    axes.scatter(data[features[index]], data['Wage (€K)'])
    axes.set_title(f"{features[index]} vs Wage (€K)")
    axes.set_xlabel(f"{features[index]}")
    axes.set_ylabel('Wage (€K)')

plt.tight_layout();

plt.savefig("images/scatter_plots_vs_wage.png")

"""
generate a correlation matrix of all the features, then remove
any low correlations with +/- 0.3 threshold and plot the
reduced feature space.
"""

numerical_data = data[numerical]
corr_mat = numerical_data.corr()

plt.figure(figsize = (15, 15))
sns.heatmap(corr_mat, annot = True, cmap = 'coolwarm', fmt = '.2f', cbar = True);

stronger = []
weaker = []

for correlation in range(len(corr_mat)):
    if corr_mat['Wage (€K)'].iloc[correlation] >= 0.3 or corr_mat['Wage (€K)'].iloc[correlation] <= -0.3:
        stronger.append(corr_mat.columns[correlation])
    else:
        weaker.append(corr_mat.columns[correlation])

stronger_corr = data[stronger]
corr_mat = stronger_corr.corr()

plt.figure(figsize = (15, 15))
sns.heatmap(corr_mat, annot = True, cmap = 'coolwarm', fmt = '.2f', cbar = True);

stronger.remove('Wage (€K)')

with open('data/processed_data/strong_correlations.pkl', 'wb') as f:
    pickle.dump(stronger, f)

with open('data/processed_data/weak_correlations.pkl', 'wb') as f:
    pickle.dump(weaker, f)

"""
I want to deal with multicolinarity because it makes models
easier to interpret and enhances performance for many models
as well, so here I mark down which features have high correlations
with each other, set at 0.7.
"""

correlation_matrix = corr_mat.copy()
correlation_matrix.drop(columns = ['Wage (€K)'], inplace = True)

features_to_drop = []

threshold = 0.7

for i, column in enumerate(correlation_matrix.columns):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= threshold:
            feature = correlation_matrix.columns[i]
            if feature in features_to_drop:
                pass
            else:
                features_to_drop.append(feature)

"""
Looking at the short passing stat it has seems that 'Total Stats'
correlates strongly not only with 'Short Passing', but all the other
metrics, but it makes more sense from a feature point of view to
keep 'Total Stats' over 'Short Passing' since it represents the
skill set as a whole of the player.
"""

features_to_drop.remove('Total Stats')
features_to_drop.append('Short Passing')
                
with open('data/processed_data/drop_columns.pkl', 'wb') as f:
    pickle.dump(features_to_drop, f)

"""
produce a correlation matrix and visualize it for the reduced
feature set.
"""

reduced_corr = data[stronger].drop(columns = features_to_drop)
reduced_corr['Wage (€K)'] =  data['Wage (€K)']
reduced_corr_mat = reduced_corr.corr()

plt.figure(figsize = (15, 15))
sns.heatmap(reduced_corr_mat, annot = True, cmap = 'coolwarm', fmt = '.2f', cbar = True);

plt.tight_layout();

plt.savefig("images/strong_correlation_heatmap.png")
