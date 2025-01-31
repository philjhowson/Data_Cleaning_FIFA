import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
import pickle
import json

"""
load in the previous data made from the train test split. However,
because I want to have also a validation set, I will split the
training data into train/val sets. The test set will remain as the
same test set used in the other models.
"""

X = pd.read_csv('data/processed_data/X_train.csv')
y = pd.read_csv('data/processed_data/y_train.csv')

X_tensor = torch.tensor(X.values, dtype = torch.float32)
y_tensor = torch.tensor(y.values, dtype = torch.float32)

x_test = pd.read_csv('data/processed_data/X_test.csv')
Y_test = pd.read_csv('data/processed_data/y_test.csv')

X_test = torch.tensor(x_test.values, dtype = torch.float32)
y_test = torch.tensor(Y_test.values, dtype = torch.float32)

X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size = 0.3, random_state = 42)

"""
create the fnn model. Here, I went with 4 hidden layers and
a maximum of 1024 neurons. I utilized batch normilization
and drop out to help prevent overfitting. ReLU activation.
"""

class FootballerWagePrediction(nn.Module):
    def __init__(self, input_size = 6, output_size = 1):
        super(FootballerWagePrediction, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.fnn(x)
        return x

"""
set the device to "cuda" if it's available, create the model
object and move it to the device, and initialize weights.
I use Xavier here because it tends to go well with fully
connected layers, like in an FNN model.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FootballerWagePrediction().to(device)

def initialize_weights(m):
    if isinstance(m, nn.Linear):  # Apply to Linear layers only
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(initialize_weights)

"""
create the dataset objects and set the batch size, then create
the loader objects.
"""

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

batches = 256

train_loader = DataLoader(train_data, batch_size = batches, shuffle = True)
val_loader = DataLoader(val_data, batch_size = batches, shuffle = True)
test_loader = DataLoader(test_data, batch_size = batches, shuffle = True)

"""
I chose an Adam optimizer, as it tends to perform well in many tasks,
with MSELoss criterion, which is suitable for this type of model.
I set the scheduler to ReduceLROnPlateau to facilitate better and more
continuous learning.
"""

optimizer = Adam(model.parameters(), lr = 1e-03, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 1e-04)
criterion = nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3)

"""
I have version because a little playing with the optimizer params and batch
size are often necessary, as with the model architecture. So as not to overwrite
potentially better models and results, versioning is optional here.

I create also the epochs, patience, and best_val_loss variables for use in the
loop below. As well as the no_improve_counter and I initialize an empty dictionary
to keep track of metrics to be plotted later.
"""

version = 7
epochs = 10001
patience = 20
best_val_loss = float('inf')
no_improve_counter = 0

history = {'epoch' : [], 'train_loss' : [], 'train_r2' : [],
           'val_loss' : [], 'val_r2' : []}

for epoch in range(1, epochs):
    train_loss = 0
    predictions = []
    labels = []
    val_predictions = []
    val_labels = []

    model.train()

    """
    load training data, move to device, get the model predictions
    and the loss, zero_grad() the optimizer before loss.backward()
    and optimizer.step(). Continue to accumulate the total loss
    and appened the predictions and true labels for analysis after
    the batch.
    """
    
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        train_predictions = model(X_batch)
        loss = criterion(train_predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predictions.append(train_predictions.detach().cpu().numpy())
        labels.append(y_batch.detach().cpu().numpy())

    """
    convert the labels and predictions into a single 1d list,
    divide the training loss by the number of batches to get the
    average training loss, and calculate the training r2. Append
    all values to the history.
    """
    
    labels_flat = np.concatenate(labels).flatten()
    predictions_flat = np.concatenate(predictions).flatten()
    train_loss /= len(train_loader)
    r2 = r2_score(labels_flat, predictions_flat)

    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_r2'].append(r2)

    """
    for validation, the same coding/logic is used, except that the
    model is set first to .eval() and everything is done with
    torch.no_grad() to speed up calculations as these are not necessary.
    """
    
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for val_batch in val_loader:
            X_val, y_val = val_batch
            X_val, y_val = X_val.to(device), y_val.to(device)
    
            val_prediction = model(X_val)
            loss = criterion(val_prediction, y_val)
    
            val_loss += loss.item()
            val_predictions.append(val_prediction.detach().cpu().numpy())
            val_labels.append(y_val.detach().cpu().numpy())

    val_labels_flat = np.concatenate(val_labels).flatten()
    val_predictions_flat = np.concatenate(val_predictions).flatten()
    val_loss /= len(val_loader)
    val_r2 = r2_score(val_labels_flat, val_predictions_flat)

    history['val_loss'].append(val_loss)
    history['val_r2'].append(val_r2)

    """
    step the schedule with the val_loss. To reduce clutter on a potentially
    high epoch run, I use modulo for 1000 to print intermediate results. If
    val_loss does not improve, the counter adds by one, if it is the val_loss
    does improve, save the model and reset the counter to 0. If the counter
    is greater or equal to patience, break.
    """
    
    scheduler.step(val_loss)

    if epoch % 1000 == 0:
    
        print(f"Epoch: {epoch}; Training Loss: {train_loss}, Training r2: {r2}")
        print(f"Validation Loss: {val_loss}, Validation r2: {val_r2}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve_counter = 0
        torch.save(model.state_dict(), f"models/fnn_v{version}_best_save.pth")
    else:
        no_improve_counter += 1

    if no_improve_counter >= patience:
        print(f"Early stopping triggered after {epoch} epochs.")
        break

"""
plot the training and validation metrics per epoch.
"""

fig, ax = plt.subplots(1, 2, figsize = (15, 5))

epochs = max(history['epoch'])

train_index = ['train_loss', 'train_r2']
val_index = ['val_loss', 'val_r2']

train = ['Training Loss', 'Training r²']
val = ['Validation Loss', 'Validation r²']
y_index = ['Loss', 'r²']

titles = ['Training and Validation Loss', 'Training and Validation r²']

for index, axes in enumerate(ax.flat):
    axes.plot(range(1, epochs + 1), history[train_index[index]], label = train[index], color = 'blue')
    axes.plot(range(1, epochs + 1), history[val_index[index]], label = val[index], color = 'purple')
    axes.set_title(titles[index])
    axes.set_ylabel(y_index[index])
    axes.legend(loc = 'best')

"""
because the final model may not be the best model, reload the best weights
in order to evaluate the results. The coding is the same as the validation
coding above.
"""

model.load_state_dict(torch.load(f"models/fnn_v{version}_best_save.pth", weights_only = True))

model.eval()
test_predictions = []
test_labels = []

with torch.no_grad():
    for test_batch in test_loader:
        X_val, y_val = test_batch
        X_val, y_val = X_val.to(device), y_val.to(device)
    
        prediction = model(X_val)
    
        test_predictions.append(prediction.detach().cpu().numpy())
        test_labels.append(y_val.detach().cpu().numpy())

    test_labels_flat = np.concatenate(test_labels).flatten()
    test_predictions_flat = np.concatenate(test_predictions).flatten()
    test_r2 = r2_score(test_labels_flat, test_predictions_flat)

    print(f"Test r2: {test_r2}")

"""
create and save the scores for training and testing. Then
use the shap library to calculate the shap values for the
dataset and save the shap values.
"""

scores = {'Training Score' : [max(history['train_r2'])],
          'Test Score' : [test_r2]}

with open('metrics/fnn_v{version}_scores.json', 'w') as f:
    json.dump(scores, f, indent = 4)

explainer = shap.DeepExplainer(model, X_train.to(device))
shap_values = explainer.shap_values(X_test.to(device))

with open(f'metrics/fnn_v{version}_shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f)

"""
squeeze(-1) so the shape values and X_test dimensions match and
then plot the results. Mean values are calculated and normalized
so that they are comparable to the other models and those values
are saved.
"""

shap_values = shap_values.squeeze(-1)
shap.summary_plot(shap_values, X_test, feature_names = X.columns)

mean_shap_values = np.abs(shap_values).mean(axis = 0).flatten()
total_importance = mean_shap_values.sum()

normalized_importance = mean_shap_values / total_importance

feature_importance_fnn = pd.Series(normalized_importance,
    index = X.columns,
)

feature_importance = feature_importance_fnn.to_dict()

with open('metrics/feature_importance_fnn.json', 'w') as f:
    json.dump(feature_importance, f, indent = 4)