import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

categorical = ['Warehouse_block', 'Mode_of_Shipment','Product_importance','Gender']
data = pd.read_csv('./Datasets/E-Commerce_train.csv', sep = ';')
data.drop('ID', axis = 1, inplace = True)
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(data[categorical]).toarray()
feature_labels = ohe.categories_
feature_labels = np.hstack(feature_labels).ravel()
features = pd.DataFrame(feature_array, columns = feature_labels)
df = pd.concat([data, features], axis = 1)
df.drop(categorical, axis = 1, inplace = True)

# Defining our variables
X = df.drop('Reached.on.Time_Y.N', axis = 1)
y = df['Reached.on.Time_Y.N']
# Split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 101)

# Working with test dataset
test_data = pd.read_csv('./Datasets/E-Commerce_test.csv', sep = ";")
test_data.drop('ID', axis = 1, inplace = True)

# Defining categorical features
categorical = ['Warehouse_block', 'Mode_of_Shipment','Product_importance','Gender']
# Encoding categorical features
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(test_data[categorical]).toarray()
feature_labels = np.hstack(ohe.categories_).ravel()
# Concating dataframes and creating the final one
features = pd.DataFrame(feature_array, columns = feature_labels)
test_df = pd.concat([test_data, features], axis = 1)
test_df.drop(categorical, axis = 1, inplace = True)

# Defining our variables
X = test_df
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Model
clf = SVC()
clf.fit(X_train, y_train)
y_test_preds = clf.predict(X_scaled)
y_test_preds
file = pd.DataFrame(y_test_preds, columns = ['pred'])
file.to_csv('Jeanfabra.csv', index = False)