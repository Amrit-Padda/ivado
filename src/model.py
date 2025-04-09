from scraper import get_museum_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

museum_df = get_museum_data()
city_df = pd.read_csv('data/population_data.csv')

# Assuming all museum data is from 2023 (makes the projection easier) 
# we will train a regression model on this data and project the visitors for 2024

joined_data = museum_df.merge(city_df, left_on='city', right_on='City')
joined_data = joined_data[['name', 'type', 'collection_size', 'visitors', 'city', 'Population_2024', 'Population_2023', 'Growth Rate']]


# We perform some cleaning of the data to prepare for model training
mean_items = joined_data['collection_size'].mean()
joined_data['collection_size'] = joined_data['collection_size'].fillna(mean_items)
joined_data['collection_size'] = joined_data['collection_size'].astype('int64')

mode_type = joined_data['type'].mode()
joined_data['type'] = joined_data['type'].fillna(mode_type.iloc[0])

# Creating data for 2024 by multiplying visitors by the growth rate of the city
joined_data["visitors_2024"] = joined_data["visitors"] * (1 + joined_data["Growth Rate"])
joined_data["visitors_2024"] = joined_data["visitors_2024"].round().astype('int64')

# Encoding features, but keeping a copy of the df for validation later
encoder = LabelEncoder()
data_copy = joined_data.copy()
joined_data['type'] = encoder.fit_transform(joined_data['type'])
joined_data['city'] = encoder.fit_transform(joined_data['city'])

# Splitting features (X) and target variable (Y)
# The museum name is used solely for identification so we exclude it from encoding.
X = joined_data.drop(columns=['name', 'visitors_2024'], axis=1)
Y = joined_data['visitors_2024']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# Training model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Testing model on training data
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R Squared (Training Data) = ', r2_train)

# Evaluate model on test data
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared (Test Data) = ', r2_test)

# Predicting values using full dataset
prediction = regressor.predict(X)
data_copy['predicted_2024'] = prediction
data_copy['delta'] = data_copy['predicted_2024'] - data_copy['visitors_2024']
data_copy = data_copy[['name', 'city', 'Growth Rate', 'visitors', 'visitors_2024', 'predicted_2024', 'delta']]
print(data_copy.to_string())

