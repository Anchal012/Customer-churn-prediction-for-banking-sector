# -*- coding: utf-8 -*-
"""Copy of Churn Modelling (ANN Final) .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fR_QYgtzR0Z_p6x50bfxs5UylLKy2gPN

# Artificial Neural Network
"""
# Loading Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import load_model
model=load_model('mymodel.h5')

def preprocess_input(data):
    # Label encoding the "Gender" column
    le = LabelEncoder()
    data[:, 2] = le.fit_transform(data[:, 2])

    # One hot encoding the "Geography" column
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    data = np.array(ct.fit_transform(data))

    # Initialize StandardScaler and fit it to the training data
    sc = StandardScaler()
    sc.fit(data)

    # Feature scaling
    data = sc.transform(data)

    return data

def predict_churn(data):
    # Make predictions using the loaded model
    prediction = model.predict(data)
    return prediction.flatten() > 0.5  # Convert probabilities to binary predictions

# Importing dataset
dataset = pd.read_csv("Churn_Modelling.csv")

# Filtering data
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Preprocess input data
x = preprocess_input(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Building the ANN
ann = tf.keras.models.Sequential()  # Initializing the ANN
ann.add(tf.keras.layers.Dense(units=16, input_dim=12, activation='relu'))  # Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))  # Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=8, activation='relu'))  # Adding the third hidden layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Adding the output layer

# Compiling the ANN model
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN model on the Training set
history = ann.fit(x_train, y_train, batch_size=64, epochs=500)

# Save the model
ann.save('mymodel.h5')