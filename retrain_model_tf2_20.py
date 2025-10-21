import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")

# Loading the dataset
data = pd.read_csv("Churn_Modelling.csv")

# Preprocessing the data - Drop irrelevant features
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encoding the categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

# Applying One-hot encoding to the Geography column
one_hot_encoder_geo = OneHotEncoder()
geo_encoder = one_hot_encoder_geo.fit_transform(data[['Geography']])
geo_encoder = geo_encoder.toarray()

# Create DataFrame for encoded geography
geo_encoded_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder_geo.get_feature_names_out())

# Combining the encoded columns with the original dataset
data = pd.concat([data.drop('Geography', axis=1), geo_encoded_df], axis=1)

# Saving the encoders
with open('label_encoder_gender.pkl', 'wb') as file:
    pickle.dump(label_encoder_gender, file)

with open('onehot_encoder_geo.pkl', 'wb') as file:
    pickle.dump(one_hot_encoder_geo, file)

# Splitting the data into independent and dependent features
X = data.drop('Exited', axis=1)
y = data['Exited']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Building the deep learning model (same architecture as original)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model with TensorFlow 2.20.0 compatible settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='binary_crossentropy',  # Using string instead of tf.losses.BinaryCrossentropy()
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Setting up Early Stopping
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

print("Training the model...")
# Training the model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test),
    epochs=100,
    callbacks=[early_stopping_callback],
    verbose=1
)

# Save the model
model.save('model.h5')
print("Model saved as 'model.h5'")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

print("Model retraining completed successfully!")