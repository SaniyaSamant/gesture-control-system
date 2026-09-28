import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# 📌 Load dataset
df = pd.read_csv("D:/sem 5/AML/AI virtual mouse/gesture_control_ppt/final dataset.csv")

# 📌 Remove missing values
df.dropna(inplace=True)

# 📌 Extract features & labels
X = df.iloc[:, 1:].values  # Landmark data
y = df.iloc[:, 0].values   # Gesture labels

# 📌 Convert labels to numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 📌 Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 📌 Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y = to_categorical(y, num_classes)

# 📌 Reshape for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# 📌 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 📌 CNN Model with Improved Layers
model = Sequential([
    Conv1D(128, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    BatchNormalization(),
    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')  # Output layer
])

# 📌 Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 📌 Train with Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# 📌 Save model & label encoder
model.save("gesture_model_new1.h5")
with open("label_encoder_new2.pkl", "wb") as f:
    pickle.dump(label_encoder, f) 

print("✅ Model & Label Encoder Saved!")