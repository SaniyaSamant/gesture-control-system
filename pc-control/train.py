import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# ✅ Correct file path using raw string
df = pd.read_csv(r'C:\saniya\TY\cc\hand gesture project\final dataset.csv')  # Use raw string to avoid escape characters

# Separate features and labels
X = df.drop('gesture', axis=1)  # All columns except label
y = df['gesture']               # Label column

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted Gesture')
plt.ylabel('Actual Gesture')
plt.tight_layout()
plt.show()

# Save the trained model
with open('random_forest_gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Random Forest model saved as 'random_forest_gesture_model.pkl'")
