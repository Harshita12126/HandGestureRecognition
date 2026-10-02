import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("dataset/gestures.csv")

print("Dataset loaded!")
print("Total samples:", len(data))

# Separate features and labels
X = data.drop("label", axis=1)
y = data["label"]

print("\nGestures found:")
print(y.value_counts())


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create model
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)


# Train
print("\nTraining model...")
model.fit(X_train, y_train)


# Test
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save model
joblib.dump(model, "gesture_model.pkl")

print("\n✅ Model saved as: gesture_model.pkl")