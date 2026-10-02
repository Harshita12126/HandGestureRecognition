import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# LOAD V2 DATASET
# ==========================================

data = pd.read_csv("dataset_v2/gestures.csv")

print("Dataset loaded!")
print("Total samples:", len(data))

# ==========================================
# FEATURES AND LABEL
# ==========================================

X = data.drop("label", axis=1)
y = data["label"]

print("\nNumber of features:", X.shape[1])

print("\nGesture distribution:")
print(y.value_counts().sort_index())


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining V2 model...")

model.fit(X_train, y_train)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("V2 MODEL TRAINING COMPLETED")
print("===================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "gesture_model_v2.pkl"
)

print("\n✅ Model saved as: gesture_model_v2.pkl")