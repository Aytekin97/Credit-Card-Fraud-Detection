import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Import the dataset
data = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# Drop unnecessary columns
data_cleaned = data.drop(columns=["isFlaggedFraud", "nameOrig", "nameDest"])

# Encode the 'type' column using Label Encoding
label_encoder = LabelEncoder()
data_cleaned["type"] = label_encoder.fit_transform(data_cleaned["type"])

# Define features (X) and target variable (y)
X = data_cleaned.drop(columns=["isFraud"])
y = data_cleaned["isFraud"]

# Split the dataset into training and testing sets (apply resampling on training only)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42)

print("Original training class distribution:")
print(y_train.value_counts())

# STEP 1: Undersample the majority class on the training set.
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("\nAfter undersampling, training class distribution:")
print(pd.Series(y_train_under).value_counts())

# STEP 2: Use SMOTE to oversample the minority class so that both classes become balanced.
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

print("\nAfter SMOTE, resampled training class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Train the Random Forest model on the resampled (balanced) training data
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the (untouched) test set
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot feature importances
importances = random_forest_model.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
