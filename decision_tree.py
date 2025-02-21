import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# Train the Decision Tree model on the resampled (balanced) training data
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the (untouched) test set
y_pred = decision_tree_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Plot the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model,
          feature_names=X.columns,
          class_names=["Non-Fraud", "Fraud"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree")
plt.show()
