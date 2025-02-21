import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

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
# Here, sampling_strategy=0.5 means that after undersampling, the number of majority examples 
# will be twice that of the minority. (You can adjust this value based on your needs.)
rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print("\nAfter undersampling, training class distribution:")
print(pd.Series(y_train_under).value_counts())

# STEP 2: Use SMOTE to oversample the minority class so that classes become balanced.
# sampling_strategy=1.0 tells SMOTE to upsample the minority class to have the same number as the majority.
smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

print("\nAfter SMOTE, resampled training class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Standardize the features using training data statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model on the resampled (balanced) training data
log_reg_model = LogisticRegression(random_state=42)
log_reg_model.fit(X_train_scaled, y_train_resampled)

# Make predictions on the (untouched) test set
y_pred = log_reg_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
