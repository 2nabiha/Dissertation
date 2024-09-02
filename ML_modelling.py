# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score



# Load the dataset
file_path = '/Users/nabiha/PycharmProjects/Dissertation_COM726/CleanedCrimeData.csv'
data = pd.read_csv(file_path)

# --- Crime Type Prediction using RandomForestClassifier ---
# Encode the crime type as a categorical variable
label_encoder = LabelEncoder()
data['Crime type'] = label_encoder.fit_transform(data['Crime type'])

# Extract relevant features and target
features = data[['Latitude', 'Longitude']]  # You can add more features if available
target = data['Crime type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Feature scaling (standardizing the feature values)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model using accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Example prediction for new data
new_data = pd.DataFrame({'Latitude': [40.7128], 'Longitude': [-74.0060]})  # Example coordinates
new_data_scaled = scaler.transform(new_data)
predicted_crime_type = classifier.predict(new_data_scaled)
print(f"Predicted Crime Type: {label_encoder.inverse_transform(predicted_crime_type)}")
