import pandas as pd

# Load data
file_path = '/Users/nabiha/PycharmProjects/Dissertation_COM726/CrimeData.csv'
crime_data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Descriptive Information:")
print(crime_data.info())

# Generate summary statistics for numeric and categorical fields
summary_statistics = crime_data.describe(include='all')
print(summary_statistics)

# Display unique values count for each column
print("\nNumber of unique values in each column:")
for column in crime_data.columns:
    unique_count = crime_data[column].nunique()
    print(f"{column}: {unique_count} unique values")

# List all unique values in the specified column
unique_values = crime_data['Crime type'].unique()
# Print the unique values
print(f"Unique values in the column '{'Crime type'}':")
print(unique_values)


# Pre-processing
# Drop rows with missing values in essential columns
crime_data_cleaned = crime_data.dropna(subset=['Month', 'Longitude', 'Latitude', 'Crime type'])

# Remove duplicates only if all columns in the row are identical
crime_data_cleaned = crime_data_cleaned.drop_duplicates()

# Convert the 'Month' column to a datetime object
crime_data_cleaned['Month'] = pd.to_datetime(crime_data_cleaned['Month'])

# Create a subset of the cleaned data with only the relevant columns
crime_data_subset = crime_data_cleaned[['Month', 'Longitude', 'Latitude', 'Crime type']]

# Save the subset as a new CSV file
crime_data_subset.to_csv('/Users/nabiha/PycharmProjects/Dissertation_COM726/CleanedCrimeData.csv', index=False)

# Display cleaned data
print("Missing values after cleaning:")
print(crime_data_cleaned[['Month', 'Longitude', 'Latitude', 'Crime type']].isnull().sum())
print("Cleaned data sample:")
print(crime_data_subset.head())
print(crime_data_subset.isnull().sum())
