# EDA
import pandas as pd


# Load the cleaned subset data
file_path = '/Users/nabiha/PycharmProjects/Dissertation_COM726/CleanedCrimeData.csv'
crime_data_subset = pd.read_csv(file_path)


# Univariate analysis
# Display basic information about the dataset
print("Descriptive Information:")
print(crime_data_subset.info())

# Generate summary statistics for numeric and categorical fields
summary_statistics = crime_data_subset.describe(include='all')
print(summary_statistics)

# Display unique values count for each column
print("\nNumber of unique values in each column:")
for column in crime_data_subset.columns:
    unique_count = crime_data_subset[column].nunique()
    print(f"{column}: {unique_count} unique values")

# Plotting the distribution of crimes by type
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(y='Crime type', data=crime_data_subset, order=crime_data_subset['Crime type'].value_counts().index)
plt.title('Distribution of Crime Types')
plt.show()



# Bivariate analysis

# Monthly distribution of crime types
# Convert 'Month' to datetime if not already done
crime_data_subset['Month'] = pd.to_datetime(crime_data_subset['Month'])

plt.figure(figsize=(14, 8))
sns.countplot(x=crime_data_subset['Month'].dt.month, hue='Crime type', data=crime_data_subset, palette='tab10')
plt.title('Monthly Distribution of Crime Types')
plt.xlabel('Month')
plt.ylabel('Count of Crimes')
plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# temporal analysis
# Group by Month to see trends over time
import pandas as pd
import matplotlib.pyplot as plt

# Ensure that the 'Month' column is in datetime format
crime_data_subset['Month'] = pd.to_datetime(crime_data_subset['Month'], errors='coerce')

# Group by Month to see trends over time
crime_by_month = crime_data_subset.groupby(crime_data_subset['Month'].dt.to_period('M')).size()

plt.figure(figsize=(14, 7))
crime_by_month.plot(kind='line')
plt.title('Crime Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.show()

# Basic Anomaly detection
# Outliers Detection
q1 = crime_by_month.quantile(0.25)
q3 = crime_by_month.quantile(0.75)
iqr = q3 - q1

# Define outliers as points outside 1.5 * IQR
outliers = crime_by_month[(crime_by_month < (q1 - 1.5 * iqr)) | (crime_by_month > (q3 + 1.5 * iqr))]

print("Outliers in monthly crime data:")
print(outliers)

# advanced anomaly detection: Z-score analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Calculate the Z-scores for the monthly crime counts
z_scores = stats.zscore(crime_by_month.values)

# Convert Z-scores to a Pandas Series to maintain the index
z_scores_series = pd.Series(z_scores, index=crime_by_month.index)

# Identify outliers (e.g., data points where Z-score is greater than 2 or less than -2)
outliers = crime_by_month[(z_scores_series > 2) | (z_scores_series < -2)]

# Print the outliers
print("Outliers based on Z-score analysis:")
print(outliers)

# Optionally, plot the Z-scores
plt.figure(figsize=(14, 7))
plt.plot(crime_by_month.index.to_timestamp(), z_scores_series, marker='o', linestyle='-', color='b')
plt.axhline(y=2, color='r', linestyle='--')
plt.axhline(y=-2, color='r', linestyle='--')
plt.title('Z-scores of Monthly Crime Data')
plt.xlabel('Month')
plt.ylabel('Z-score')
plt.show()

# Plot time series with outliers highlighted
plt.figure(figsize=(14, 7))
crime_by_month.plot(kind='line', label='Monthly Crime Counts')

if not outliers.empty:
    plt.scatter(outliers.index.to_timestamp(), outliers, color='red', label='Outliers')

plt.title('Outliers in Monthly Crime Data')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.legend()
plt.show()

# MULTIVARIATE ANALYSIS
# Correlation analysis
# Create a correlation matrix
crime_types_dummies = pd.get_dummies(crime_data_subset['Crime type'])
correlation_matrix = crime_types_dummies.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Crime Types')
plt.show()


# KDE
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


# Convert the crime data into a GeoDataFrame if it's not already one
gdf = gpd.GeoDataFrame(crime_data_subset,
                       geometry=gpd.points_from_xy(crime_data_subset.Longitude, crime_data_subset.Latitude))

# Set up the plot
plt.figure(figsize=(10, 8))

# KDE Plot using seaborn
sns.kdeplot(
    x=gdf.geometry.x,  # Longitude
    y=gdf.geometry.y,  # Latitude
    cmap="Reds",  # Color map
    fill=True,  # Fill the contours
    thresh=0,  # No threshold, display full range
    levels=100  # Number of contour levels
)

# Overlay the scatter plot of crime incidents for reference
plt.scatter(gdf.geometry.x, gdf.geometry.y, s=1, color='black', alpha=0.3)

# Title and labels
plt.title('Crime Density Estimation Using KDE')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Display the plot
plt.show()


# Severity of crimes

# Ensure 'Month' is in datetime format
crime_data_subset['Month'] = pd.to_datetime(crime_data_subset['Month'])

# Extract the year and month for grouping
crime_data_subset['Year'] = crime_data_subset['Month'].dt.year
crime_data_subset['Month Name'] = crime_data_subset['Month'].dt.strftime('%B')

# Classify crimes into Violent and Non-Violent
violent_crimes = ['Violence and sexual offences', 'Robbery']
crime_data_subset['Severity'] = crime_data_subset['Crime type'].apply(lambda x: 'Violent' if x in
                                                                                             violent_crimes
else'Non-Violent')

# Sort months chronologically
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Group the data by year, month, and severity
severity_by_month = crime_data_subset.groupby(['Year', 'Month Name', 'Severity']).size().reset_index(name='Crime Count')

# Plotting
years = severity_by_month['Year'].unique()

for year in years:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Month Name', y='Crime Count', hue='Severity', data=severity_by_month[severity_by_month['Year'] == year], order=month_order)
    plt.title(f'Violent vs Non-Violent Crimes per Month in {year}')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45)
    plt.legend(title='Severity')
    plt.show()

# Outcome category
import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (assuming the dataset is in a DataFrame called crime_data)
crime_data = pd.read_csv('CrimeData.csv')

# Ensure 'Month' is in datetime format and extract the year and month name
crime_data['Month'] = pd.to_datetime(crime_data['Month'])
crime_data['Year'] = crime_data['Month'].dt.year
crime_data['Month Name'] = crime_data['Month'].dt.strftime('%B')

# Define month order for consistent plotting
month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Get the unique years in the dataset
years = crime_data['Year'].unique()

# Loop through each year and create a chart for each
for year in years:
    # Filter the data for the specific year
    data_for_year = crime_data[crime_data['Year'] == year]

    # Group by month and 'Last outcome category'
    outcome_by_month = data_for_year.groupby(['Month Name', 'Last outcome category']).size().unstack().fillna(0)

    # Sort the DataFrame by month order
    outcome_by_month = outcome_by_month.reindex(month_order, axis=0)

    # Plot the results
    outcome_by_month.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title(f'Crime Outcomes by Month in {year}')
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.xticks(rotation=45)
    plt.legend(title='Last Outcome Category')
    plt.show()

# Chi-square test for independence
import pandas as pd
from scipy.stats import chi2_contingency

# Extract the month name for the analysis
crime_data_subset['Month Name'] = crime_data_subset['Month'].dt.month_name()

# Verify the new 'Month Name' column
print(crime_data_subset[['Month', 'Month Name']].head())

# Create a contingency table: Crime Type vs. Month
contingency_table = pd.crosstab(crime_data_subset['Month Name'], crime_data_subset['Crime type'])

# Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output the results
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)


# Chi square for location and crime type
import pandas as pd
from scipy.stats import chi2_contingency

# Combine Longitude and Latitude into a single 'Location' identifier
crime_data_subset['Location'] = crime_data_subset['Longitude'].astype(str) + ', ' + crime_data_subset['Latitude'].astype(str)

# Create a contingency table: Combined Location vs. Crime Type
contingency_table = pd.crosstab(crime_data_subset['Location'], crime_data_subset['Crime type'])

# Perform the Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Output the results
print("Chi-Square Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)
