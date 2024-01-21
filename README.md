import pandas as pd
# Read the two CSV files into DataFrames
file1 = pd.read_csv('/content/US_births_1994-2014_CDC_NCHS.csv')
# Display the first few rows of each DataFrame
print("File 1:")
print(file1.head())
print (file1.tail())

# Summary statistics
summary_stats = file1['births'].describe()
print("Summary Statistics for Births:")
print(summary_stats)
import matplotlib.pyplot as plt
# Bar plot
plt.bar(file1['year'], file1['births'], label='Bar Plot', color='green')
plt.title("Year vs Births")
plt.xlabel("Year")
plt.ylabel("Births")
plt.legend()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.kdeplot(file1['births'], fill=True, color='skyblue')
plt.title('Kernel Density Estimate (KDE) Plot of Births')
plt.xlabel('Number of Births')
plt.ylabel('Density')
plt.grid(True)
plt.show()

import pandas as pd
import plotly.express as px
# Read the dataset into a DataFrame
df = pd.read_csv("/content/US_births_1994-2014_CDC_NCHS.csv")
# Group by date_of_month and calculate the mean births for each date
daily_births = df.groupby('date_of_month')['births'].mean().reset_index()
# Create an interactive line plot using Plotly Express
fig = px.line(daily_births, x='date_of_month', y='births', markers=True, title='Interactive Plot - Number of Births')
# Add data labels to each point
fig.update_traces(text=daily_births['births'].astype(int).tolist(), textposition='top center')
# Update layout for better visibility
fig.update_layout(xaxis_title='Date of Month', yaxis_title='Number of Births', hovermode='closest')
# Show the plot
fig.show()

import matplotlib.pyplot as plt
# Create a DataFrame for actual births vs. month
actual_vs_month = pd.DataFrame({'Month': X_test['month'], 'Actual Births': y_test})
# Sort the DataFrame by month for better visualization
actual_vs_month = actual_vs_month.sort_values(by='Month')
# Plot the actual births vs. month
plt.figure(figsize=(10, 6))
plt.scatter(actual_vs_month['Month'], actual_vs_month['Actual Births'], label='Actual Births', marker='o', color='blue')
plt.xlabel('Month')
plt.ylabel('Number of Births')
plt.title('Actual Births vs. Month')
plt.legend()
plt.grid(True)
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Read the dataset into a DataFrame
df = pd.read_csv("/content/US_births_1994-2014_CDC_NCHS.csv")

# Group by date_of_month and calculate the mean births for each date
daily_births = df.groupby('date_of_month')['births'].mean().reset_index()

# Plot the relationship between date_of_month and mean births
plt.figure(figsize=(12, 6))
sns.lineplot(x='date_of_month', y='births', data=daily_births, marker='o', color='skyblue')

# Annotate specific points (e.g., highest and lowest)
max_births_date = daily_births.loc[daily_births['births'].idxmax()]
min_births_date = daily_births.loc[daily_births['births'].idxmin()]

plt.annotate(f"Max Births: {max_births_date['births']}",
             xy=(max_births_date['date_of_month'], max_births_date['births']),
             xytext=(-20, 10),
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'))

plt.annotate(f"Min Births: {min_births_date['births']}",
             xy=(min_births_date['date_of_month'], min_births_date['births']),
             xytext=(20, -10),
             textcoords='offset points',
             arrowprops=dict(arrowstyle="->", color='black'))

plt.xlabel('Date of Month')
plt.ylabel('Mean Number of Births')
plt.title('Relationship between Date of Month and Mean Births with Annotations')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plotCorrelationMatrix(df, graphWidth):

    df = df.dropna('columns')  # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return

    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title('Correlation Matrix', fontsize=15)  # Removed reference to df in the title
    plt.show()
df = pd.read_csv('/content/US_births_1994-2014_CDC_NCHS.csv')
# Select features and target variable
features = df[['year', 'month', 'date_of_month', 'day_of_week']]
target = df['births']
df_for_corr = pd.concat([features, target], axis=1)
# Use the plotCorrelationMatrix function to plot the correlation matrix
plotCorrelationMatrix(df_for_corr, 10)  # Adjust the graphWidth parameter as needed

correlation_matrix = features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
# Creating X and y


file1 = pd.read_csv('/content/US_births_1994-2014_CDC_NCHS.csv')
X = file1['year']
y = file1['births']
# Splitting the varaibles as training and testing
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the resulting sets
print("Training set - Features shape:", X_train.shape)
print("Training set - Target shape:", y_train.shape)
print("Testing set - Features shape:", X_test.shape)
print("Testing set - Target shape:", y_test.shape)

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_csv('/content/US_births_1994-2014_CDC_NCHS.csv')
features = df[['year', 'month', 'date_of_month', 'day_of_week']]
target = df['births']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)  # Regression line
plt.xlabel("Actual Births")
plt.ylabel("Predicted Births")
plt.title("Actual vs. Predicted Births (Linear Regression)")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming df is your original DataFrame
df = pd.read_csv('/content/US_births_1994-2014_CDC_NCHS.csv')

# Extract features and target variable
features = df[['year', 'month', 'date_of_month', 'day_of_week']]
target = df['births']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)




# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
specific_year = 2025
dates_for_specific_year = pd.date_range(start=f"{specific_year}-01-01", end=f"{specific_year}-12-31", freq='D')
features_for_specific_year = pd.DataFrame({
    'year': specific_year,
    'month': dates_for_specific_year.month,
    'date_of_month': dates_for_specific_year.day,
    'day_of_week': dates_for_specific_year.dayofweek
})
predictions_for_specific_year = model.predict(features_for_specific_year)
print(f'Predicted Births for the Year {specific_year}:\n')
for date, prediction in zip(dates_for_specific_year, predictions_for_specific_year):
    print(f"{date.strftime('%Y-%m-%d')}: {prediction:.2f} births")
    # Find the date corresponding to the minimum and maximum predicted births
min_date = dates_for_specific_year[predictions_for_specific_year.argmin()]
max_date = dates_for_specific_year[predictions_for_specific_year.argmax()]

# Find the corresponding minimum and maximum predicted births
min_births = predictions_for_specific_year.min()
max_births = predictions_for_specific_year.max()
# Print the results
print(f'Minimum Predicted Births ({min_date.strftime("%Y-%m-%d")}): {min_births:.2f} births')
print(f'Maximum Predicted Births ({max_date.strftime("%Y-%m-%d")}): {max_births:.2f} births')

# Split the data into training and testing sets with a fixed random state
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)


# Print the mse error and mae error
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

import matplotlib.pyplot as plt

# Plot the predicted births with arrows
plt.figure(figsize=(12, 6))
plt.plot(dates_for_specific_year, predictions_for_specific_year, label='Predicted Births', color='blue')
plt.scatter([min_date, max_date], [min_births, max_births],
            color=['red', 'green'], label=['Minimum Births', 'Maximum Births'], zorder=5)

# Draw arrows with text annotation
plt.annotate(f'Minimum: {min_births:.2f} births', xy=(min_date, min_births), xytext=(min_date, min_births - 500),
             arrowprops=dict(facecolor='red', arrowstyle='->'), color='red', fontsize=9, ha='center')
plt.annotate(f'Maximum: {max_births:.2f} births', xy=(max_date, max_births), xytext=(max_date, max_births + 500),
             arrowprops=dict(facecolor='green', arrowstyle='->'), color='green', fontsize=9, ha='center')

plt.title(f'Predicted Births for the Year {specific_year}')
plt.xlabel('Date')
plt.ylabel('Predicted Number of Births')
plt.legend()
plt.grid(True)
plt.show()
# Print minimum and maximum birth values and their dates
min_births = predictions_for_specific_year.min()
max_births = predictions_for_specific_year.max()

print(f"Minimum Predicted Births ({min_date}): {min_births:.2f}")
print(f"Maximum Predicted Births ({max_date}): {max_births:.2f}")

import plotly.express as px

# Predict births for the specific year
predictions_for_specific_year = model.predict(features_for_specific_year)

# Combine actual and predicted values into a DataFrame for plotting
plot_data = pd.DataFrame({
    'date': dates_for_specific_year,
    'Predicted Births': predictions_for_specific_year
})

# Create an interactive line plot using Plotly Express
fig = px.line(plot_data, x='date', y='Predicted Births',
              markers=True, title=f'Interactive Plot - Predicted Number of Births in {specific_year}')

# Add data labels to each point
fig.update_traces(text=plot_data['Predicted Births'].astype(int).tolist(), textposition='top center')

# Update layout for better visibility
fig.update_layout(xaxis_title='Date', yaxis_title='Number of Births', hovermode='closest')

# Show the plot
fig.show()


