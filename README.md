# Financial Advisory with Generative AI

This project aims to revolutionize financial advisory services using generative AI to provide personalized, data-driven financial advice to customers. The objective is to analyze customer financial data and market trends to generate tailored investment strategies and offer real-time advisory services that adapt to changing financial conditions and customer goals. This project ensures transparency and explainability in the AI-driven advisory process to build customer trust and confidence.

## Table of Contents

- [Objective](#objective)
- [Challenge](#challenge)
- [Solution Overview](#solution-overview)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Scalability](#scalability)
- [Comparison with Alternatives](#comparison-with-alternatives)
- [Azure Tools and Resources](#azure-tools-and-resources)
- [Implementation](#implementation)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Feature Engineering](#feature-engineering)
  - [Model Training and Evaluation](#model-training-and-evaluation)
  - [Forecasting](#forecasting)
- [Business Applications](#business-applications)
- [Unique Aspects](#unique-aspects)
- [User Experience Enhancement](#user-experience-enhancement)
- [Security Measures](#security-measures)
- [How to Run the Project](#how-to-run-the-project)
- [Contributing](#contributing)
- [License](#license)

## Objective

To revolutionize financial advisory services using generative AI to provide personalized, data-driven financial advice to customers.

## Challenge

1. Analyze customer financial data and market trends to generate tailored investment strategies.
2. Offer real-time advisory services that adapt to changing financial conditions and customer goals.
3. Ensure transparency and explainability in the AI-driven advisory process to build customer trust and confidence.

## Solution Overview

This project leverages machine learning models to predict stock prices using historical stock market data. The models used include Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Support Vector Regressor. The best-performing model is then used for forecasting future stock prices.

## Methodology

1. **Data Collection**: Historical stock data is collected and preprocessed.
2. **Exploratory Data Analysis (EDA)**: Various graphs and visualizations are used to understand the data.
3. **Feature Engineering**: New features are created to improve model performance.
4. **Model Training and Evaluation**: Multiple machine learning models are trained and evaluated.
5. **Forecasting**: The best-performing model is used to forecast future stock prices.

## Architecture

The project architecture includes data preprocessing, exploratory data analysis, feature engineering, model training and evaluation, and forecasting. 

## Scalability

The solution is designed to be scalable to accommodate growth without compromising performance. This is achieved through efficient data processing techniques and the use of robust machine learning models.

## Comparison with Alternatives

Compared to traditional financial advisory services, this AI-driven approach provides personalized, real-time financial advice. Existing alternatives include human advisors and rule-based automated advisors. Our solution leverages generative AI for more accurate and adaptive advice.

## Azure Tools and Resources

If the idea gets selected, the following Azure tools and resources will be used:
- Azure Machine Learning
- Azure Storage
- Azure Databricks
- Azure Cognitive Services

## Implementation

### Data Preprocessing

The data is preprocessed to ensure it is in the right format for analysis and modeling.

```python
import pandas as pd

# Load the data
df = pd.read_csv('ADANIPORTS.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)
```

### Exploratory Data Analysis

EDA is performed using various plots and visualizations.

```python
import matplotlib.pyplot as plt

# Plot Closing Prices Over Time
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price')
plt.title('Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
```

### Feature Engineering

New features are created to improve model performance.

```python
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()
df['Volatility'] = df['Close'].rolling(window=10).std()
df['Returns'] = df['Close'].pct_change()
df['Lag1'] = df['Close'].shift(1)
df['Lag2'] = df['Close'].shift(2)
df['Lag3'] = df['Close'].shift(3)
df.dropna(inplace=True)
```

### Model Training and Evaluation

Multiple machine learning models are trained and evaluated.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np

# Define features and target variable
features = ['MA10', 'MA50', 'Volatility', 'Returns', 'Lag1', 'Lag2', 'Lag3', 'Volume']
X = df[features]
y = df['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Support Vector Regressor': SVR(kernel='rbf')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'{name} RMSE: {rmse}')
```

### Forecasting

The best-performing model (Linear Regression) is used to forecast future stock prices.

```python
# Define the number of days to forecast
forecast_days = 30

# Create a DataFrame to hold the forecasted values
forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_days + 1, inclusive='right')
forecast_df = pd.DataFrame(index=forecast_dates, columns=df.columns)

# Initialize the last known values
last_known_values = df.iloc[-1].copy()

# Perform the forecasting
for date in forecast_dates:
    input_data = last_known_values[features].values

    # Ensure no NaN values in input_data
    if pd.isnull(input_data).any():
        continue

    input_data = input_data.reshape(1, -1)
        
    forecasted_close = lr_model.predict(input_data)[0]
    
    # Update the forecast_df with the forecasted values
    forecast_df.at[date, 'Close'] = forecasted_close
    forecast_df.at[date, 'Lag1'] = last_known_values['Close']
    forecast_df.at[date, 'Lag2'] = last_known_values['Lag1']
    forecast_df.at[date, 'Lag3'] = last_known_values['Lag2']
    forecast_df.at[date, 'Volume'] = last_known_values['Volume']
    
    # Calculate new features
    forecast_df['MA10'] = forecast_df['Close'].rolling(window=10).mean()
    forecast_df['MA50'] = forecast_df['Close'].rolling(window=50).mean()
    forecast_df['Volatility'] = forecast_df['Close'].rolling(window=10).std()
    forecast_df['Returns'] = forecast_df['Close'].pct_change()
    
    # Update last known values
    last_known_values = forecast_df.loc[date].copy()

# Concatenate the original and forecasted data
full_df = pd.concat([df, forecast_df])

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(full_df.index, full_df['Close'], label='Actual and Forecasted Closing Prices')
plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Forecast Start')
plt.title('Stock Price Forecast using Linear Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
```

## Business Applications

The solution can be used in various financial advisory services, including:
- Personalized investment strategies
- Real-time market analysis and recommendations
- Risk management and portfolio optimization

## Unique Aspects

The unique aspects of this solution include:
- Personalized, data-driven financial advice
- Real-time adaptation to market conditions
- Transparency and explainability of AI-driven decisions

## User Experience Enhancement

The solution enhances user experience by providing clear, actionable financial advice and maintaining transparency in the decision-making process.

## Security Measures

Measures incorporated to ensure the security and integrity of the solution include:
- Secure data storage and transmission
- Regular security audits and compliance checks
- Data anonymization and encryption

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-advisory.git
   ```
2. Navigate to the project directory:
   ```bash
   cd financial-advisory
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Jupyter notebook to execute the code:
   ```bash
   jupyter notebook
   ```

## Contributing

Contributions are welcome! Please read

 the [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
