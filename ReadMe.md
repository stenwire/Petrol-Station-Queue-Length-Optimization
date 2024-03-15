# Petrol Station Queue Length Optimization
> In partial fullfilment of MCT-504

This project aims to analyze and model wait times and queue lengths at petrol stations, with the ultimate goal of reducing queue lengths and improving customer experience. The code provided covers various aspects of data analysis, including data preprocessing, exploratory data analysis, correlation analysis, time series analysis, and modeling using both traditional statistical methods and deep learning techniques.

## Technologies
- Python
- Jupyter Notebook
- Kaggle
- Anthropic's Claude.ai

## Table of Contents

- [Data Loading and Preprocessing](#data-loading-and-preprocessing)
- [Time Series Analysis](#time-series-analysis)
- [Statistical Summary](#statistical-summary)
- [Waiting Time Distribution by Queue Length](#waiting-time-distribution-by-queue-length)
- [Correlation Analysis](#correlation-analysis)
- [Vector Autoregression (VAR) Model](#vector-autoregression-var-model)
- [LSTM Model for Wait Time Prediction](#lstm-model-for-wait-time-prediction)
- [LSTM Model for Time Series Forecasting](#lstm-model-for-time-series-forecasting)
- [How It Can Reduce Queue Length](#how-it-can-reduce-queue-length)

## Data Loading and Preprocessing

The code loads multiple CSV files from a directory and concatenates them into a single DataFrame. It converts date/time columns to datetime format and calculates the waiting time based on the arrival and start times. The code also handles missing values in the waiting time column by dropping rows with NaN values.

```python
# Lines 5-18: Loading and concatenating CSV files
# Lines 24-28: Converting date/time columns and calculating waiting time
# Lines 32-35: Handling missing values in the waiting time column
```

## Time Series Analysis

The code plots the queue length over time to identify patterns or trends. It also analyzes the variation of waiting times throughout the day using a boxplot.

```python
# Lines 43-49: Plotting queue length over time
# Lines 51-54: Analyzing waiting time variation throughout the day
```

## Statistical Summary

The code generates descriptive statistics for the numeric columns in the DataFrame.

```python
# Lines 61-63: Generating descriptive statistics
```

## Waiting Time Distribution by Queue Length

The code visualizes the distribution of waiting times for different queue lengths using a boxplot.

```python
# Lines 70-73: Visualizing waiting time distribution by queue length
```

## Correlation Analysis

The code explores correlations between different numeric columns in the dataset using a correlation matrix and heatmap. It also identifies highly correlated variable pairs based on a specified correlation threshold.

```python
# Lines 89-92: Plotting correlation heatmap
# Lines 96-100: Plotting filtered correlation matrix
# Lines 104-109: Identifying highly correlated variable pairs
```

## Vector Autoregression (VAR) Model

The code sets up the necessary data for fitting a VAR model, which is a multivariate time series model. It handles missing values and transforms datetime columns for compatibility with the VAR model.

```python
# Lines 116-118: Selecting columns for VAR model
# Lines 122-125: Train-test split
# Lines 129-130: Checking for missing values
# Lines 134-136: Transforming datetime columns
```

## LSTM Model for Wait Time Prediction

The code implements a Long Short-Term Memory (LSTM) neural network model to predict wait times. It scales the input features, creates sequences from the data, splits the data into training and testing sets, and trains the LSTM model. The model's performance is evaluated using mean squared error (MSE) on the test set, and the results are plotted.

```python
# Lines 154-177: Building and training LSTM model
# Lines 180-197: Making predictions and evaluating model performance
```

## LSTM Model for Time Series Forecasting

This section builds another LSTM model for time series forecasting of wait times. It creates sequences from the data, splits the data into training and testing sets, and trains the LSTM model. The model's performance is evaluated using root mean squared error (RMSE) on the test set, and the results are plotted. The code also includes plotting the training history and saving the best model.

```python
# Lines 202-238: Building, training, and evaluating LSTM model for time series forecasting
```

## How It Can Reduce Queue Length

By analyzing and modeling wait times and queue lengths at petrol stations, this project can provide valuable insights and predictions that can be used to optimize operations and reduce queue lengths. Here are some potential ways it can contribute to queue length reduction:

1. **Identifying Patterns and Trends**: The time series analysis and visualizations can help identify patterns and trends in queue lengths, allowing for better resource allocation and staffing during peak hours or periods of high demand.

2. **Correlation Analysis**: By understanding the correlations between different factors (e.g., time of day, day of the week, weather conditions, etc.) and queue lengths, petrol station operators can anticipate and prepare for situations that may lead to longer queues.

3. **Predictive Modeling**: The LSTM models and other predictive techniques can forecast future wait times and queue lengths based on historical data and other relevant factors. This information can be used to proactively implement measures to mitigate long queues, such as opening additional service lanes or temporarily increasing staff.

4. **Resource Optimization**: By analyzing the waiting time distribution by queue length, petrol stations can optimize their resource allocation and staffing levels to ensure efficient service and minimize customer wait times.

5. **Real-time Monitoring and Adjustment**: The models and analyses developed in this project can be integrated into real-time monitoring systems, allowing petrol stations to detect and respond to unexpected queue buildup promptly.

6. **Customer Communication and Guidance**: The insights gained from this project can be used to provide customers with real-time information about queue lengths and estimated wait times, allowing them to make informed decisions about which petrol station to visit or when to refuel.

#### By leveraging the analyses and models provided in this project, petrol station operators can gain a comprehensive understanding of the factors influencing queue lengths and develop strategies to optimize their operations, enhance customer satisfaction, and ultimately reduce queue lengths.

### Reference
> 🤗 Please checkout the original code and dataset source here: https://www.kaggle.com/datasets/sanjeebtiwary/queue-waiting-time-prediction/code

### Honourable mentions:

- Sanjeeb: [Github Profile](https://github.com/sanjeebtiwary)
- Ogochukwu(Project Partner)
- Engr Timothy(Project Supervisor): [LinkedIn](https://www.linkedin.com/in/engr-araoye-timothy-a53bb8120)