```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, anderson
import seaborn as sns
from datetime import timedelta
from IPython.display import display
```


```python
# Preprocessing with Dask
df = pd.read_csv('../input/power_data.csv')
```


```python
# Convert 'date' column to datetime for easy manipulation
df['date'] = pd.to_datetime(df['date'])
```


```python
# Add a 'year' column to simplify annual analysis
df['year'] = df['date'].dt.year
```


```python
# Extract additional time-based features
def extract_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds additional time-based features to the DataFrame. Assumes 'date' column is present.
    """
    features = ['hour', 'day_of_week', 'month', 'week_of_year', 'time_delta']
    if all(f in data.columns for f in features):
        return data  # Avoid duplicating work if features already exist

    data['hour'] = data['date'].dt.hour
    data['day_of_week'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['week_of_year'] = data['date'].dt.isocalendar().week
    data['time_delta'] = data['date'].diff().dt.total_seconds().fillna(0)
    return data
```


```python
# Split the DataFrame by year
def split_by_year(data: pd.DataFrame):
    """
    Splits the DataFrame by year and returns a dictionary {year: df_for_year}.
    """
    return {yr: group for yr, group in data.groupby(data['year'])}
```


```python
# Generic method to plot price action
def plot_price_action(data: pd.DataFrame, year: int = None):
    """
    Plots the 'close' price for the given DataFrame.

    If a year is provided, the plot focuses on that year's input.
    If no year is provided, the plot spans the entire period in the DataFrame.
    """
    title = (
        f"Price Action for the Year {year}" if year
        else f"Price Action for the Period {data['date'].min().strftime('%m/%Y')} - {data['date'].max().strftime('%m/%Y')}"
    )

    # Filter by year if provided
    filtered_data = data[data['year'] == year] if year else data

    # Plotting
    plt.figure(figsize=(21, 7))
    plt.plot(filtered_data['date'], filtered_data['close'], label='Close Price', color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Clear the plot after display
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure completely to free memory
```


```python
# Analyze input symmetry, skewness, and outliers
def analyze_data_distribution(data, columns=None):
    """
    Analyze input symmetry, skewness, and outliers for specified columns.

    Args:
    input (pd.DataFrame): The DataFrame containing the input.
    columns (list, optional): List of column names to analyze. Analyzes all numeric columns by default.

    Returns:
    pd.DataFrame: Summary of symmetry, skewness, and outlier presence for each column.
    """
    columns = columns or data.select_dtypes(include=[np.number]).columns.tolist()
    summary = []

    for column in columns:
        col_data = data[column].dropna()  # Remove NaN values for analysis

        # Calculate metrics
        mean, median, std = col_data.mean(), col_data.median(), col_data.std()
        skewness = col_data.skew()
        symmetry = "Symmetric" if np.isclose(mean, median, atol=std * 0.1) else "Skewed"
        skew_type = "Right-Skewed" if skewness > 0 else "Left-Skewed" if skewness < 0 else "Symmetric"

        # Outliers using IQR
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)].count()

        summary.append({
            "Column": column,
            "Mean": mean,
            "Median": median,
            "Skewness": skewness,
            "Skew Type": skew_type,
            "Symmetry": symmetry,
            "Outliers Count": outliers,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound
        })

    return pd.DataFrame(summary)
```


```python
# Handle missing values
def handle_missing_values(data, threshold=5, fill_method='median', rolling_window_days=90, date_column='date'):
    """
    Handles missing values in a DataFrame based on a specified threshold and filling method.

    Args:
    input (pd.DataFrame): The DataFrame to process.
    threshold (float): Percentage threshold to decide whether to fill or drop columns.
    fill_method (str): Method to fill missing values ('mean', 'median', or 'rolling_median').
    rolling_window_days (int): Rolling window size in days (used only if fill_method='rolling_median').
    date_column (str): The column name containing datetime information.

    Returns:
    pd.DataFrame: Processed DataFrame with missing values handled.
    """
    total_rows = len(data)
    for column in data.columns:
        missing_count = data[column].isna().sum()
        missing_percentage = (missing_count / total_rows) * 100

        if missing_percentage == 0:
            continue  # No missing values to handle

        print(f"Column: {column}, Missing Percentage: {missing_percentage:.2f}%")

        if missing_percentage < threshold:
            if fill_method == 'mean':
                fill_value = data[column].mean()
                data[column] = data[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with mean ({fill_value:.2f})")
            elif fill_method == 'median':
                fill_value = data[column].median()
                data[column] = data[column].fillna(fill_value)
                print(f"Filled missing values in '{column}' with median ({fill_value:.2f})")
            elif fill_method == 'rolling_median':
                # Use rolling median strategy
                data[column] = fill_missing_with_rolling_median(data, date_column, column, rolling_window_days)
                print(f"Filled missing values in '{column}' with rolling median over {rolling_window_days} days")
            else:
                raise ValueError("Invalid fill method. Choose 'mean', 'median', or 'rolling_median'.")
        else:
            # Drop the column if missing percentage is above the threshold
            data.drop(columns=[column], inplace=True)
            print(f"Dropped column '{column}' due to high missing percentage ({missing_percentage:.2f}%)")

    return data
```


```python
# Specialized function for rolling median
def fill_missing_with_rolling_median(data, date_column, target_column, rolling_window_days):
    """
    Fills missing values in a target column using the median of a rolling window.

    Args:
    input (pd.DataFrame): The DataFrame containing the input.
    date_column (str): The column name for the datetime information.
    target_column (str): The column with missing values to fill.
    rolling_window_days (int): Number of days for the rolling window.

    Returns:
    pd.Series: A series with filled missing values.
    """
    filled_column = data[target_column].copy()

    for index, row in data.iterrows():
        if pd.isna(row[target_column]):
            # Define the rolling window timeframe
            start_date = row[date_column] - timedelta(days=rolling_window_days)
            end_date = row[date_column]

            # Filter the input to the rolling window
            window_data = data[
                (data[date_column] >= start_date) &
                (data[date_column] <= end_date)
            ][target_column].dropna()

            # Compute the median and fill the missing value
            if not window_data.empty:
                filled_column.at[index] = window_data.median()

    return filled_column
```


```python
# Print the price trend for the entire dataset or a specific year
def print_price_trend(data: pd.DataFrame, year: int = None):
    """
    Prints the price trend for the entire dataset or a specific year.

    Args:
    input (pd.DataFrame): The dataset containing 'date' and 'close' columns.
    year (int, optional): The specific year to analyze. Defaults to None.
    """
    # Filter input for the specified year if provided
    filtered_data = data[data['year'] == year] if year else data

    # Calculate starting and ending prices
    start_price = filtered_data['close'].iloc[0]
    end_price = filtered_data['close'].iloc[-1]

    # Determine the trend
    trend = "increased" if end_price > start_price else "decreased" if end_price < start_price else "remained the same"
    percentage_change = ((end_price - start_price) / start_price) * 100

    # Print the trend
    if year:
        print(f"Price trend for the year {year}:")
    else:
        print("Price trend for the entire dataset:")

    print(f"Start Price: {start_price:.2f}")
    print(f"End Price: {end_price:.2f}")
    print(f"Trend: The price {trend} by {percentage_change:.2f}%.\n")
```


```python
def perform_normality_tests(data, columns=None, shapiro_sample_size=5000):
    """
    Perform normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
    for specified columns in a DataFrame.

    Args:
    input (pd.DataFrame): The DataFrame containing the input.
    columns (list, optional): List of column names to test. Defaults to all numeric columns.
    shapiro_sample_size (int): Maximum sample size for the Shapiro-Wilk test.

    Returns:
    pd.DataFrame: Summary of normality test results.
    """
    columns = columns or data.select_dtypes(include=[np.number]).columns
    results = []

    for column in columns:
        col_data = data[column].dropna()  # Drop missing values

        # Skip columns with insufficient input or zero range
        if len(col_data) < 3:
            print(f"Skipping column '{column}' due to insufficient input.")
            continue

        if col_data.max() - col_data.min() == 0:
            print(f"Skipping column '{column}' due to zero range (all values are identical).")
            continue

        # Subsample for Shapiro-Wilk
        shapiro_data = col_data.sample(min(len(col_data), shapiro_sample_size), random_state=42)

        # Check range for the subsample
        if shapiro_data.max() - shapiro_data.min() == 0:
            print(f"Skipping column '{column}' (Subsample has zero range).")
            continue

        # Shapiro-Wilk Test
        shapiro_stat, shapiro_p = shapiro(shapiro_data)
        shapiro_result = "Normal" if shapiro_p > 0.05 else "Not Normal"

        # Kolmogorov-Smirnov Test
        ks_stat, ks_p = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
        ks_result = "Normal" if ks_p > 0.05 else "Not Normal"

        # Anderson-Darling Test
        ad_result = anderson(col_data, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance level
        ad_result_final = "Normal" if ad_stat < ad_critical else "Not Normal"

        # Append results
        results.append({
            "Column": column,
            "Shapiro-Wilk": shapiro_result,
            "Shapiro p-value": shapiro_p,
            "Kolmogorov-Smirnov": ks_result,
            "KS p-value": ks_p,
            "Anderson-Darling": ad_result_final,
            "AD Statistic": ad_stat,
            "AD Critical Value (5%)": ad_critical
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df
```


```python
# Check for missing values
print("Missing values before dropping rows:")
print(df.isna().sum())
```

    Missing values before dropping rows:
    date             0
    symbol           0
    open             0
    high             0
    low              0
    close            0
    Volume USDT      0
    tradecount       0
    token            0
    hour             0
    day              0
    ema_5            0
    ema_15           0
    ema_30           0
    ema_60           0
    ema_100          0
    ema_200          0
    WMA             13
    MACD            25
    MACD_Signal     33
    MACD_Hist       33
    ATR             14
    HMA             11
    KAMA             9
    CMO             14
    Z-Score        154
    QStick           9
    year             0
    dtype: int64



```python
# Analyze the distribution for numeric columns
distribution_summary = analyze_data_distribution(df)

# Display the summary
print("Distribution Analysis Summary (Entire Dataset):")
display(distribution_summary)
```

    Distribution Analysis Summary (Entire Dataset):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Skewness</th>
      <th>Skew Type</th>
      <th>Symmetry</th>
      <th>Outliers Count</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>2.834777e+04</td>
      <td>2.675128e+04</td>
      <td>0.491224</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713382e+04</td>
      <td>7.318862e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>2.836410e+04</td>
      <td>2.675846e+04</td>
      <td>0.491263</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.716511e+04</td>
      <td>7.325628e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>2.833144e+04</td>
      <td>2.674433e+04</td>
      <td>0.491139</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.710287e+04</td>
      <td>7.312216e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>2.834778e+04</td>
      <td>2.675128e+04</td>
      <td>0.491224</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713381e+04</td>
      <td>7.318861e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>1.943680e+06</td>
      <td>1.084630e+06</td>
      <td>7.597379</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>161625</td>
      <td>-2.120176e+06</td>
      <td>4.872645e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>1.510783e+03</td>
      <td>7.650000e+02</td>
      <td>4.900996</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>196526</td>
      <td>-1.573500e+03</td>
      <td>3.702500e+03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>1.150389e+01</td>
      <td>1.200000e+01</td>
      <td>-0.000802</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-1.200000e+01</td>
      <td>3.600000e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>2.834779e+04</td>
      <td>2.675206e+04</td>
      <td>0.491212</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713296e+04</td>
      <td>7.318649e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>2.834782e+04</td>
      <td>2.675294e+04</td>
      <td>0.491189</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713484e+04</td>
      <td>7.318890e+04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>2.834787e+04</td>
      <td>2.675294e+04</td>
      <td>0.491156</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713815e+04</td>
      <td>7.319366e+04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>2.834796e+04</td>
      <td>2.675422e+04</td>
      <td>0.491091</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713194e+04</td>
      <td>7.318413e+04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>2.834808e+04</td>
      <td>2.675485e+04</td>
      <td>0.491007</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713316e+04</td>
      <td>7.318891e+04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>2.834840e+04</td>
      <td>2.676068e+04</td>
      <td>0.490797</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.707357e+04</td>
      <td>7.310329e+04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>2.834780e+04</td>
      <td>2.675213e+04</td>
      <td>0.491205</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713385e+04</td>
      <td>7.318754e+04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>-4.347376e-02</td>
      <td>-8.477548e-02</td>
      <td>57.897439</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>330029</td>
      <td>-3.426663e+01</td>
      <td>3.392145e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>-1.626428e-05</td>
      <td>2.694116e-04</td>
      <td>61.456090</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>321188</td>
      <td>-1.086615e+01</td>
      <td>1.086747e+01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>-4.349390e-02</td>
      <td>-1.015021e-01</td>
      <td>54.602224</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>330974</td>
      <td>-3.212150e+01</td>
      <td>3.173856e+01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>4.093417e+01</td>
      <td>2.515166e+01</td>
      <td>4.717802</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>86365</td>
      <td>-5.865867e+01</td>
      <td>1.274915e+02</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>2.834778e+04</td>
      <td>2.675102e+04</td>
      <td>0.491233</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713570e+04</td>
      <td>7.319160e+04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>2.834719e+04</td>
      <td>2.675157e+04</td>
      <td>0.491076</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-1.713456e+04</td>
      <td>7.318939e+04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>-4.089469e-01</td>
      <td>-1.232780e-01</td>
      <td>-0.046832</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>4456</td>
      <td>-6.437943e+01</td>
      <td>6.372217e+01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>-1.446761e-02</td>
      <td>-8.871890e-03</td>
      <td>-0.015695</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>209</td>
      <td>-4.247895e+00</td>
      <td>4.224303e+00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>1.115379e-02</td>
      <td>9.000000e-03</td>
      <td>-0.635869</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>324431</td>
      <td>-9.660500e+00</td>
      <td>9.695500e+00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>2.021420e+03</td>
      <td>2.021000e+03</td>
      <td>0.081154</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>2.017000e+03</td>
      <td>2.025000e+03</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Process the dataset for statistical analysis
df = extract_time_features(df)
df = handle_missing_values(df)
```

    Column: WMA, Missing Percentage: 0.00%
    Filled missing values in 'WMA' with median (26752.13)
    Column: MACD, Missing Percentage: 0.00%
    Filled missing values in 'MACD' with median (-0.08)
    Column: MACD_Signal, Missing Percentage: 0.00%
    Filled missing values in 'MACD_Signal' with median (0.00)
    Column: MACD_Hist, Missing Percentage: 0.00%
    Filled missing values in 'MACD_Hist' with median (-0.10)
    Column: ATR, Missing Percentage: 0.00%
    Filled missing values in 'ATR' with median (25.15)
    Column: HMA, Missing Percentage: 0.00%
    Filled missing values in 'HMA' with median (26751.02)
    Column: KAMA, Missing Percentage: 0.00%
    Filled missing values in 'KAMA' with median (26751.57)
    Column: CMO, Missing Percentage: 0.00%
    Filled missing values in 'CMO' with median (-0.12)
    Column: Z-Score, Missing Percentage: 0.01%
    Filled missing values in 'Z-Score' with median (-0.01)
    Column: QStick, Missing Percentage: 0.00%
    Filled missing values in 'QStick' with median (0.01)



```python
# Verify results
print("\nProcessed Data Summary:")
print(df.info())
```

    
    Processed Data Summary:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1997210 entries, 0 to 1997209
    Data columns (total 32 columns):
     #   Column        Dtype         
    ---  ------        -----         
     0   date          datetime64[ns]
     1   symbol        object        
     2   open          float64       
     3   high          float64       
     4   low           float64       
     5   close         float64       
     6   Volume USDT   int64         
     7   tradecount    int64         
     8   token         object        
     9   hour          int32         
     10  day           object        
     11  ema_5         float64       
     12  ema_15        float64       
     13  ema_30        float64       
     14  ema_60        float64       
     15  ema_100       float64       
     16  ema_200       float64       
     17  WMA           float64       
     18  MACD          float64       
     19  MACD_Signal   float64       
     20  MACD_Hist     float64       
     21  ATR           float64       
     22  HMA           float64       
     23  KAMA          float64       
     24  CMO           float64       
     25  Z-Score       float64       
     26  QStick        float64       
     27  year          int32         
     28  day_of_week   int32         
     29  month         int32         
     30  week_of_year  UInt32        
     31  time_delta    float64       
    dtypes: UInt32(1), datetime64[ns](1), float64(21), int32(4), int64(2), object(3)
    memory usage: 451.4+ MB
    None



```python
# Split the dataset by year and analyze each year's input
yearly_data = split_by_year(df)
for year, data in yearly_data.items():
    print(f"\nDistribution Analysis Summary (Year {year}):")
    yearly_summary = analyze_data_distribution(data)
    display(yearly_summary)  # Use display for better formatting
```

    
    Distribution Analysis Summary (Year 2020):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Skewness</th>
      <th>Skew Type</th>
      <th>Symmetry</th>
      <th>Outliers Count</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>1.106489e+04</td>
      <td>9692.440000</td>
      <td>1.792518</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71844</td>
      <td>4.698100e+03</td>
      <td>1.579122e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>1.107059e+04</td>
      <td>9696.130000</td>
      <td>1.794272</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71863</td>
      <td>4.705435e+03</td>
      <td>1.579276e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>1.105896e+04</td>
      <td>9688.550000</td>
      <td>1.790685</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71811</td>
      <td>4.692450e+03</td>
      <td>1.578773e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>1.106493e+04</td>
      <td>9692.420000</td>
      <td>1.792549</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71847</td>
      <td>4.698145e+03</td>
      <td>1.579115e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>1.356381e+06</td>
      <td>712573.000000</td>
      <td>13.255836</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>46227</td>
      <td>-1.250797e+06</td>
      <td>3.023739e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>5.806934e+02</td>
      <td>379.000000</td>
      <td>7.397680</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>36420</td>
      <td>-5.410000e+02</td>
      <td>1.435000e+03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>1.149996e+01</td>
      <td>11.000000</td>
      <td>0.000272</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-1.450000e+01</td>
      <td>3.750000e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>1.106501e+04</td>
      <td>9692.536258</td>
      <td>1.792598</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71848</td>
      <td>4.698677e+03</td>
      <td>1.579076e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>1.106522e+04</td>
      <td>9692.508614</td>
      <td>1.792725</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71874</td>
      <td>4.699420e+03</td>
      <td>1.578937e+04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>1.106553e+04</td>
      <td>9692.381077</td>
      <td>1.792918</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71866</td>
      <td>4.700083e+03</td>
      <td>1.578853e+04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>1.106615e+04</td>
      <td>9691.910614</td>
      <td>1.793312</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71850</td>
      <td>4.698396e+03</td>
      <td>1.579129e+04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>1.106698e+04</td>
      <td>9691.647889</td>
      <td>1.793849</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71817</td>
      <td>4.696536e+03</td>
      <td>1.579544e+04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>1.106905e+04</td>
      <td>9691.869212</td>
      <td>1.795200</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71723</td>
      <td>4.690464e+03</td>
      <td>1.580244e+04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>1.106506e+04</td>
      <td>9692.563429</td>
      <td>1.792367</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71851</td>
      <td>4.698468e+03</td>
      <td>1.579055e+04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>-2.891694e-01</td>
      <td>-0.122835</td>
      <td>1.176383</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>56165</td>
      <td>-1.516777e+01</td>
      <td>1.469405e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>-6.079800e-05</td>
      <td>-0.005909</td>
      <td>1.563933</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>50146</td>
      <td>-4.924650e+00</td>
      <td>4.905009e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>-2.892483e-01</td>
      <td>-0.121606</td>
      <td>1.151555</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>56763</td>
      <td>-1.414546e+01</td>
      <td>1.366548e+01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>1.467182e+01</td>
      <td>10.224262</td>
      <td>4.610774</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>42650</td>
      <td>-8.950260e+00</td>
      <td>3.277382e+01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>1.106490e+04</td>
      <td>9692.599091</td>
      <td>1.792313</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71846</td>
      <td>4.698479e+03</td>
      <td>1.579114e+04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>1.106401e+04</td>
      <td>9691.970879</td>
      <td>1.790861</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>71797</td>
      <td>4.698946e+03</td>
      <td>1.578970e+04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>-9.592608e-01</td>
      <td>-0.502966</td>
      <td>-0.082216</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>523</td>
      <td>-6.267081e+01</td>
      <td>6.100830e+01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>-3.720379e-02</td>
      <td>-0.035587</td>
      <td>-0.011720</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>80</td>
      <td>-4.223327e+00</td>
      <td>4.154789e+00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>4.324322e-02</td>
      <td>0.028000</td>
      <td>-2.038058</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>52043</td>
      <td>-4.364000e+00</td>
      <td>4.444000e+00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>2.020000e+03</td>
      <td>2020.000000</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>2.020000e+03</td>
      <td>2.020000e+03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>2.998806e+00</td>
      <td>3.000000</td>
      <td>0.003311</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>month</td>
      <td>6.510140e+00</td>
      <td>7.000000</td>
      <td>-0.005217</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.900000e+01</td>
    </tr>
    <tr>
      <th>26</th>
      <td>week_of_year</td>
      <td>2.691269e+01</td>
      <td>27.000000</td>
      <td>0.000525</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-2.500000e+01</td>
      <td>7.900000e+01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>time_delta</td>
      <td>6.003976e+01</td>
      <td>60.000000</td>
      <td>620.569238</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>4</td>
      <td>6.000000e+01</td>
      <td>6.000000e+01</td>
    </tr>
  </tbody>
</table>
</div>


    
    Distribution Analysis Summary (Year 2021):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Skewness</th>
      <th>Skew Type</th>
      <th>Symmetry</th>
      <th>Outliers Count</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>4.735781e+04</td>
      <td>4.789814e+04</td>
      <td>-0.044405</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.083860e+04</td>
      <td>8.342458e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>4.739365e+04</td>
      <td>4.792999e+04</td>
      <td>-0.044783</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.087829e+04</td>
      <td>8.346951e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>4.732203e+04</td>
      <td>4.786696e+04</td>
      <td>-0.044059</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.079332e+04</td>
      <td>8.338442e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>4.735784e+04</td>
      <td>4.789814e+04</td>
      <td>-0.044405</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.083838e+04</td>
      <td>8.342477e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>2.199850e+06</td>
      <td>1.449567e+06</td>
      <td>7.083396</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>40790</td>
      <td>-1.621944e+06</td>
      <td>5.002266e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>1.278538e+03</td>
      <td>9.840000e+02</td>
      <td>7.274845</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>32985</td>
      <td>-5.730000e+02</td>
      <td>2.731000e+03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>1.151315e+01</td>
      <td>1.200000e+01</td>
      <td>-0.003343</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-1.200000e+01</td>
      <td>3.600000e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>4.735776e+04</td>
      <td>4.789932e+04</td>
      <td>-0.044412</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.083729e+04</td>
      <td>8.342501e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>4.735755e+04</td>
      <td>4.789771e+04</td>
      <td>-0.044499</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.084066e+04</td>
      <td>8.341767e+04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>4.735724e+04</td>
      <td>4.789764e+04</td>
      <td>-0.044639</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>1</td>
      <td>1.085172e+04</td>
      <td>8.340988e+04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>4.735662e+04</td>
      <td>4.789744e+04</td>
      <td>-0.044931</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>2</td>
      <td>1.084811e+04</td>
      <td>8.342377e+04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>4.735578e+04</td>
      <td>4.789966e+04</td>
      <td>-0.045333</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>4</td>
      <td>1.083034e+04</td>
      <td>8.344623e+04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>4.735369e+04</td>
      <td>4.790784e+04</td>
      <td>-0.046380</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>9</td>
      <td>1.084468e+04</td>
      <td>8.343747e+04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>4.735766e+04</td>
      <td>4.789963e+04</td>
      <td>-0.044509</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.083411e+04</td>
      <td>8.342600e+04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>2.894932e-01</td>
      <td>-2.390972e-02</td>
      <td>49.378628</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>31895</td>
      <td>-1.133346e+02</td>
      <td>1.132487e+02</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>1.189720e-04</td>
      <td>9.784440e-02</td>
      <td>51.972418</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>29476</td>
      <td>-3.599530e+01</td>
      <td>3.597586e+01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>2.893742e-01</td>
      <td>-2.299536e-01</td>
      <td>46.613911</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>32301</td>
      <td>-1.060548e+02</td>
      <td>1.059108e+02</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>8.898989e+01</td>
      <td>7.647123e+01</td>
      <td>6.081946</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>27089</td>
      <td>-1.411620e+01</td>
      <td>1.761430e+02</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>4.735783e+04</td>
      <td>4.789901e+04</td>
      <td>-0.044461</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.083822e+04</td>
      <td>8.342478e+04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>4.735646e+04</td>
      <td>4.789872e+04</td>
      <td>-0.044403</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>1.084159e+04</td>
      <td>8.340830e+04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>-3.009098e-01</td>
      <td>1.692479e-01</td>
      <td>-0.082382</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>938</td>
      <td>-6.256087e+01</td>
      <td>6.232654e+01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>-6.288335e-03</td>
      <td>1.279855e-02</td>
      <td>-0.041874</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>31</td>
      <td>-4.255608e+00</td>
      <td>4.254223e+00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>3.078918e-02</td>
      <td>-8.800000e-02</td>
      <td>-0.505177</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>31916</td>
      <td>-3.212600e+01</td>
      <td>3.211400e+01</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>2.021000e+03</td>
      <td>2.021000e+03</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>2.021000e+03</td>
      <td>2.021000e+03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>3.001064e+00</td>
      <td>3.000000e+00</td>
      <td>-0.003007</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>month</td>
      <td>6.528079e+00</td>
      <td>7.000000e+00</td>
      <td>-0.011524</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.900000e+01</td>
    </tr>
    <tr>
      <th>26</th>
      <td>week_of_year</td>
      <td>2.658732e+01</td>
      <td>2.700000e+01</td>
      <td>0.001577</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-2.500000e+01</td>
      <td>7.900000e+01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>time_delta</td>
      <td>6.011357e+01</td>
      <td>6.000000e+01</td>
      <td>382.265210</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>6</td>
      <td>6.000000e+01</td>
      <td>6.000000e+01</td>
    </tr>
  </tbody>
</table>
</div>


    
    Distribution Analysis Summary (Year 2022):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Skewness</th>
      <th>Skew Type</th>
      <th>Symmetry</th>
      <th>Outliers Count</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>2.822646e+04</td>
      <td>2.314820e+04</td>
      <td>0.445932</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.712028e+03</td>
      <td>6.824451e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>2.824132e+04</td>
      <td>2.316019e+04</td>
      <td>0.445486</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.717170e+03</td>
      <td>6.827899e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>2.821182e+04</td>
      <td>2.313561e+04</td>
      <td>0.446356</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.707934e+03</td>
      <td>6.821396e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>2.822640e+04</td>
      <td>2.314795e+04</td>
      <td>0.445938</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.712107e+03</td>
      <td>6.824451e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>2.281915e+06</td>
      <td>1.427310e+06</td>
      <td>5.800340</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>37603</td>
      <td>-2.464308e+06</td>
      <td>5.934434e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>2.284043e+03</td>
      <td>1.563000e+03</td>
      <td>3.508150</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>22759</td>
      <td>-3.215500e+03</td>
      <td>6.956500e+03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>1.150000e+01</td>
      <td>1.150000e+01</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-1.150000e+01</td>
      <td>3.450000e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>2.822634e+04</td>
      <td>2.314866e+04</td>
      <td>0.445943</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.710992e+03</td>
      <td>6.824252e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>2.822617e+04</td>
      <td>2.314816e+04</td>
      <td>0.445957</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.713436e+03</td>
      <td>6.824666e+04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>2.822592e+04</td>
      <td>2.314786e+04</td>
      <td>0.445978</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.713956e+03</td>
      <td>6.824823e+04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>2.822542e+04</td>
      <td>2.314977e+04</td>
      <td>0.446021</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.711278e+03</td>
      <td>6.824248e+04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>2.822475e+04</td>
      <td>2.315017e+04</td>
      <td>0.446075</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.684626e+03</td>
      <td>6.820726e+04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>2.822308e+04</td>
      <td>2.314013e+04</td>
      <td>0.446200</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.667056e+03</td>
      <td>6.818455e+04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>2.822626e+04</td>
      <td>2.314837e+04</td>
      <td>0.445949</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.710553e+03</td>
      <td>6.824172e+04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>2.332804e-01</td>
      <td>-9.909277e-03</td>
      <td>-22.664090</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>57177</td>
      <td>-4.024468e+01</td>
      <td>4.054080e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>-2.802538e-04</td>
      <td>-1.678378e-02</td>
      <td>-25.322445</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>53721</td>
      <td>-1.271963e+01</td>
      <td>1.268855e+01</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>2.335607e-01</td>
      <td>-1.747784e-02</td>
      <td>-21.227584</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>57431</td>
      <td>-3.780667e+01</td>
      <td>3.808748e+01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>3.684159e+01</td>
      <td>2.955508e+01</td>
      <td>2.801710</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>26101</td>
      <td>-2.609050e+01</td>
      <td>9.091648e+01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>2.822639e+04</td>
      <td>2.314860e+04</td>
      <td>0.445937</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.711835e+03</td>
      <td>6.824416e+04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>2.822652e+04</td>
      <td>2.314931e+04</td>
      <td>0.445873</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-9.711077e+03</td>
      <td>6.824455e+04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>7.976136e-02</td>
      <td>5.161427e-02</td>
      <td>-0.000999</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>514</td>
      <td>-6.405795e+01</td>
      <td>6.416107e+01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>-1.592496e-03</td>
      <td>-1.349821e-04</td>
      <td>-0.002148</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>41</td>
      <td>-4.234739e+00</td>
      <td>4.230825e+00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>-5.856862e-02</td>
      <td>7.000000e-03</td>
      <td>0.400925</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>55525</td>
      <td>-1.133538e+01</td>
      <td>1.128963e+01</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>2.022000e+03</td>
      <td>2.022000e+03</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>2.022000e+03</td>
      <td>2.022000e+03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>3.005479e+00</td>
      <td>3.000000e+00</td>
      <td>-0.005479</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>month</td>
      <td>6.526027e+00</td>
      <td>7.000000e+00</td>
      <td>-0.010456</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.900000e+01</td>
    </tr>
    <tr>
      <th>26</th>
      <td>week_of_year</td>
      <td>2.656986e+01</td>
      <td>2.700000e+01</td>
      <td>-0.000594</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-2.500000e+01</td>
      <td>7.900000e+01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>time_delta</td>
      <td>6.000000e+01</td>
      <td>6.000000e+01</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>6.000000e+01</td>
      <td>6.000000e+01</td>
    </tr>
  </tbody>
</table>
</div>


    
    Distribution Analysis Summary (Year 2023):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Skewness</th>
      <th>Skew Type</th>
      <th>Symmetry</th>
      <th>Outliers Count</th>
      <th>Lower Bound</th>
      <th>Upper Bound</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>2.642933e+04</td>
      <td>27003.235000</td>
      <td>-1.072032</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16898</td>
      <td>1.851178e+04</td>
      <td>3.525594e+04</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>2.643648e+04</td>
      <td>27010.005000</td>
      <td>-1.073062</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16897</td>
      <td>1.853248e+04</td>
      <td>3.526155e+04</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>2.642214e+04</td>
      <td>26996.725000</td>
      <td>-1.070974</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16898</td>
      <td>1.849231e+04</td>
      <td>3.525008e+04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>2.642937e+04</td>
      <td>27003.240000</td>
      <td>-1.072023</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16897</td>
      <td>1.851205e+04</td>
      <td>3.525579e+04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>1.936919e+06</td>
      <td>686671.500000</td>
      <td>6.006053</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>40002</td>
      <td>-2.608447e+06</td>
      <td>5.061919e+06</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>1.999181e+03</td>
      <td>625.000000</td>
      <td>3.925960</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>35202</td>
      <td>-3.139000e+03</td>
      <td>6.141000e+03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>1.150212e+01</td>
      <td>12.000000</td>
      <td>0.000020</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-1.200000e+01</td>
      <td>3.600000e+01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>2.642951e+04</td>
      <td>27003.695017</td>
      <td>-1.071600</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16897</td>
      <td>1.851411e+04</td>
      <td>3.525485e+04</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>2.642986e+04</td>
      <td>27004.008458</td>
      <td>-1.070209</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16900</td>
      <td>1.851329e+04</td>
      <td>3.525427e+04</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>2.643040e+04</td>
      <td>27003.987380</td>
      <td>-1.068072</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16904</td>
      <td>1.851819e+04</td>
      <td>3.524944e+04</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>2.643147e+04</td>
      <td>27004.722835</td>
      <td>-1.063713</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16912</td>
      <td>1.850771e+04</td>
      <td>3.526611e+04</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>2.643290e+04</td>
      <td>27006.561496</td>
      <td>-1.057805</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16919</td>
      <td>1.848756e+04</td>
      <td>3.527931e+04</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>2.643650e+04</td>
      <td>27007.734006</td>
      <td>-1.042826</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16932</td>
      <td>1.844968e+04</td>
      <td>3.532011e+04</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>2.642968e+04</td>
      <td>27003.828095</td>
      <td>-1.070665</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16899</td>
      <td>1.851003e+04</td>
      <td>3.525965e+04</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>-4.972645e-01</td>
      <td>-0.088448</td>
      <td>-79.632339</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>38047</td>
      <td>-2.370995e+01</td>
      <td>2.333803e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>2.008872e-04</td>
      <td>0.003863</td>
      <td>-88.940522</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>34626</td>
      <td>-7.597227e+00</td>
      <td>7.608797e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>-4.974654e-01</td>
      <td>-0.104012</td>
      <td>-74.510620</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>38369</td>
      <td>-2.214112e+01</td>
      <td>2.176715e+01</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>1.898047e+01</td>
      <td>14.774750</td>
      <td>7.564913</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>23231</td>
      <td>-1.408509e+01</td>
      <td>4.670312e+01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>2.642939e+04</td>
      <td>27002.779758</td>
      <td>-1.071577</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16897</td>
      <td>1.851267e+04</td>
      <td>3.525499e+04</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>2.642927e+04</td>
      <td>27002.653488</td>
      <td>-1.071935</td>
      <td>Left-Skewed</td>
      <td>Skewed</td>
      <td>16893</td>
      <td>1.850947e+04</td>
      <td>3.525623e+04</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>-4.653184e-01</td>
      <td>-0.240206</td>
      <td>-0.031788</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>1760</td>
      <td>-6.959515e+01</td>
      <td>6.876740e+01</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>-1.228425e-02</td>
      <td>-0.011726</td>
      <td>-0.006510</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>55</td>
      <td>-4.281221e+00</td>
      <td>4.259650e+00</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>3.362288e-02</td>
      <td>0.003000</td>
      <td>-1.160606</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>35172</td>
      <td>-6.668000e+00</td>
      <td>6.724000e+00</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>2.023000e+03</td>
      <td>2023.000000</td>
      <td>0.000000</td>
      <td>Symmetric</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>2.023000e+03</td>
      <td>2.023000e+03</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>3.016923e+00</td>
      <td>3.000000</td>
      <td>-0.010392</td>
      <td>Left-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-5.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>25</th>
      <td>month</td>
      <td>5.366186e+00</td>
      <td>5.000000</td>
      <td>0.024370</td>
      <td>Right-Skewed</td>
      <td>Skewed</td>
      <td>0</td>
      <td>-4.500000e+00</td>
      <td>1.550000e+01</td>
    </tr>
    <tr>
      <th>26</th>
      <td>week_of_year</td>
      <td>2.148851e+01</td>
      <td>21.000000</td>
      <td>0.047058</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>0</td>
      <td>-2.050000e+01</td>
      <td>6.350000e+01</td>
    </tr>
    <tr>
      <th>27</th>
      <td>time_delta</td>
      <td>6.064038e+01</td>
      <td>60.000000</td>
      <td>521.356455</td>
      <td>Right-Skewed</td>
      <td>Symmetric</td>
      <td>87</td>
      <td>6.000000e+01</td>
      <td>6.000000e+01</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Display price action for the entire dataset
print("Displaying price action for the entire dataset...")
plot_price_action(df)
```

    Displaying price action for the entire dataset...



    
![png](data_analysis_files/data_analysis_17_1.png)
    



```python
# Display price action for each year
print("Displaying price action for each year...")
yearly_data = split_by_year(df)  # Split the dataset by year
for year, yearly_df in yearly_data.items():
    plot_price_action(yearly_df, year)
```

    Displaying price action for each year...



    
![png](data_analysis_files/data_analysis_18_1.png)
    



    
![png](data_analysis_files/data_analysis_18_2.png)
    



    
![png](data_analysis_files/data_analysis_18_3.png)
    



    
![png](data_analysis_files/data_analysis_18_4.png)
    



```python
# Print price trend for the entire dataset
print_price_trend(df)
```

    Price trend for the entire dataset:
    Start Price: 7179.01
    End Price: 29992.46
    Trend: The price increased by 317.78%.
    



```python
# Print price trend for each year
yearly_data = split_by_year(df)  # Split the dataset by year
for year, yearly_df in yearly_data.items():
    print_price_trend(yearly_df, year)
```

    Price trend for the year 2020:
    Start Price: 7179.01
    End Price: 28923.63
    Trend: The price increased by 302.89%.
    
    Price trend for the year 2021:
    Start Price: 28961.66
    End Price: 46216.93
    Trend: The price increased by 59.58%.
    
    Price trend for the year 2022:
    Start Price: 46250.00
    End Price: 16542.40
    Trend: The price decreased by -64.23%.
    
    Price trend for the year 2023:
    Start Price: 16543.67
    End Price: 29992.46
    Trend: The price increased by 81.29%.
    



```python
from IPython.display import display

# General Overview and Structure
print("=" * 50)
print("GENERAL OVERVIEW AND STRUCTURE")
print("=" * 50)

print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")

print("Dataset Structure:")
print("-" * 50)
display(df.info())  # df.info() prints directly; we leave this as it does not return a DataFrame.

print("\nDescriptive Statistics (Entire Dataset):")
print("-" * 50)
display(df.describe(include='all'))  # Use display for better formatting
```

    ==================================================
    GENERAL OVERVIEW AND STRUCTURE
    ==================================================
    Total Rows: 1997210
    Total Columns: 32
    Date Range: 2020-01-01 00:01:00 to 2023-10-22 23:59:00
    
    Dataset Structure:
    --------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1997210 entries, 0 to 1997209
    Data columns (total 32 columns):
     #   Column        Dtype         
    ---  ------        -----         
     0   date          datetime64[ns]
     1   symbol        object        
     2   open          float64       
     3   high          float64       
     4   low           float64       
     5   close         float64       
     6   Volume USDT   int64         
     7   tradecount    int64         
     8   token         object        
     9   hour          int32         
     10  day           object        
     11  ema_5         float64       
     12  ema_15        float64       
     13  ema_30        float64       
     14  ema_60        float64       
     15  ema_100       float64       
     16  ema_200       float64       
     17  WMA           float64       
     18  MACD          float64       
     19  MACD_Signal   float64       
     20  MACD_Hist     float64       
     21  ATR           float64       
     22  HMA           float64       
     23  KAMA          float64       
     24  CMO           float64       
     25  Z-Score       float64       
     26  QStick        float64       
     27  year          int32         
     28  day_of_week   int32         
     29  month         int32         
     30  week_of_year  UInt32        
     31  time_delta    float64       
    dtypes: UInt32(1), datetime64[ns](1), float64(21), int32(4), int64(2), object(3)
    memory usage: 451.4+ MB



    None


    
    Descriptive Statistics (Entire Dataset):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>token</th>
      <th>hour</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1997210</td>
      <td>1997210</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1997210</td>
      <td>1.997210e+06</td>
      <td>...</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1.997210e+06</td>
      <td>1997210.0</td>
      <td>1.997210e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>BTCUSDT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BTC</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>1470521</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1997210</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2021-11-25 05:55:32.246686464</td>
      <td>NaN</td>
      <td>2.834777e+04</td>
      <td>2.836410e+04</td>
      <td>2.833144e+04</td>
      <td>2.834778e+04</td>
      <td>1.943680e+06</td>
      <td>1.510783e+03</td>
      <td>NaN</td>
      <td>1.150389e+01</td>
      <td>...</td>
      <td>2.834777e+04</td>
      <td>2.834718e+04</td>
      <td>-4.089449e-01</td>
      <td>-1.446718e-02</td>
      <td>1.115378e-02</td>
      <td>2.021420e+03</td>
      <td>3.004968e+00</td>
      <td>6.278287e+00</td>
      <td>25.595482</td>
      <td>6.017508e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2020-01-01 00:01:00</td>
      <td>NaN</td>
      <td>3.706960e+03</td>
      <td>3.779000e+03</td>
      <td>3.621810e+03</td>
      <td>3.706960e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>3.710075e+03</td>
      <td>0.000000e+00</td>
      <td>-9.851256e+01</td>
      <td>-5.294722e+00</td>
      <td>-6.405370e+02</td>
      <td>2.020000e+03</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2020-12-12 18:43:15</td>
      <td>NaN</td>
      <td>1.673710e+04</td>
      <td>1.674291e+04</td>
      <td>1.673151e+04</td>
      <td>1.673710e+04</td>
      <td>5.021320e+05</td>
      <td>4.050000e+02</td>
      <td>NaN</td>
      <td>6.000000e+00</td>
      <td>...</td>
      <td>1.673706e+04</td>
      <td>1.673693e+04</td>
      <td>-1.634124e+01</td>
      <td>-1.070748e+00</td>
      <td>-2.402000e+00</td>
      <td>2.020000e+03</td>
      <td>1.000000e+00</td>
      <td>3.000000e+00</td>
      <td>13.0</td>
      <td>6.000000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2021-11-25 09:48:30</td>
      <td>NaN</td>
      <td>2.675128e+04</td>
      <td>2.675846e+04</td>
      <td>2.674433e+04</td>
      <td>2.675128e+04</td>
      <td>1.084630e+06</td>
      <td>7.650000e+02</td>
      <td>NaN</td>
      <td>1.200000e+01</td>
      <td>...</td>
      <td>2.675102e+04</td>
      <td>2.675157e+04</td>
      <td>-1.232780e-01</td>
      <td>-8.871890e-03</td>
      <td>9.000000e-03</td>
      <td>2.021000e+03</td>
      <td>3.000000e+00</td>
      <td>6.000000e+00</td>
      <td>25.0</td>
      <td>6.000000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2022-11-07 03:30:45</td>
      <td>NaN</td>
      <td>3.931771e+04</td>
      <td>3.934826e+04</td>
      <td>3.928777e+04</td>
      <td>3.931770e+04</td>
      <td>2.250337e+06</td>
      <td>1.724000e+03</td>
      <td>NaN</td>
      <td>1.800000e+01</td>
      <td>...</td>
      <td>3.931876e+04</td>
      <td>3.931787e+04</td>
      <td>1.568401e+01</td>
      <td>1.047175e+00</td>
      <td>2.437000e+00</td>
      <td>2.022000e+03</td>
      <td>5.000000e+00</td>
      <td>9.000000e+00</td>
      <td>38.0</td>
      <td>6.000000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023-10-22 23:59:00</td>
      <td>NaN</td>
      <td>6.900000e+04</td>
      <td>6.900000e+04</td>
      <td>6.878670e+04</td>
      <td>6.900000e+04</td>
      <td>1.609358e+08</td>
      <td>1.073150e+05</td>
      <td>NaN</td>
      <td>2.300000e+01</td>
      <td>...</td>
      <td>6.887801e+04</td>
      <td>6.863613e+04</td>
      <td>9.992688e+01</td>
      <td>5.316525e+00</td>
      <td>5.377490e+02</td>
      <td>2.023000e+03</td>
      <td>6.000000e+00</td>
      <td>1.200000e+01</td>
      <td>53.0</td>
      <td>1.729200e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.530104e+04</td>
      <td>1.531184e+04</td>
      <td>1.529021e+04</td>
      <td>1.530103e+04</td>
      <td>3.008599e+06</td>
      <td>2.137080e+03</td>
      <td>NaN</td>
      <td>6.922103e+00</td>
      <td>...</td>
      <td>1.530107e+04</td>
      <td>1.530036e+04</td>
      <td>2.257556e+01</td>
      <td>1.322931e+00</td>
      <td>1.057125e+01</td>
      <td>1.091780e+00</td>
      <td>1.997170e+00</td>
      <td>3.357639e+00</td>
      <td>14.673031</td>
      <td>1.385395e+02</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 32 columns</p>
</div>



```python
# Overview for each year
print("\n" + "="*50)
print("BASIC STATISTICS BY YEAR")
print("="*50)
yearly_data = split_by_year(df)

for year, yearly_df in yearly_data.items():
    print(f"\nYear {year}:")
    print(f"Rows: {len(yearly_df)}")
    print(f"Date Range: {yearly_df['date'].min()} to {yearly_df['date'].max()}")
    print(f"Descriptive Statistics for {year}:")
    print("-"*50)
    display(yearly_df.describe(include='all'))
```

    
    ==================================================
    BASIC STATISTICS BY YEAR
    ==================================================
    
    Year 2020:
    Rows: 526689
    Date Range: 2020-01-01 00:01:00 to 2020-12-31 23:59:00
    Descriptive Statistics for 2020:
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>token</th>
      <th>hour</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>526689</td>
      <td>526689</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>5.266890e+05</td>
      <td>526689.000000</td>
      <td>526689</td>
      <td>526689.000000</td>
      <td>...</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.0</td>
      <td>526689.000000</td>
      <td>526689.000000</td>
      <td>526689.0</td>
      <td>526689.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>BTC/USDT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BTC</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>526689</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>526689</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2020-07-01 21:17:42.939799040</td>
      <td>NaN</td>
      <td>11064.888930</td>
      <td>11070.590997</td>
      <td>11058.959931</td>
      <td>11064.932102</td>
      <td>1.356381e+06</td>
      <td>580.693386</td>
      <td>NaN</td>
      <td>11.499955</td>
      <td>...</td>
      <td>11064.899723</td>
      <td>11064.013238</td>
      <td>-0.959261</td>
      <td>-0.037204</td>
      <td>0.043243</td>
      <td>2020.0</td>
      <td>2.998806</td>
      <td>6.510140</td>
      <td>26.912692</td>
      <td>60.039758</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2020-01-01 00:01:00</td>
      <td>NaN</td>
      <td>3706.960000</td>
      <td>3779.000000</td>
      <td>3621.810000</td>
      <td>3706.960000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3710.075374</td>
      <td>0.000000</td>
      <td>-94.333210</td>
      <td>-5.294643</td>
      <td>-145.597000</td>
      <td>2020.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2020-04-01 10:33:00</td>
      <td>NaN</td>
      <td>8858.020000</td>
      <td>8863.180000</td>
      <td>8853.180000</td>
      <td>8858.020000</td>
      <td>3.521540e+05</td>
      <td>200.000000</td>
      <td>NaN</td>
      <td>5.000000</td>
      <td>...</td>
      <td>8858.226808</td>
      <td>8857.978925</td>
      <td>-16.291146</td>
      <td>-1.081533</td>
      <td>-1.061000</td>
      <td>2020.0</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>14.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2020-07-01 21:05:00</td>
      <td>NaN</td>
      <td>9692.440000</td>
      <td>9696.130000</td>
      <td>9688.550000</td>
      <td>9692.420000</td>
      <td>7.125730e+05</td>
      <td>379.000000</td>
      <td>NaN</td>
      <td>11.000000</td>
      <td>...</td>
      <td>9692.599091</td>
      <td>9691.970879</td>
      <td>-0.502966</td>
      <td>-0.035587</td>
      <td>0.028000</td>
      <td>2020.0</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>27.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2020-10-01 07:37:00</td>
      <td>NaN</td>
      <td>11631.300000</td>
      <td>11635.010000</td>
      <td>11627.000000</td>
      <td>11631.270000</td>
      <td>1.420788e+06</td>
      <td>694.000000</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>...</td>
      <td>11631.391697</td>
      <td>11630.667214</td>
      <td>14.628633</td>
      <td>1.012996</td>
      <td>1.141000</td>
      <td>2020.0</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>40.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2020-12-31 23:59:00</td>
      <td>NaN</td>
      <td>29290.000000</td>
      <td>29300.000000</td>
      <td>29249.840000</td>
      <td>29293.140000</td>
      <td>1.609358e+08</td>
      <td>29025.000000</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>...</td>
      <td>29301.398364</td>
      <td>29187.550781</td>
      <td>69.144459</td>
      <td>4.743345</td>
      <td>90.429000</td>
      <td>2020.0</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>53.0</td>
      <td>13860.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>4237.990462</td>
      <td>4241.026833</td>
      <td>4234.719921</td>
      <td>4238.058336</td>
      <td>2.743690e+06</td>
      <td>767.469860</td>
      <td>NaN</td>
      <td>6.922931</td>
      <td>...</td>
      <td>4237.906698</td>
      <td>4236.071742</td>
      <td>21.521757</td>
      <td>1.312844</td>
      <td>3.600614</td>
      <td>0.0</td>
      <td>1.994608</td>
      <td>3.449643</td>
      <td>15.089928</td>
      <td>20.268171</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 32 columns</p>
</div>


    
    Year 2021:
    Rows: 524607
    Date Range: 2021-01-01 00:00:00 to 2021-12-31 23:59:00
    Descriptive Statistics for 2021:
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>token</th>
      <th>hour</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>524607</td>
      <td>524607</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>5.246070e+05</td>
      <td>524607.000000</td>
      <td>524607</td>
      <td>524607.000000</td>
      <td>...</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.0</td>
      <td>524607.000000</td>
      <td>524607.000000</td>
      <td>524607.0</td>
      <td>524607.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>BTCUSDT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BTC</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>524607</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>524607</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2021-07-02 13:22:38.476918528</td>
      <td>NaN</td>
      <td>47357.809055</td>
      <td>47393.651627</td>
      <td>47322.028206</td>
      <td>47357.839908</td>
      <td>2.199850e+06</td>
      <td>1278.537999</td>
      <td>NaN</td>
      <td>11.513146</td>
      <td>...</td>
      <td>47357.826010</td>
      <td>47356.462888</td>
      <td>-0.300910</td>
      <td>-0.006288</td>
      <td>0.030789</td>
      <td>2021.0</td>
      <td>3.001064</td>
      <td>6.528079</td>
      <td>26.587316</td>
      <td>60.113571</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2021-01-01 00:00:00</td>
      <td>NaN</td>
      <td>28241.950000</td>
      <td>28764.230000</td>
      <td>28130.000000</td>
      <td>28235.470000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
      <td>16642.295596</td>
      <td>24505.641355</td>
      <td>-80.307978</td>
      <td>-4.756327</td>
      <td>-640.537000</td>
      <td>2021.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2021-04-02 04:40:30</td>
      <td>NaN</td>
      <td>38058.340000</td>
      <td>38100.000000</td>
      <td>38014.980000</td>
      <td>38058.275000</td>
      <td>8.621350e+05</td>
      <td>666.000000</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>...</td>
      <td>38058.179611</td>
      <td>38054.104975</td>
      <td>-15.728092</td>
      <td>-1.064421</td>
      <td>-8.036000</td>
      <td>2021.0</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>14.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2021-07-02 13:46:00</td>
      <td>NaN</td>
      <td>47898.140000</td>
      <td>47929.990000</td>
      <td>47866.960000</td>
      <td>47898.140000</td>
      <td>1.449567e+06</td>
      <td>984.000000</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>...</td>
      <td>47899.013434</td>
      <td>47898.715876</td>
      <td>0.169248</td>
      <td>0.012799</td>
      <td>-0.088000</td>
      <td>2021.0</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>27.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2021-10-01 22:07:30</td>
      <td>NaN</td>
      <td>56204.835000</td>
      <td>56247.805000</td>
      <td>56162.755000</td>
      <td>56204.875000</td>
      <td>2.518188e+06</td>
      <td>1492.000000</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>...</td>
      <td>56204.820530</td>
      <td>56195.781741</td>
      <td>15.493760</td>
      <td>1.063037</td>
      <td>8.024000</td>
      <td>2021.0</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>40.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2021-12-31 23:59:00</td>
      <td>NaN</td>
      <td>69000.000000</td>
      <td>69000.000000</td>
      <td>68786.700000</td>
      <td>69000.000000</td>
      <td>1.155494e+08</td>
      <td>55181.000000</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>...</td>
      <td>68878.012101</td>
      <td>68636.128490</td>
      <td>99.926881</td>
      <td>5.294810</td>
      <td>537.749000</td>
      <td>2021.0</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>53.0</td>
      <td>17100.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>9822.002675</td>
      <td>9821.962228</td>
      <td>9821.482643</td>
      <td>9821.979396</td>
      <td>2.760353e+06</td>
      <td>1207.579415</td>
      <td>NaN</td>
      <td>6.921611</td>
      <td>...</td>
      <td>9822.198390</td>
      <td>9819.967035</td>
      <td>22.083263</td>
      <td>1.322369</td>
      <td>17.929809</td>
      <td>0.0</td>
      <td>1.997973</td>
      <td>3.449229</td>
      <td>15.066706</td>
      <td>37.488186</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 32 columns</p>
</div>


    
    Year 2022:
    Rows: 525600
    Date Range: 2022-01-01 00:00:00 to 2022-12-31 23:59:00
    Descriptive Statistics for 2022:
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>token</th>
      <th>hour</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>525600</td>
      <td>525600</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>5.256000e+05</td>
      <td>525600.000000</td>
      <td>525600</td>
      <td>525600.000000</td>
      <td>...</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.0</td>
      <td>525600.000000</td>
      <td>525600.000000</td>
      <td>525600.0</td>
      <td>525600.0</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>BTCUSDT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BTC</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>525600</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>525600</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2022-07-02 11:59:29.999999232</td>
      <td>NaN</td>
      <td>28226.460083</td>
      <td>28241.320202</td>
      <td>28211.818542</td>
      <td>28226.401720</td>
      <td>2.281915e+06</td>
      <td>2284.043409</td>
      <td>NaN</td>
      <td>11.500000</td>
      <td>...</td>
      <td>28226.391011</td>
      <td>28226.524774</td>
      <td>0.079761</td>
      <td>-0.001592</td>
      <td>-0.058569</td>
      <td>2022.0</td>
      <td>3.005479</td>
      <td>6.526027</td>
      <td>26.569863</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2022-01-01 00:00:00</td>
      <td>NaN</td>
      <td>15513.840000</td>
      <td>15544.470000</td>
      <td>15476.000000</td>
      <td>15513.840000</td>
      <td>1.122100e+04</td>
      <td>89.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
      <td>13584.206333</td>
      <td>15612.932532</td>
      <td>-96.896262</td>
      <td>-5.291539</td>
      <td>-137.259000</td>
      <td>2022.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2022-04-02 05:59:45</td>
      <td>NaN</td>
      <td>19521.675000</td>
      <td>19531.390000</td>
      <td>19512.775000</td>
      <td>19521.625000</td>
      <td>6.852200e+05</td>
      <td>599.000000</td>
      <td>NaN</td>
      <td>5.750000</td>
      <td>...</td>
      <td>19521.661952</td>
      <td>19522.282018</td>
      <td>-15.975816</td>
      <td>-1.060152</td>
      <td>-2.851000</td>
      <td>2022.0</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>14.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2022-07-02 11:59:30</td>
      <td>NaN</td>
      <td>23148.200000</td>
      <td>23160.185000</td>
      <td>23135.610000</td>
      <td>23147.950000</td>
      <td>1.427310e+06</td>
      <td>1563.000000</td>
      <td>NaN</td>
      <td>11.500000</td>
      <td>...</td>
      <td>23148.597000</td>
      <td>23149.311336</td>
      <td>0.051614</td>
      <td>-0.000135</td>
      <td>0.007000</td>
      <td>2022.0</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>27.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2022-10-01 17:59:15</td>
      <td>NaN</td>
      <td>39010.810000</td>
      <td>39030.430000</td>
      <td>38993.247500</td>
      <td>39010.780000</td>
      <td>2.784906e+06</td>
      <td>3142.000000</td>
      <td>NaN</td>
      <td>17.250000</td>
      <td>...</td>
      <td>39010.659836</td>
      <td>39011.188329</td>
      <td>16.078940</td>
      <td>1.056238</td>
      <td>2.805250</td>
      <td>2022.0</td>
      <td>5.000000</td>
      <td>10.000000</td>
      <td>40.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2022-12-31 23:59:00</td>
      <td>NaN</td>
      <td>48164.320000</td>
      <td>48189.840000</td>
      <td>48120.010000</td>
      <td>48164.320000</td>
      <td>9.739113e+07</td>
      <td>69114.000000</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>...</td>
      <td>48165.899283</td>
      <td>48062.587912</td>
      <td>74.011506</td>
      <td>4.796966</td>
      <td>208.459000</td>
      <td>2022.0</td>
      <td>6.000000</td>
      <td>12.000000</td>
      <td>52.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>10191.352879</td>
      <td>10196.076757</td>
      <td>10186.613974</td>
      <td>10191.335897</td>
      <td>2.943094e+06</td>
      <td>2450.809039</td>
      <td>NaN</td>
      <td>6.922193</td>
      <td>...</td>
      <td>10191.354982</td>
      <td>10191.293529</td>
      <td>22.424022</td>
      <td>1.327995</td>
      <td>8.492095</td>
      <td>0.0</td>
      <td>1.999994</td>
      <td>3.447855</td>
      <td>15.046924</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 32 columns</p>
</div>


    
    Year 2023:
    Rows: 420314
    Date Range: 2023-01-01 00:00:00 to 2023-10-22 23:59:00
    Descriptive Statistics for 2023:
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>symbol</th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>token</th>
      <th>hour</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420314</td>
      <td>420314</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>4.203140e+05</td>
      <td>420314.000000</td>
      <td>420314</td>
      <td>420314.000000</td>
      <td>...</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.0</td>
      <td>420314.000000</td>
      <td>420314.000000</td>
      <td>420314.0</td>
      <td>420314.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>BTCUSDT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>BTC</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>420314</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>420314</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;NA&gt;</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2023-05-27 16:32:24.860271104</td>
      <td>NaN</td>
      <td>26429.334724</td>
      <td>26436.481508</td>
      <td>26422.143658</td>
      <td>26429.368001</td>
      <td>1.936919e+06</td>
      <td>1999.180534</td>
      <td>NaN</td>
      <td>11.502115</td>
      <td>...</td>
      <td>26429.391302</td>
      <td>26429.266399</td>
      <td>-0.465318</td>
      <td>-0.012284</td>
      <td>0.033623</td>
      <td>2023.0</td>
      <td>3.016923</td>
      <td>5.366186</td>
      <td>21.488511</td>
      <td>60.640378</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2023-01-01 00:00:00</td>
      <td>NaN</td>
      <td>16506.040000</td>
      <td>16508.730000</td>
      <td>16499.010000</td>
      <td>16505.870000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>...</td>
      <td>16507.074475</td>
      <td>16512.619444</td>
      <td>-98.512559</td>
      <td>-5.294722</td>
      <td>-252.684000</td>
      <td>2023.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2023-03-14 23:18:15</td>
      <td>NaN</td>
      <td>24790.837500</td>
      <td>24805.877500</td>
      <td>24776.470000</td>
      <td>24790.952500</td>
      <td>2.679400e+05</td>
      <td>341.000000</td>
      <td>NaN</td>
      <td>6.000000</td>
      <td>...</td>
      <td>24791.040654</td>
      <td>24789.503610</td>
      <td>-17.709191</td>
      <td>-1.078394</td>
      <td>-1.646000</td>
      <td>2023.0</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>11.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2023-05-26 23:56:30</td>
      <td>NaN</td>
      <td>27003.235000</td>
      <td>27010.005000</td>
      <td>26996.725000</td>
      <td>27003.240000</td>
      <td>6.866715e+05</td>
      <td>625.000000</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>...</td>
      <td>27002.779758</td>
      <td>27002.653488</td>
      <td>-0.240206</td>
      <td>-0.011726</td>
      <td>0.003000</td>
      <td>2023.0</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>21.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2023-08-08 23:27:45</td>
      <td>NaN</td>
      <td>28976.877500</td>
      <td>28988.145000</td>
      <td>28965.912500</td>
      <td>28976.887500</td>
      <td>2.185532e+06</td>
      <td>2661.000000</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>...</td>
      <td>28976.622144</td>
      <td>28976.195198</td>
      <td>16.881446</td>
      <td>1.056823</td>
      <td>1.702000</td>
      <td>2023.0</td>
      <td>5.000000</td>
      <td>8.000000</td>
      <td>32.0</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023-10-22 23:59:00</td>
      <td>NaN</td>
      <td>31798.010000</td>
      <td>31804.200000</td>
      <td>31723.970000</td>
      <td>31798.000000</td>
      <td>1.459557e+08</td>
      <td>107315.000000</td>
      <td>NaN</td>
      <td>23.000000</td>
      <td>...</td>
      <td>42318.067980</td>
      <td>39404.615551</td>
      <td>96.449769</td>
      <td>5.316525</td>
      <td>194.468000</td>
      <td>2023.0</td>
      <td>6.000000</td>
      <td>10.000000</td>
      <td>52.0</td>
      <td>172920.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>3261.678056</td>
      <td>3261.226737</td>
      <td>3262.052336</td>
      <td>3261.646452</td>
      <td>3.546687e+06</td>
      <td>3081.368711</td>
      <td>NaN</td>
      <td>6.921583</td>
      <td>...</td>
      <td>3261.758678</td>
      <td>3261.376840</td>
      <td>24.559374</td>
      <td>1.329511</td>
      <td>4.830114</td>
      <td>0.0</td>
      <td>1.995796</td>
      <td>2.808770</td>
      <td>12.238709</td>
      <td>298.213912</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 32 columns</p>
</div>



```python
# Correlations and Relationships
print("\n" + "="*50)
print("CORRELATIONS AND RELATIONSHIPS")
print("="*50)

# Filter numeric columns
numeric_columns = df.select_dtypes(include=[np.number])

# Entire dataset correlation
print("\nCorrelation Matrix (Entire Dataset):")
print("-"*50)
correlation_matrix = numeric_columns.corr()
display(correlation_matrix)  # Use display for better formatting

# Correlation heatmap for the entire dataset
plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap (Entire Dataset)")
plt.show()

# Clear the plot after display
plt.clf()  # Clear the current figure
plt.close()  # Close the figure completely to free memory

# Per year correlation and heatmap
for year, yearly_df in yearly_data.items():
    print(f"\nCorrelation Matrix (Year {year}):")
    print("-"*50)
    yearly_numeric = yearly_df.select_dtypes(include=[np.number])
    yearly_correlation = yearly_numeric.corr()
    display(yearly_correlation)  # Use display for yearly correlations

    plt.figure(figsize=(20, 16))
    sns.heatmap(yearly_correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(f"Correlation Heatmap (Year {year})")
    plt.show()

    # Clear the plot after display
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure completely to free memory
```

    
    ==================================================
    CORRELATIONS AND RELATIONSHIPS
    ==================================================
    
    Correlation Matrix (Entire Dataset):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>hour</th>
      <th>ema_5</th>
      <th>ema_15</th>
      <th>ema_30</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>open</th>
      <td>1.000000</td>
      <td>0.999998</td>
      <td>0.999998</td>
      <td>0.999998</td>
      <td>0.020577</td>
      <td>-0.041412</td>
      <td>0.001115</td>
      <td>0.999994</td>
      <td>0.999983</td>
      <td>0.999967</td>
      <td>...</td>
      <td>0.999993</td>
      <td>0.999984</td>
      <td>0.013225</td>
      <td>0.011752</td>
      <td>-0.004112</td>
      <td>0.222582</td>
      <td>-0.002157</td>
      <td>-0.017614</td>
      <td>-0.030041</td>
      <td>0.000167</td>
    </tr>
    <tr>
      <th>high</th>
      <td>0.999998</td>
      <td>1.000000</td>
      <td>0.999997</td>
      <td>0.999998</td>
      <td>0.021273</td>
      <td>-0.040932</td>
      <td>0.001166</td>
      <td>0.999995</td>
      <td>0.999984</td>
      <td>0.999968</td>
      <td>...</td>
      <td>0.999994</td>
      <td>0.999985</td>
      <td>0.013246</td>
      <td>0.011769</td>
      <td>-0.003713</td>
      <td>0.222338</td>
      <td>-0.002251</td>
      <td>-0.017761</td>
      <td>-0.030181</td>
      <td>0.000169</td>
    </tr>
    <tr>
      <th>low</th>
      <td>0.999998</td>
      <td>0.999997</td>
      <td>1.000000</td>
      <td>0.999998</td>
      <td>0.019827</td>
      <td>-0.041917</td>
      <td>0.001064</td>
      <td>0.999995</td>
      <td>0.999984</td>
      <td>0.999968</td>
      <td>...</td>
      <td>0.999994</td>
      <td>0.999985</td>
      <td>0.013279</td>
      <td>0.011810</td>
      <td>-0.003776</td>
      <td>0.222832</td>
      <td>-0.002063</td>
      <td>-0.017474</td>
      <td>-0.029908</td>
      <td>0.000166</td>
    </tr>
    <tr>
      <th>close</th>
      <td>0.999998</td>
      <td>0.999998</td>
      <td>0.999998</td>
      <td>1.000000</td>
      <td>0.020551</td>
      <td>-0.041424</td>
      <td>0.001116</td>
      <td>0.999996</td>
      <td>0.999985</td>
      <td>0.999969</td>
      <td>...</td>
      <td>0.999996</td>
      <td>0.999987</td>
      <td>0.013285</td>
      <td>0.011809</td>
      <td>-0.003429</td>
      <td>0.222581</td>
      <td>-0.002157</td>
      <td>-0.017615</td>
      <td>-0.030041</td>
      <td>0.000168</td>
    </tr>
    <tr>
      <th>Volume USDT</th>
      <td>0.020577</td>
      <td>0.021273</td>
      <td>0.019827</td>
      <td>0.020551</td>
      <td>1.000000</td>
      <td>0.818687</td>
      <td>0.030709</td>
      <td>0.020576</td>
      <td>0.020617</td>
      <td>0.020655</td>
      <td>...</td>
      <td>0.020562</td>
      <td>0.020589</td>
      <td>-0.008449</td>
      <td>-0.010933</td>
      <td>0.009006</td>
      <td>0.073384</td>
      <td>-0.081176</td>
      <td>-0.066191</td>
      <td>-0.068525</td>
      <td>-0.000078</td>
    </tr>
    <tr>
      <th>tradecount</th>
      <td>-0.041412</td>
      <td>-0.040932</td>
      <td>-0.041917</td>
      <td>-0.041424</td>
      <td>0.818687</td>
      <td>1.000000</td>
      <td>0.029523</td>
      <td>-0.041406</td>
      <td>-0.041378</td>
      <td>-0.041347</td>
      <td>...</td>
      <td>-0.041415</td>
      <td>-0.041395</td>
      <td>-0.004418</td>
      <td>-0.006668</td>
      <td>0.006411</td>
      <td>0.280445</td>
      <td>-0.072149</td>
      <td>-0.000794</td>
      <td>-0.004343</td>
      <td>-0.000230</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.001115</td>
      <td>0.001166</td>
      <td>0.001064</td>
      <td>0.001116</td>
      <td>0.030709</td>
      <td>0.029523</td>
      <td>1.000000</td>
      <td>0.001117</td>
      <td>0.001120</td>
      <td>0.001129</td>
      <td>...</td>
      <td>0.001115</td>
      <td>0.001097</td>
      <td>-0.012427</td>
      <td>-0.012221</td>
      <td>0.001232</td>
      <td>-0.000097</td>
      <td>0.000280</td>
      <td>-0.000103</td>
      <td>-0.000118</td>
      <td>-0.001721</td>
    </tr>
    <tr>
      <th>ema_5</th>
      <td>0.999994</td>
      <td>0.999995</td>
      <td>0.999995</td>
      <td>0.999996</td>
      <td>0.020576</td>
      <td>-0.041406</td>
      <td>0.001117</td>
      <td>1.000000</td>
      <td>0.999994</td>
      <td>0.999980</td>
      <td>...</td>
      <td>0.999998</td>
      <td>0.999992</td>
      <td>0.012325</td>
      <td>0.010814</td>
      <td>-0.002080</td>
      <td>0.222583</td>
      <td>-0.002157</td>
      <td>-0.017617</td>
      <td>-0.030042</td>
      <td>0.000171</td>
    </tr>
    <tr>
      <th>ema_15</th>
      <td>0.999983</td>
      <td>0.999984</td>
      <td>0.999984</td>
      <td>0.999985</td>
      <td>0.020617</td>
      <td>-0.041378</td>
      <td>0.001120</td>
      <td>0.999994</td>
      <td>1.000000</td>
      <td>0.999995</td>
      <td>...</td>
      <td>0.999987</td>
      <td>0.999991</td>
      <td>0.010742</td>
      <td>0.009342</td>
      <td>-0.000080</td>
      <td>0.222588</td>
      <td>-0.002155</td>
      <td>-0.017624</td>
      <td>-0.030046</td>
      <td>0.000173</td>
    </tr>
    <tr>
      <th>ema_30</th>
      <td>0.999967</td>
      <td>0.999968</td>
      <td>0.999968</td>
      <td>0.999969</td>
      <td>0.020655</td>
      <td>-0.041347</td>
      <td>0.001129</td>
      <td>0.999980</td>
      <td>0.999995</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.999970</td>
      <td>0.999980</td>
      <td>0.009372</td>
      <td>0.008266</td>
      <td>0.000987</td>
      <td>0.222596</td>
      <td>-0.002153</td>
      <td>-0.017634</td>
      <td>-0.030052</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>ema_60</th>
      <td>0.999935</td>
      <td>0.999936</td>
      <td>0.999936</td>
      <td>0.999937</td>
      <td>0.020701</td>
      <td>-0.041302</td>
      <td>0.001150</td>
      <td>0.999950</td>
      <td>0.999971</td>
      <td>0.999989</td>
      <td>...</td>
      <td>0.999937</td>
      <td>0.999952</td>
      <td>0.007981</td>
      <td>0.007380</td>
      <td>0.001653</td>
      <td>0.222610</td>
      <td>-0.002148</td>
      <td>-0.017653</td>
      <td>-0.030065</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>ema_100</th>
      <td>0.999893</td>
      <td>0.999895</td>
      <td>0.999893</td>
      <td>0.999895</td>
      <td>0.020738</td>
      <td>-0.041261</td>
      <td>0.001173</td>
      <td>0.999908</td>
      <td>0.999933</td>
      <td>0.999960</td>
      <td>...</td>
      <td>0.999895</td>
      <td>0.999912</td>
      <td>0.007099</td>
      <td>0.006912</td>
      <td>0.001937</td>
      <td>0.222630</td>
      <td>-0.002139</td>
      <td>-0.017679</td>
      <td>-0.030081</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>ema_200</th>
      <td>0.999789</td>
      <td>0.999790</td>
      <td>0.999789</td>
      <td>0.999791</td>
      <td>0.020776</td>
      <td>-0.041205</td>
      <td>0.001182</td>
      <td>0.999804</td>
      <td>0.999832</td>
      <td>0.999868</td>
      <td>...</td>
      <td>0.999790</td>
      <td>0.999809</td>
      <td>0.006211</td>
      <td>0.006494</td>
      <td>0.002139</td>
      <td>0.222678</td>
      <td>-0.002121</td>
      <td>-0.017741</td>
      <td>-0.030121</td>
      <td>0.000174</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>0.999987</td>
      <td>0.999988</td>
      <td>0.999987</td>
      <td>0.999989</td>
      <td>0.020601</td>
      <td>-0.041390</td>
      <td>0.001115</td>
      <td>0.999997</td>
      <td>0.999999</td>
      <td>0.999988</td>
      <td>...</td>
      <td>0.999991</td>
      <td>0.999993</td>
      <td>0.011362</td>
      <td>0.009862</td>
      <td>-0.000566</td>
      <td>0.222586</td>
      <td>-0.002156</td>
      <td>-0.017622</td>
      <td>-0.030046</td>
      <td>0.000173</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>0.009351</td>
      <td>0.009331</td>
      <td>0.009395</td>
      <td>0.009375</td>
      <td>-0.011685</td>
      <td>-0.009421</td>
      <td>-0.002344</td>
      <td>0.008554</td>
      <td>0.005698</td>
      <td>0.002449</td>
      <td>...</td>
      <td>0.009657</td>
      <td>0.007555</td>
      <td>0.438694</td>
      <td>0.360701</td>
      <td>-0.386123</td>
      <td>-0.000955</td>
      <td>-0.000553</td>
      <td>0.002688</td>
      <td>0.001532</td>
      <td>-0.000271</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.000218</td>
      <td>0.000226</td>
      <td>0.000246</td>
      <td>0.000238</td>
      <td>-0.006584</td>
      <td>-0.003398</td>
      <td>0.001208</td>
      <td>-0.001381</td>
      <td>-0.003954</td>
      <td>-0.004637</td>
      <td>...</td>
      <td>0.000199</td>
      <td>-0.002288</td>
      <td>0.304526</td>
      <td>0.343938</td>
      <td>-0.592737</td>
      <td>0.000002</td>
      <td>0.000123</td>
      <td>-0.000021</td>
      <td>-0.000035</td>
      <td>-0.000716</td>
    </tr>
    <tr>
      <th>MACD_Hist</th>
      <td>0.009866</td>
      <td>0.009842</td>
      <td>0.009904</td>
      <td>0.009885</td>
      <td>-0.010300</td>
      <td>-0.008918</td>
      <td>-0.002880</td>
      <td>0.009533</td>
      <td>0.007325</td>
      <td>0.004092</td>
      <td>...</td>
      <td>0.010198</td>
      <td>0.008763</td>
      <td>0.368289</td>
      <td>0.272749</td>
      <td>-0.219808</td>
      <td>-0.001014</td>
      <td>-0.000627</td>
      <td>0.002861</td>
      <td>0.001638</td>
      <td>-0.000058</td>
    </tr>
    <tr>
      <th>ATR</th>
      <td>0.596346</td>
      <td>0.597083</td>
      <td>0.595578</td>
      <td>0.596297</td>
      <td>0.331288</td>
      <td>0.185892</td>
      <td>0.052404</td>
      <td>0.596132</td>
      <td>0.595881</td>
      <td>0.595702</td>
      <td>...</td>
      <td>0.596233</td>
      <td>0.596030</td>
      <td>0.015249</td>
      <td>0.013334</td>
      <td>-0.044406</td>
      <td>-0.068391</td>
      <td>-0.077549</td>
      <td>-0.129092</td>
      <td>-0.130489</td>
      <td>0.000149</td>
    </tr>
    <tr>
      <th>HMA</th>
      <td>0.999993</td>
      <td>0.999994</td>
      <td>0.999994</td>
      <td>0.999996</td>
      <td>0.020562</td>
      <td>-0.041415</td>
      <td>0.001115</td>
      <td>0.999998</td>
      <td>0.999987</td>
      <td>0.999970</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.999988</td>
      <td>0.012951</td>
      <td>0.011409</td>
      <td>-0.003181</td>
      <td>0.222582</td>
      <td>-0.002157</td>
      <td>-0.017616</td>
      <td>-0.030043</td>
      <td>0.000171</td>
    </tr>
    <tr>
      <th>KAMA</th>
      <td>0.999984</td>
      <td>0.999985</td>
      <td>0.999985</td>
      <td>0.999987</td>
      <td>0.020589</td>
      <td>-0.041395</td>
      <td>0.001097</td>
      <td>0.999992</td>
      <td>0.999991</td>
      <td>0.999980</td>
      <td>...</td>
      <td>0.999988</td>
      <td>1.000000</td>
      <td>0.010984</td>
      <td>0.009384</td>
      <td>-0.000254</td>
      <td>0.222620</td>
      <td>-0.002167</td>
      <td>-0.017640</td>
      <td>-0.030066</td>
      <td>0.000173</td>
    </tr>
    <tr>
      <th>CMO</th>
      <td>0.013225</td>
      <td>0.013246</td>
      <td>0.013279</td>
      <td>0.013285</td>
      <td>-0.008449</td>
      <td>-0.004418</td>
      <td>-0.012427</td>
      <td>0.012325</td>
      <td>0.010742</td>
      <td>0.009372</td>
      <td>...</td>
      <td>0.012951</td>
      <td>0.010984</td>
      <td>1.000000</td>
      <td>0.925434</td>
      <td>-0.510775</td>
      <td>0.010143</td>
      <td>0.002963</td>
      <td>-0.000256</td>
      <td>-0.002471</td>
      <td>-0.001316</td>
    </tr>
    <tr>
      <th>Z-Score</th>
      <td>0.011752</td>
      <td>0.011769</td>
      <td>0.011810</td>
      <td>0.011809</td>
      <td>-0.010933</td>
      <td>-0.006668</td>
      <td>-0.012221</td>
      <td>0.010814</td>
      <td>0.009342</td>
      <td>0.008266</td>
      <td>...</td>
      <td>0.011409</td>
      <td>0.009384</td>
      <td>0.925434</td>
      <td>1.000000</td>
      <td>-0.510994</td>
      <td>0.007133</td>
      <td>0.002661</td>
      <td>0.002314</td>
      <td>0.000652</td>
      <td>-0.001298</td>
    </tr>
    <tr>
      <th>QStick</th>
      <td>-0.004112</td>
      <td>-0.003713</td>
      <td>-0.003776</td>
      <td>-0.003429</td>
      <td>0.009006</td>
      <td>0.006411</td>
      <td>0.001232</td>
      <td>-0.002080</td>
      <td>-0.000080</td>
      <td>0.000987</td>
      <td>...</td>
      <td>-0.003181</td>
      <td>-0.000254</td>
      <td>-0.510775</td>
      <td>-0.510994</td>
      <td>1.000000</td>
      <td>-0.001504</td>
      <td>-0.000317</td>
      <td>-0.000666</td>
      <td>0.000254</td>
      <td>0.001071</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.222582</td>
      <td>0.222338</td>
      <td>0.222832</td>
      <td>0.222581</td>
      <td>0.073384</td>
      <td>0.280445</td>
      <td>-0.000097</td>
      <td>0.222583</td>
      <td>0.222588</td>
      <td>0.222596</td>
      <td>...</td>
      <td>0.222582</td>
      <td>0.222620</td>
      <td>0.010143</td>
      <td>0.007133</td>
      <td>-0.001504</td>
      <td>1.000000</td>
      <td>0.003115</td>
      <td>-0.103620</td>
      <td>-0.113583</td>
      <td>0.001226</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>-0.002157</td>
      <td>-0.002251</td>
      <td>-0.002063</td>
      <td>-0.002157</td>
      <td>-0.081176</td>
      <td>-0.072149</td>
      <td>0.000280</td>
      <td>-0.002157</td>
      <td>-0.002155</td>
      <td>-0.002153</td>
      <td>...</td>
      <td>-0.002157</td>
      <td>-0.002167</td>
      <td>0.002963</td>
      <td>0.002661</td>
      <td>-0.000317</td>
      <td>0.003115</td>
      <td>1.000000</td>
      <td>-0.000269</td>
      <td>-0.001804</td>
      <td>-0.000768</td>
    </tr>
    <tr>
      <th>month</th>
      <td>-0.017614</td>
      <td>-0.017761</td>
      <td>-0.017474</td>
      <td>-0.017615</td>
      <td>-0.066191</td>
      <td>-0.000794</td>
      <td>-0.000103</td>
      <td>-0.017617</td>
      <td>-0.017624</td>
      <td>-0.017634</td>
      <td>...</td>
      <td>-0.017616</td>
      <td>-0.017640</td>
      <td>-0.000256</td>
      <td>0.002314</td>
      <td>-0.000666</td>
      <td>-0.103620</td>
      <td>-0.000269</td>
      <td>1.000000</td>
      <td>0.970745</td>
      <td>0.000457</td>
    </tr>
    <tr>
      <th>week_of_year</th>
      <td>-0.030041</td>
      <td>-0.030181</td>
      <td>-0.029908</td>
      <td>-0.030041</td>
      <td>-0.068525</td>
      <td>-0.004343</td>
      <td>-0.000118</td>
      <td>-0.030042</td>
      <td>-0.030046</td>
      <td>-0.030052</td>
      <td>...</td>
      <td>-0.030043</td>
      <td>-0.030066</td>
      <td>-0.002471</td>
      <td>0.000652</td>
      <td>0.000254</td>
      <td>-0.113583</td>
      <td>-0.001804</td>
      <td>0.970745</td>
      <td>1.000000</td>
      <td>0.000547</td>
    </tr>
    <tr>
      <th>time_delta</th>
      <td>0.000167</td>
      <td>0.000169</td>
      <td>0.000166</td>
      <td>0.000168</td>
      <td>-0.000078</td>
      <td>-0.000230</td>
      <td>-0.001721</td>
      <td>0.000171</td>
      <td>0.000173</td>
      <td>0.000174</td>
      <td>...</td>
      <td>0.000171</td>
      <td>0.000173</td>
      <td>-0.001316</td>
      <td>-0.001298</td>
      <td>0.001071</td>
      <td>0.001226</td>
      <td>-0.000768</td>
      <td>0.000457</td>
      <td>0.000547</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 28 columns</p>
</div>



    
![png](data_analysis_files/data_analysis_23_2.png)
    


    
    Correlation Matrix (Year 2020):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>hour</th>
      <th>ema_5</th>
      <th>ema_15</th>
      <th>ema_30</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>open</th>
      <td>1.000000</td>
      <td>0.999997</td>
      <td>0.999997</td>
      <td>0.999996</td>
      <td>-0.032387</td>
      <td>0.226796</td>
      <td>0.004450</td>
      <td>0.999993</td>
      <td>0.999985</td>
      <td>0.999973</td>
      <td>...</td>
      <td>0.999990</td>
      <td>0.999773</td>
      <td>-0.018944</td>
      <td>-0.008744</td>
      <td>0.014683</td>
      <td>NaN</td>
      <td>-0.009416</td>
      <td>0.768378</td>
      <td>0.776067</td>
      <td>0.005143</td>
    </tr>
    <tr>
      <th>high</th>
      <td>0.999997</td>
      <td>1.000000</td>
      <td>0.999994</td>
      <td>0.999998</td>
      <td>-0.031530</td>
      <td>0.227848</td>
      <td>0.004504</td>
      <td>0.999995</td>
      <td>0.999986</td>
      <td>0.999974</td>
      <td>...</td>
      <td>0.999991</td>
      <td>0.999775</td>
      <td>-0.018894</td>
      <td>-0.008707</td>
      <td>0.015149</td>
      <td>NaN</td>
      <td>-0.009486</td>
      <td>0.768246</td>
      <td>0.775940</td>
      <td>0.005172</td>
    </tr>
    <tr>
      <th>low</th>
      <td>0.999997</td>
      <td>0.999994</td>
      <td>1.000000</td>
      <td>0.999997</td>
      <td>-0.033618</td>
      <td>0.225387</td>
      <td>0.004397</td>
      <td>0.999994</td>
      <td>0.999986</td>
      <td>0.999973</td>
      <td>...</td>
      <td>0.999991</td>
      <td>0.999774</td>
      <td>-0.018833</td>
      <td>-0.008625</td>
      <td>0.015159</td>
      <td>NaN</td>
      <td>-0.009353</td>
      <td>0.768516</td>
      <td>0.776200</td>
      <td>0.005118</td>
    </tr>
    <tr>
      <th>close</th>
      <td>0.999996</td>
      <td>0.999998</td>
      <td>0.999997</td>
      <td>1.000000</td>
      <td>-0.032558</td>
      <td>0.226667</td>
      <td>0.004456</td>
      <td>0.999997</td>
      <td>0.999988</td>
      <td>0.999976</td>
      <td>...</td>
      <td>0.999994</td>
      <td>0.999777</td>
      <td>-0.018824</td>
      <td>-0.008633</td>
      <td>0.015534</td>
      <td>NaN</td>
      <td>-0.009423</td>
      <td>0.768376</td>
      <td>0.776065</td>
      <td>0.005165</td>
    </tr>
    <tr>
      <th>Volume USDT</th>
      <td>-0.032387</td>
      <td>-0.031530</td>
      <td>-0.033618</td>
      <td>-0.032558</td>
      <td>1.000000</td>
      <td>0.866663</td>
      <td>0.029773</td>
      <td>-0.032532</td>
      <td>-0.032497</td>
      <td>-0.032473</td>
      <td>...</td>
      <td>-0.032548</td>
      <td>-0.032569</td>
      <td>-0.017288</td>
      <td>-0.021838</td>
      <td>-0.008562</td>
      <td>NaN</td>
      <td>-0.046253</td>
      <td>-0.053967</td>
      <td>-0.053793</td>
      <td>0.001667</td>
    </tr>
    <tr>
      <th>tradecount</th>
      <td>0.226796</td>
      <td>0.227848</td>
      <td>0.225387</td>
      <td>0.226667</td>
      <td>0.866663</td>
      <td>1.000000</td>
      <td>0.046588</td>
      <td>0.226712</td>
      <td>0.226765</td>
      <td>0.226803</td>
      <td>...</td>
      <td>0.226702</td>
      <td>0.226711</td>
      <td>-0.026493</td>
      <td>-0.028188</td>
      <td>0.006481</td>
      <td>NaN</td>
      <td>-0.058182</td>
      <td>0.207535</td>
      <td>0.210692</td>
      <td>0.006456</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.004450</td>
      <td>0.004504</td>
      <td>0.004397</td>
      <td>0.004456</td>
      <td>0.029773</td>
      <td>0.046588</td>
      <td>1.000000</td>
      <td>0.004467</td>
      <td>0.004489</td>
      <td>0.004522</td>
      <td>...</td>
      <td>0.004443</td>
      <td>0.004276</td>
      <td>-0.026079</td>
      <td>-0.022928</td>
      <td>0.005513</td>
      <td>NaN</td>
      <td>0.000326</td>
      <td>-0.000047</td>
      <td>-0.000039</td>
      <td>0.000590</td>
    </tr>
    <tr>
      <th>ema_5</th>
      <td>0.999993</td>
      <td>0.999995</td>
      <td>0.999994</td>
      <td>0.999997</td>
      <td>-0.032532</td>
      <td>0.226712</td>
      <td>0.004467</td>
      <td>1.000000</td>
      <td>0.999996</td>
      <td>0.999985</td>
      <td>...</td>
      <td>0.999995</td>
      <td>0.999784</td>
      <td>-0.020063</td>
      <td>-0.009915</td>
      <td>0.017184</td>
      <td>NaN</td>
      <td>-0.009435</td>
      <td>0.768375</td>
      <td>0.776065</td>
      <td>0.005175</td>
    </tr>
    <tr>
      <th>ema_15</th>
      <td>0.999985</td>
      <td>0.999986</td>
      <td>0.999986</td>
      <td>0.999988</td>
      <td>-0.032497</td>
      <td>0.226765</td>
      <td>0.004489</td>
      <td>0.999996</td>
      <td>1.000000</td>
      <td>0.999996</td>
      <td>...</td>
      <td>0.999987</td>
      <td>0.999788</td>
      <td>-0.022033</td>
      <td>-0.011771</td>
      <td>0.019608</td>
      <td>NaN</td>
      <td>-0.009462</td>
      <td>0.768372</td>
      <td>0.776064</td>
      <td>0.005175</td>
    </tr>
    <tr>
      <th>ema_30</th>
      <td>0.999973</td>
      <td>0.999974</td>
      <td>0.999973</td>
      <td>0.999976</td>
      <td>-0.032473</td>
      <td>0.226803</td>
      <td>0.004522</td>
      <td>0.999985</td>
      <td>0.999996</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.999974</td>
      <td>0.999783</td>
      <td>-0.023683</td>
      <td>-0.013107</td>
      <td>0.020885</td>
      <td>NaN</td>
      <td>-0.009499</td>
      <td>0.768366</td>
      <td>0.776062</td>
      <td>0.005173</td>
    </tr>
    <tr>
      <th>ema_60</th>
      <td>0.999949</td>
      <td>0.999951</td>
      <td>0.999950</td>
      <td>0.999953</td>
      <td>-0.032452</td>
      <td>0.226846</td>
      <td>0.004594</td>
      <td>0.999962</td>
      <td>0.999979</td>
      <td>0.999992</td>
      <td>...</td>
      <td>0.999950</td>
      <td>0.999765</td>
      <td>-0.025284</td>
      <td>-0.014178</td>
      <td>0.021658</td>
      <td>NaN</td>
      <td>-0.009567</td>
      <td>0.768354</td>
      <td>0.776057</td>
      <td>0.005174</td>
    </tr>
    <tr>
      <th>ema_100</th>
      <td>0.999920</td>
      <td>0.999921</td>
      <td>0.999920</td>
      <td>0.999923</td>
      <td>-0.032436</td>
      <td>0.226894</td>
      <td>0.004691</td>
      <td>0.999933</td>
      <td>0.999952</td>
      <td>0.999971</td>
      <td>...</td>
      <td>0.999920</td>
      <td>0.999738</td>
      <td>-0.026244</td>
      <td>-0.014723</td>
      <td>0.021946</td>
      <td>NaN</td>
      <td>-0.009649</td>
      <td>0.768336</td>
      <td>0.776048</td>
      <td>0.005174</td>
    </tr>
    <tr>
      <th>ema_200</th>
      <td>0.999849</td>
      <td>0.999851</td>
      <td>0.999849</td>
      <td>0.999852</td>
      <td>-0.032412</td>
      <td>0.226998</td>
      <td>0.004863</td>
      <td>0.999862</td>
      <td>0.999883</td>
      <td>0.999909</td>
      <td>...</td>
      <td>0.999849</td>
      <td>0.999670</td>
      <td>-0.027178</td>
      <td>-0.015206</td>
      <td>0.022097</td>
      <td>NaN</td>
      <td>-0.009834</td>
      <td>0.768290</td>
      <td>0.776025</td>
      <td>0.005180</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>0.999984</td>
      <td>0.999986</td>
      <td>0.999985</td>
      <td>0.999988</td>
      <td>-0.032507</td>
      <td>0.226757</td>
      <td>0.004457</td>
      <td>0.999995</td>
      <td>0.999996</td>
      <td>0.999988</td>
      <td>...</td>
      <td>0.999992</td>
      <td>0.999813</td>
      <td>-0.021283</td>
      <td>-0.011124</td>
      <td>0.019033</td>
      <td>NaN</td>
      <td>-0.009449</td>
      <td>0.768391</td>
      <td>0.776081</td>
      <td>0.005178</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>-0.030549</td>
      <td>-0.030541</td>
      <td>-0.030483</td>
      <td>-0.030491</td>
      <td>-0.007983</td>
      <td>-0.022193</td>
      <td>-0.010388</td>
      <td>-0.031206</td>
      <td>-0.033726</td>
      <td>-0.036555</td>
      <td>...</td>
      <td>-0.030219</td>
      <td>-0.033411</td>
      <td>0.609492</td>
      <td>0.514483</td>
      <td>-0.530473</td>
      <td>NaN</td>
      <td>0.012439</td>
      <td>-0.024720</td>
      <td>-0.026146</td>
      <td>0.000584</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.000345</td>
      <td>0.000366</td>
      <td>0.000425</td>
      <td>0.000410</td>
      <td>-0.009803</td>
      <td>-0.012996</td>
      <td>-0.000204</td>
      <td>-0.001051</td>
      <td>-0.003362</td>
      <td>-0.003976</td>
      <td>...</td>
      <td>0.000385</td>
      <td>-0.003173</td>
      <td>0.439306</td>
      <td>0.489270</td>
      <td>-0.805188</td>
      <td>NaN</td>
      <td>0.001481</td>
      <td>-0.000007</td>
      <td>-0.000062</td>
      <td>0.000938</td>
    </tr>
    <tr>
      <th>MACD_Hist</th>
      <td>-0.032726</td>
      <td>-0.032724</td>
      <td>-0.032681</td>
      <td>-0.032685</td>
      <td>-0.005272</td>
      <td>-0.019370</td>
      <td>-0.011025</td>
      <td>-0.032966</td>
      <td>-0.034889</td>
      <td>-0.037701</td>
      <td>...</td>
      <td>-0.032387</td>
      <td>-0.034604</td>
      <td>0.504693</td>
      <td>0.386944</td>
      <td>-0.299757</td>
      <td>NaN</td>
      <td>0.012770</td>
      <td>-0.026365</td>
      <td>-0.027868</td>
      <td>0.000313</td>
    </tr>
    <tr>
      <th>ATR</th>
      <td>0.547692</td>
      <td>0.548555</td>
      <td>0.546613</td>
      <td>0.547576</td>
      <td>0.324890</td>
      <td>0.561074</td>
      <td>0.051452</td>
      <td>0.547385</td>
      <td>0.547086</td>
      <td>0.546848</td>
      <td>...</td>
      <td>0.547519</td>
      <td>0.546932</td>
      <td>0.015087</td>
      <td>0.015198</td>
      <td>-0.087168</td>
      <td>NaN</td>
      <td>-0.051029</td>
      <td>0.321133</td>
      <td>0.328615</td>
      <td>0.007415</td>
    </tr>
    <tr>
      <th>HMA</th>
      <td>0.999990</td>
      <td>0.999991</td>
      <td>0.999991</td>
      <td>0.999994</td>
      <td>-0.032548</td>
      <td>0.226702</td>
      <td>0.004443</td>
      <td>0.999995</td>
      <td>0.999987</td>
      <td>0.999974</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.999793</td>
      <td>-0.019285</td>
      <td>-0.009170</td>
      <td>0.015836</td>
      <td>NaN</td>
      <td>-0.009425</td>
      <td>0.768390</td>
      <td>0.776077</td>
      <td>0.005178</td>
    </tr>
    <tr>
      <th>KAMA</th>
      <td>0.999773</td>
      <td>0.999775</td>
      <td>0.999774</td>
      <td>0.999777</td>
      <td>-0.032569</td>
      <td>0.226711</td>
      <td>0.004276</td>
      <td>0.999784</td>
      <td>0.999788</td>
      <td>0.999783</td>
      <td>...</td>
      <td>0.999793</td>
      <td>1.000000</td>
      <td>-0.021926</td>
      <td>-0.011845</td>
      <td>0.019522</td>
      <td>NaN</td>
      <td>-0.009454</td>
      <td>0.768463</td>
      <td>0.776138</td>
      <td>0.005188</td>
    </tr>
    <tr>
      <th>CMO</th>
      <td>-0.018944</td>
      <td>-0.018894</td>
      <td>-0.018833</td>
      <td>-0.018824</td>
      <td>-0.017288</td>
      <td>-0.026493</td>
      <td>-0.026079</td>
      <td>-0.020063</td>
      <td>-0.022033</td>
      <td>-0.023683</td>
      <td>...</td>
      <td>-0.019285</td>
      <td>-0.021926</td>
      <td>1.000000</td>
      <td>0.932326</td>
      <td>-0.532727</td>
      <td>NaN</td>
      <td>0.012789</td>
      <td>-0.018743</td>
      <td>-0.019318</td>
      <td>-0.000761</td>
    </tr>
    <tr>
      <th>Z-Score</th>
      <td>-0.008744</td>
      <td>-0.008707</td>
      <td>-0.008625</td>
      <td>-0.008633</td>
      <td>-0.021838</td>
      <td>-0.028188</td>
      <td>-0.022928</td>
      <td>-0.009915</td>
      <td>-0.011771</td>
      <td>-0.013107</td>
      <td>...</td>
      <td>-0.009170</td>
      <td>-0.011845</td>
      <td>0.932326</td>
      <td>1.000000</td>
      <td>-0.528827</td>
      <td>NaN</td>
      <td>0.008378</td>
      <td>-0.007860</td>
      <td>-0.008185</td>
      <td>-0.000373</td>
    </tr>
    <tr>
      <th>QStick</th>
      <td>0.014683</td>
      <td>0.015149</td>
      <td>0.015159</td>
      <td>0.015534</td>
      <td>-0.008562</td>
      <td>0.006481</td>
      <td>0.005513</td>
      <td>0.017184</td>
      <td>0.019608</td>
      <td>0.020885</td>
      <td>...</td>
      <td>0.015836</td>
      <td>0.019522</td>
      <td>-0.532727</td>
      <td>-0.528827</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-0.007190</td>
      <td>0.011479</td>
      <td>0.012231</td>
      <td>0.000949</td>
    </tr>
    <tr>
      <th>year</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>-0.009416</td>
      <td>-0.009486</td>
      <td>-0.009353</td>
      <td>-0.009423</td>
      <td>-0.046253</td>
      <td>-0.058182</td>
      <td>0.000326</td>
      <td>-0.009435</td>
      <td>-0.009462</td>
      <td>-0.009499</td>
      <td>...</td>
      <td>-0.009425</td>
      <td>-0.009454</td>
      <td>0.012789</td>
      <td>0.008378</td>
      <td>-0.007190</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.005376</td>
      <td>-0.024741</td>
      <td>-0.002279</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.768378</td>
      <td>0.768246</td>
      <td>0.768516</td>
      <td>0.768376</td>
      <td>-0.053967</td>
      <td>0.207535</td>
      <td>-0.000047</td>
      <td>0.768375</td>
      <td>0.768372</td>
      <td>0.768366</td>
      <td>...</td>
      <td>0.768390</td>
      <td>0.768463</td>
      <td>-0.018743</td>
      <td>-0.007860</td>
      <td>0.011479</td>
      <td>NaN</td>
      <td>-0.005376</td>
      <td>1.000000</td>
      <td>0.996326</td>
      <td>0.003042</td>
    </tr>
    <tr>
      <th>week_of_year</th>
      <td>0.776067</td>
      <td>0.775940</td>
      <td>0.776200</td>
      <td>0.776065</td>
      <td>-0.053793</td>
      <td>0.210692</td>
      <td>-0.000039</td>
      <td>0.776065</td>
      <td>0.776064</td>
      <td>0.776062</td>
      <td>...</td>
      <td>0.776077</td>
      <td>0.776138</td>
      <td>-0.019318</td>
      <td>-0.008185</td>
      <td>0.012231</td>
      <td>NaN</td>
      <td>-0.024741</td>
      <td>0.996326</td>
      <td>1.000000</td>
      <td>0.003213</td>
    </tr>
    <tr>
      <th>time_delta</th>
      <td>0.005143</td>
      <td>0.005172</td>
      <td>0.005118</td>
      <td>0.005165</td>
      <td>0.001667</td>
      <td>0.006456</td>
      <td>0.000590</td>
      <td>0.005175</td>
      <td>0.005175</td>
      <td>0.005173</td>
      <td>...</td>
      <td>0.005178</td>
      <td>0.005188</td>
      <td>-0.000761</td>
      <td>-0.000373</td>
      <td>0.000949</td>
      <td>NaN</td>
      <td>-0.002279</td>
      <td>0.003042</td>
      <td>0.003213</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 28 columns</p>
</div>



    
![png](data_analysis_files/data_analysis_23_5.png)
    


    
    Correlation Matrix (Year 2021):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>hour</th>
      <th>ema_5</th>
      <th>ema_15</th>
      <th>ema_30</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>open</th>
      <td>1.000000</td>
      <td>0.999988</td>
      <td>0.999988</td>
      <td>0.999982</td>
      <td>0.000192</td>
      <td>-0.025024</td>
      <td>0.001809</td>
      <td>0.999957</td>
      <td>0.999880</td>
      <td>0.999766</td>
      <td>...</td>
      <td>0.999952</td>
      <td>0.999919</td>
      <td>0.019363</td>
      <td>0.021310</td>
      <td>-0.010403</td>
      <td>NaN</td>
      <td>-0.006700</td>
      <td>0.317055</td>
      <td>0.264965</td>
      <td>0.000124</td>
    </tr>
    <tr>
      <th>high</th>
      <td>0.999988</td>
      <td>1.000000</td>
      <td>0.999979</td>
      <td>0.999990</td>
      <td>0.002528</td>
      <td>-0.022663</td>
      <td>0.001869</td>
      <td>0.999964</td>
      <td>0.999888</td>
      <td>0.999775</td>
      <td>...</td>
      <td>0.999960</td>
      <td>0.999927</td>
      <td>0.019417</td>
      <td>0.021347</td>
      <td>-0.009338</td>
      <td>NaN</td>
      <td>-0.006935</td>
      <td>0.316426</td>
      <td>0.264337</td>
      <td>0.000176</td>
    </tr>
    <tr>
      <th>low</th>
      <td>0.999988</td>
      <td>0.999979</td>
      <td>1.000000</td>
      <td>0.999990</td>
      <td>-0.002301</td>
      <td>-0.027559</td>
      <td>0.001752</td>
      <td>0.999963</td>
      <td>0.999885</td>
      <td>0.999771</td>
      <td>...</td>
      <td>0.999959</td>
      <td>0.999925</td>
      <td>0.019551</td>
      <td>0.021509</td>
      <td>-0.009536</td>
      <td>NaN</td>
      <td>-0.006456</td>
      <td>0.317687</td>
      <td>0.265592</td>
      <td>0.000110</td>
    </tr>
    <tr>
      <th>close</th>
      <td>0.999982</td>
      <td>0.999990</td>
      <td>0.999990</td>
      <td>1.000000</td>
      <td>0.000114</td>
      <td>-0.025123</td>
      <td>0.001814</td>
      <td>0.999974</td>
      <td>0.999897</td>
      <td>0.999782</td>
      <td>...</td>
      <td>0.999970</td>
      <td>0.999936</td>
      <td>0.019557</td>
      <td>0.021492</td>
      <td>-0.008598</td>
      <td>NaN</td>
      <td>-0.006696</td>
      <td>0.317049</td>
      <td>0.264962</td>
      <td>0.000171</td>
    </tr>
    <tr>
      <th>Volume USDT</th>
      <td>0.000192</td>
      <td>0.002528</td>
      <td>-0.002301</td>
      <td>0.000114</td>
      <td>1.000000</td>
      <td>0.906750</td>
      <td>-0.001297</td>
      <td>0.000270</td>
      <td>0.000508</td>
      <td>0.000729</td>
      <td>...</td>
      <td>0.000180</td>
      <td>0.000329</td>
      <td>-0.016870</td>
      <td>-0.018544</td>
      <td>0.023292</td>
      <td>NaN</td>
      <td>-0.075613</td>
      <td>-0.170979</td>
      <td>-0.167892</td>
      <td>0.004730</td>
    </tr>
    <tr>
      <th>tradecount</th>
      <td>-0.025024</td>
      <td>-0.022663</td>
      <td>-0.027559</td>
      <td>-0.025123</td>
      <td>0.906750</td>
      <td>1.000000</td>
      <td>0.023800</td>
      <td>-0.024953</td>
      <td>-0.024697</td>
      <td>-0.024438</td>
      <td>...</td>
      <td>-0.025043</td>
      <td>-0.024894</td>
      <td>-0.021269</td>
      <td>-0.022427</td>
      <td>0.023145</td>
      <td>NaN</td>
      <td>-0.067906</td>
      <td>-0.209327</td>
      <td>-0.205394</td>
      <td>0.004895</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.001809</td>
      <td>0.001869</td>
      <td>0.001752</td>
      <td>0.001814</td>
      <td>-0.001297</td>
      <td>0.023800</td>
      <td>1.000000</td>
      <td>0.001796</td>
      <td>0.001758</td>
      <td>0.001721</td>
      <td>...</td>
      <td>0.001814</td>
      <td>0.001768</td>
      <td>-0.012949</td>
      <td>-0.013232</td>
      <td>0.001777</td>
      <td>NaN</td>
      <td>0.000745</td>
      <td>-0.000698</td>
      <td>-0.000740</td>
      <td>-0.002290</td>
    </tr>
    <tr>
      <th>ema_5</th>
      <td>0.999957</td>
      <td>0.999964</td>
      <td>0.999963</td>
      <td>0.999974</td>
      <td>0.000270</td>
      <td>-0.024953</td>
      <td>0.001796</td>
      <td>1.000000</td>
      <td>0.999960</td>
      <td>0.999862</td>
      <td>...</td>
      <td>0.999985</td>
      <td>0.999976</td>
      <td>0.016324</td>
      <td>0.018211</td>
      <td>-0.005040</td>
      <td>NaN</td>
      <td>-0.006696</td>
      <td>0.317018</td>
      <td>0.264932</td>
      <td>0.000206</td>
    </tr>
    <tr>
      <th>ema_15</th>
      <td>0.999880</td>
      <td>0.999888</td>
      <td>0.999885</td>
      <td>0.999897</td>
      <td>0.000508</td>
      <td>-0.024697</td>
      <td>0.001758</td>
      <td>0.999960</td>
      <td>1.000000</td>
      <td>0.999962</td>
      <td>...</td>
      <td>0.999907</td>
      <td>0.999964</td>
      <td>0.011002</td>
      <td>0.013326</td>
      <td>0.000230</td>
      <td>NaN</td>
      <td>-0.006699</td>
      <td>0.316933</td>
      <td>0.264852</td>
      <td>0.000225</td>
    </tr>
    <tr>
      <th>ema_30</th>
      <td>0.999766</td>
      <td>0.999775</td>
      <td>0.999771</td>
      <td>0.999782</td>
      <td>0.000729</td>
      <td>-0.024438</td>
      <td>0.001721</td>
      <td>0.999862</td>
      <td>0.999962</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.999786</td>
      <td>0.999886</td>
      <td>0.006449</td>
      <td>0.009775</td>
      <td>0.003015</td>
      <td>NaN</td>
      <td>-0.006697</td>
      <td>0.316805</td>
      <td>0.264731</td>
      <td>0.000232</td>
    </tr>
    <tr>
      <th>ema_60</th>
      <td>0.999543</td>
      <td>0.999553</td>
      <td>0.999546</td>
      <td>0.999558</td>
      <td>0.001006</td>
      <td>-0.024081</td>
      <td>0.001665</td>
      <td>0.999647</td>
      <td>0.999797</td>
      <td>0.999925</td>
      <td>...</td>
      <td>0.999557</td>
      <td>0.999689</td>
      <td>0.001866</td>
      <td>0.006876</td>
      <td>0.004725</td>
      <td>NaN</td>
      <td>-0.006678</td>
      <td>0.316546</td>
      <td>0.264486</td>
      <td>0.000231</td>
    </tr>
    <tr>
      <th>ema_100</th>
      <td>0.999248</td>
      <td>0.999259</td>
      <td>0.999250</td>
      <td>0.999263</td>
      <td>0.001238</td>
      <td>-0.023768</td>
      <td>0.001592</td>
      <td>0.999355</td>
      <td>0.999532</td>
      <td>0.999720</td>
      <td>...</td>
      <td>0.999259</td>
      <td>0.999408</td>
      <td>-0.001032</td>
      <td>0.005345</td>
      <td>0.005440</td>
      <td>NaN</td>
      <td>-0.006641</td>
      <td>0.316200</td>
      <td>0.264159</td>
      <td>0.000227</td>
    </tr>
    <tr>
      <th>ema_200</th>
      <td>0.998517</td>
      <td>0.998530</td>
      <td>0.998518</td>
      <td>0.998532</td>
      <td>0.001568</td>
      <td>-0.023332</td>
      <td>0.001291</td>
      <td>0.998626</td>
      <td>0.998826</td>
      <td>0.999076</td>
      <td>...</td>
      <td>0.998526</td>
      <td>0.998689</td>
      <td>-0.003967</td>
      <td>0.003959</td>
      <td>0.005931</td>
      <td>NaN</td>
      <td>-0.006558</td>
      <td>0.315336</td>
      <td>0.263342</td>
      <td>0.000230</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>0.999906</td>
      <td>0.999913</td>
      <td>0.999912</td>
      <td>0.999922</td>
      <td>0.000418</td>
      <td>-0.024797</td>
      <td>0.001773</td>
      <td>0.999981</td>
      <td>0.999991</td>
      <td>0.999918</td>
      <td>...</td>
      <td>0.999938</td>
      <td>0.999975</td>
      <td>0.013079</td>
      <td>0.015057</td>
      <td>-0.001051</td>
      <td>NaN</td>
      <td>-0.006697</td>
      <td>0.316974</td>
      <td>0.264891</td>
      <td>0.000222</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>0.014160</td>
      <td>0.014094</td>
      <td>0.014295</td>
      <td>0.014233</td>
      <td>-0.025708</td>
      <td>-0.029466</td>
      <td>0.004260</td>
      <td>0.012050</td>
      <td>0.004442</td>
      <td>-0.004196</td>
      <td>...</td>
      <td>0.014988</td>
      <td>0.009449</td>
      <td>0.549711</td>
      <td>0.448337</td>
      <td>-0.380258</td>
      <td>NaN</td>
      <td>-0.000002</td>
      <td>0.014146</td>
      <td>0.013237</td>
      <td>-0.001012</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.000675</td>
      <td>0.000693</td>
      <td>0.000764</td>
      <td>0.000730</td>
      <td>-0.014734</td>
      <td>-0.013392</td>
      <td>0.002192</td>
      <td>-0.003593</td>
      <td>-0.010460</td>
      <td>-0.012281</td>
      <td>...</td>
      <td>0.000620</td>
      <td>-0.005961</td>
      <td>0.389744</td>
      <td>0.430672</td>
      <td>-0.588401</td>
      <td>NaN</td>
      <td>0.000620</td>
      <td>-0.000042</td>
      <td>-0.000036</td>
      <td>-0.002404</td>
    </tr>
    <tr>
      <th>MACD_Hist</th>
      <td>0.014834</td>
      <td>0.014758</td>
      <td>0.014950</td>
      <td>0.014894</td>
      <td>-0.022577</td>
      <td>-0.027004</td>
      <td>0.003821</td>
      <td>0.013968</td>
      <td>0.008094</td>
      <td>-0.000500</td>
      <td>...</td>
      <td>0.015733</td>
      <td>0.011967</td>
      <td>0.458668</td>
      <td>0.337707</td>
      <td>-0.214477</td>
      <td>NaN</td>
      <td>-0.000202</td>
      <td>0.015050</td>
      <td>0.014083</td>
      <td>-0.000301</td>
    </tr>
    <tr>
      <th>ATR</th>
      <td>0.019187</td>
      <td>0.021328</td>
      <td>0.016905</td>
      <td>0.018968</td>
      <td>0.562301</td>
      <td>0.580208</td>
      <td>0.046267</td>
      <td>0.018018</td>
      <td>0.016555</td>
      <td>0.015480</td>
      <td>...</td>
      <td>0.018620</td>
      <td>0.017601</td>
      <td>0.023374</td>
      <td>0.017913</td>
      <td>-0.073432</td>
      <td>NaN</td>
      <td>-0.101848</td>
      <td>-0.263711</td>
      <td>-0.262694</td>
      <td>0.002489</td>
    </tr>
    <tr>
      <th>HMA</th>
      <td>0.999952</td>
      <td>0.999960</td>
      <td>0.999959</td>
      <td>0.999970</td>
      <td>0.000180</td>
      <td>-0.025043</td>
      <td>0.001814</td>
      <td>0.999985</td>
      <td>0.999907</td>
      <td>0.999786</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.999946</td>
      <td>0.018433</td>
      <td>0.020191</td>
      <td>-0.007942</td>
      <td>NaN</td>
      <td>-0.006695</td>
      <td>0.317036</td>
      <td>0.264950</td>
      <td>0.000204</td>
    </tr>
    <tr>
      <th>KAMA</th>
      <td>0.999919</td>
      <td>0.999927</td>
      <td>0.999925</td>
      <td>0.999936</td>
      <td>0.000329</td>
      <td>-0.024894</td>
      <td>0.001768</td>
      <td>0.999976</td>
      <td>0.999964</td>
      <td>0.999886</td>
      <td>...</td>
      <td>0.999946</td>
      <td>1.000000</td>
      <td>0.011798</td>
      <td>0.013446</td>
      <td>-0.000181</td>
      <td>NaN</td>
      <td>-0.006694</td>
      <td>0.317121</td>
      <td>0.265021</td>
      <td>0.000216</td>
    </tr>
    <tr>
      <th>CMO</th>
      <td>0.019363</td>
      <td>0.019417</td>
      <td>0.019551</td>
      <td>0.019557</td>
      <td>-0.016870</td>
      <td>-0.021269</td>
      <td>-0.012949</td>
      <td>0.016324</td>
      <td>0.011002</td>
      <td>0.006449</td>
      <td>...</td>
      <td>0.018433</td>
      <td>0.011798</td>
      <td>1.000000</td>
      <td>0.929958</td>
      <td>-0.650321</td>
      <td>NaN</td>
      <td>-0.005353</td>
      <td>0.017069</td>
      <td>0.013429</td>
      <td>-0.003546</td>
    </tr>
    <tr>
      <th>Z-Score</th>
      <td>0.021310</td>
      <td>0.021347</td>
      <td>0.021509</td>
      <td>0.021492</td>
      <td>-0.018544</td>
      <td>-0.022427</td>
      <td>-0.013232</td>
      <td>0.018211</td>
      <td>0.013326</td>
      <td>0.009775</td>
      <td>...</td>
      <td>0.020191</td>
      <td>0.013446</td>
      <td>0.929958</td>
      <td>1.000000</td>
      <td>-0.640370</td>
      <td>NaN</td>
      <td>-0.004430</td>
      <td>0.012972</td>
      <td>0.009945</td>
      <td>-0.003214</td>
    </tr>
    <tr>
      <th>QStick</th>
      <td>-0.010403</td>
      <td>-0.009338</td>
      <td>-0.009536</td>
      <td>-0.008598</td>
      <td>0.023292</td>
      <td>0.023145</td>
      <td>0.001777</td>
      <td>-0.005040</td>
      <td>0.000230</td>
      <td>0.003015</td>
      <td>...</td>
      <td>-0.007942</td>
      <td>-0.000181</td>
      <td>-0.650321</td>
      <td>-0.640370</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.002044</td>
      <td>-0.003383</td>
      <td>-0.002348</td>
      <td>0.005856</td>
    </tr>
    <tr>
      <th>year</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>-0.006700</td>
      <td>-0.006935</td>
      <td>-0.006456</td>
      <td>-0.006696</td>
      <td>-0.075613</td>
      <td>-0.067906</td>
      <td>0.000745</td>
      <td>-0.006696</td>
      <td>-0.006699</td>
      <td>-0.006697</td>
      <td>...</td>
      <td>-0.006695</td>
      <td>-0.006694</td>
      <td>-0.005353</td>
      <td>-0.004430</td>
      <td>0.002044</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.008041</td>
      <td>0.003437</td>
      <td>0.001345</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.317055</td>
      <td>0.316426</td>
      <td>0.317687</td>
      <td>0.317049</td>
      <td>-0.170979</td>
      <td>-0.209327</td>
      <td>-0.000698</td>
      <td>0.317018</td>
      <td>0.316933</td>
      <td>0.316805</td>
      <td>...</td>
      <td>0.317036</td>
      <td>0.317121</td>
      <td>0.017069</td>
      <td>0.012972</td>
      <td>-0.003383</td>
      <td>NaN</td>
      <td>-0.008041</td>
      <td>1.000000</td>
      <td>0.949678</td>
      <td>-0.000954</td>
    </tr>
    <tr>
      <th>week_of_year</th>
      <td>0.264965</td>
      <td>0.264337</td>
      <td>0.265592</td>
      <td>0.264962</td>
      <td>-0.167892</td>
      <td>-0.205394</td>
      <td>-0.000740</td>
      <td>0.264932</td>
      <td>0.264852</td>
      <td>0.264731</td>
      <td>...</td>
      <td>0.264950</td>
      <td>0.265021</td>
      <td>0.013429</td>
      <td>0.009945</td>
      <td>-0.002348</td>
      <td>NaN</td>
      <td>0.003437</td>
      <td>0.949678</td>
      <td>1.000000</td>
      <td>-0.000983</td>
    </tr>
    <tr>
      <th>time_delta</th>
      <td>0.000124</td>
      <td>0.000176</td>
      <td>0.000110</td>
      <td>0.000171</td>
      <td>0.004730</td>
      <td>0.004895</td>
      <td>-0.002290</td>
      <td>0.000206</td>
      <td>0.000225</td>
      <td>0.000232</td>
      <td>...</td>
      <td>0.000204</td>
      <td>0.000216</td>
      <td>-0.003546</td>
      <td>-0.003214</td>
      <td>0.005856</td>
      <td>NaN</td>
      <td>0.001345</td>
      <td>-0.000954</td>
      <td>-0.000983</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 28 columns</p>
</div>



    
![png](data_analysis_files/data_analysis_23_8.png)
    


    
    Correlation Matrix (Year 2022):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>hour</th>
      <th>ema_5</th>
      <th>ema_15</th>
      <th>ema_30</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>open</th>
      <td>1.000000</td>
      <td>0.999997</td>
      <td>0.999998</td>
      <td>0.999997</td>
      <td>-0.269610</td>
      <td>-0.501310</td>
      <td>-1.635535e-03</td>
      <td>0.999993</td>
      <td>0.999981</td>
      <td>0.999963</td>
      <td>...</td>
      <td>0.999992</td>
      <td>0.999984</td>
      <td>0.008818</td>
      <td>0.005008</td>
      <td>-0.006230</td>
      <td>NaN</td>
      <td>-2.980676e-03</td>
      <td>-9.079611e-01</td>
      <td>-8.785204e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>high</th>
      <td>0.999997</td>
      <td>1.000000</td>
      <td>0.999996</td>
      <td>0.999998</td>
      <td>-0.268769</td>
      <td>-0.500724</td>
      <td>-1.501030e-03</td>
      <td>0.999994</td>
      <td>0.999982</td>
      <td>0.999964</td>
      <td>...</td>
      <td>0.999993</td>
      <td>0.999986</td>
      <td>0.008872</td>
      <td>0.005061</td>
      <td>-0.005776</td>
      <td>NaN</td>
      <td>-3.176726e-03</td>
      <td>-9.081237e-01</td>
      <td>-8.786872e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>low</th>
      <td>0.999998</td>
      <td>0.999996</td>
      <td>1.000000</td>
      <td>0.999998</td>
      <td>-0.270468</td>
      <td>-0.501906</td>
      <td>-1.764512e-03</td>
      <td>0.999994</td>
      <td>0.999982</td>
      <td>0.999964</td>
      <td>...</td>
      <td>0.999993</td>
      <td>0.999986</td>
      <td>0.008850</td>
      <td>0.005037</td>
      <td>-0.005794</td>
      <td>NaN</td>
      <td>-2.794562e-03</td>
      <td>-9.078096e-01</td>
      <td>-8.783628e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>close</th>
      <td>0.999997</td>
      <td>0.999998</td>
      <td>0.999998</td>
      <td>1.000000</td>
      <td>-0.269620</td>
      <td>-0.501314</td>
      <td>-1.635735e-03</td>
      <td>0.999996</td>
      <td>0.999984</td>
      <td>0.999966</td>
      <td>...</td>
      <td>0.999995</td>
      <td>0.999988</td>
      <td>0.008891</td>
      <td>0.005075</td>
      <td>-0.005412</td>
      <td>NaN</td>
      <td>-2.983344e-03</td>
      <td>-9.079610e-01</td>
      <td>-8.785190e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Volume USDT</th>
      <td>-0.269610</td>
      <td>-0.268769</td>
      <td>-0.270468</td>
      <td>-0.269620</td>
      <td>1.000000</td>
      <td>0.858917</td>
      <td>3.668407e-02</td>
      <td>-0.269636</td>
      <td>-0.269661</td>
      <td>-0.269680</td>
      <td>...</td>
      <td>-0.269625</td>
      <td>-0.269640</td>
      <td>0.001611</td>
      <td>0.002768</td>
      <td>-0.005404</td>
      <td>NaN</td>
      <td>-1.278438e-01</td>
      <td>2.796406e-01</td>
      <td>2.706452e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tradecount</th>
      <td>-0.501310</td>
      <td>-0.500724</td>
      <td>-0.501906</td>
      <td>-0.501314</td>
      <td>0.858917</td>
      <td>1.000000</td>
      <td>1.856723e-02</td>
      <td>-0.501329</td>
      <td>-0.501352</td>
      <td>-0.501370</td>
      <td>...</td>
      <td>-0.501317</td>
      <td>-0.501343</td>
      <td>-0.000189</td>
      <td>0.001531</td>
      <td>-0.003441</td>
      <td>NaN</td>
      <td>-1.167457e-01</td>
      <td>5.247284e-01</td>
      <td>5.151569e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>-0.001636</td>
      <td>-0.001501</td>
      <td>-0.001765</td>
      <td>-0.001636</td>
      <td>0.036684</td>
      <td>0.018567</td>
      <td>1.000000e+00</td>
      <td>-0.001630</td>
      <td>-0.001612</td>
      <td>-0.001586</td>
      <td>...</td>
      <td>-0.001634</td>
      <td>-0.001631</td>
      <td>0.003387</td>
      <td>0.004049</td>
      <td>-0.001095</td>
      <td>NaN</td>
      <td>1.074958e-17</td>
      <td>-1.447911e-15</td>
      <td>-1.001371e-15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_5</th>
      <td>0.999993</td>
      <td>0.999994</td>
      <td>0.999994</td>
      <td>0.999996</td>
      <td>-0.269636</td>
      <td>-0.501329</td>
      <td>-1.629686e-03</td>
      <td>1.000000</td>
      <td>0.999994</td>
      <td>0.999979</td>
      <td>...</td>
      <td>0.999998</td>
      <td>0.999995</td>
      <td>0.007559</td>
      <td>0.003713</td>
      <td>-0.003784</td>
      <td>NaN</td>
      <td>-2.984537e-03</td>
      <td>-9.079613e-01</td>
      <td>-8.785160e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_15</th>
      <td>0.999981</td>
      <td>0.999982</td>
      <td>0.999982</td>
      <td>0.999984</td>
      <td>-0.269661</td>
      <td>-0.501352</td>
      <td>-1.612409e-03</td>
      <td>0.999994</td>
      <td>1.000000</td>
      <td>0.999994</td>
      <td>...</td>
      <td>0.999986</td>
      <td>0.999996</td>
      <td>0.005345</td>
      <td>0.001700</td>
      <td>-0.001362</td>
      <td>NaN</td>
      <td>-2.990126e-03</td>
      <td>-9.079596e-01</td>
      <td>-8.785063e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_30</th>
      <td>0.999963</td>
      <td>0.999964</td>
      <td>0.999964</td>
      <td>0.999966</td>
      <td>-0.269680</td>
      <td>-0.501370</td>
      <td>-1.585753e-03</td>
      <td>0.999979</td>
      <td>0.999994</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.999967</td>
      <td>0.999985</td>
      <td>0.003404</td>
      <td>0.000208</td>
      <td>-0.000031</td>
      <td>NaN</td>
      <td>-3.001969e-03</td>
      <td>-9.079569e-01</td>
      <td>-8.784918e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_60</th>
      <td>0.999928</td>
      <td>0.999929</td>
      <td>0.999929</td>
      <td>0.999931</td>
      <td>-0.269715</td>
      <td>-0.501399</td>
      <td>-1.543894e-03</td>
      <td>0.999945</td>
      <td>0.999968</td>
      <td>0.999988</td>
      <td>...</td>
      <td>0.999931</td>
      <td>0.999955</td>
      <td>0.001414</td>
      <td>-0.001043</td>
      <td>0.000855</td>
      <td>NaN</td>
      <td>-3.031009e-03</td>
      <td>-9.079512e-01</td>
      <td>-8.784639e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_100</th>
      <td>0.999880</td>
      <td>0.999881</td>
      <td>0.999881</td>
      <td>0.999883</td>
      <td>-0.269779</td>
      <td>-0.501449</td>
      <td>-1.516580e-03</td>
      <td>0.999898</td>
      <td>0.999925</td>
      <td>0.999955</td>
      <td>...</td>
      <td>0.999883</td>
      <td>0.999911</td>
      <td>0.000170</td>
      <td>-0.001695</td>
      <td>0.001255</td>
      <td>NaN</td>
      <td>-3.073977e-03</td>
      <td>-9.079427e-01</td>
      <td>-8.784279e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ema_200</th>
      <td>0.999761</td>
      <td>0.999763</td>
      <td>0.999763</td>
      <td>0.999765</td>
      <td>-0.269984</td>
      <td>-0.501606</td>
      <td>-1.529337e-03</td>
      <td>0.999780</td>
      <td>0.999811</td>
      <td>0.999851</td>
      <td>...</td>
      <td>0.999764</td>
      <td>0.999795</td>
      <td>-0.001040</td>
      <td>-0.002244</td>
      <td>0.001561</td>
      <td>NaN</td>
      <td>-3.187205e-03</td>
      <td>-9.079191e-01</td>
      <td>-8.783414e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>0.999985</td>
      <td>0.999986</td>
      <td>0.999986</td>
      <td>0.999988</td>
      <td>-0.269651</td>
      <td>-0.501342</td>
      <td>-1.622820e-03</td>
      <td>0.999997</td>
      <td>0.999999</td>
      <td>0.999987</td>
      <td>...</td>
      <td>0.999990</td>
      <td>0.999997</td>
      <td>0.006216</td>
      <td>0.002409</td>
      <td>-0.001952</td>
      <td>NaN</td>
      <td>-2.986442e-03</td>
      <td>-9.079585e-01</td>
      <td>-8.785095e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>0.015495</td>
      <td>0.015508</td>
      <td>0.015504</td>
      <td>0.015513</td>
      <td>0.002930</td>
      <td>0.000116</td>
      <td>-7.223777e-03</td>
      <td>0.014653</td>
      <td>0.011672</td>
      <td>0.008268</td>
      <td>...</td>
      <td>0.015806</td>
      <td>0.013172</td>
      <td>0.592443</td>
      <td>0.475744</td>
      <td>-0.455384</td>
      <td>NaN</td>
      <td>2.923649e-03</td>
      <td>-1.051898e-02</td>
      <td>-1.335992e-02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.000090</td>
      <td>0.000121</td>
      <td>0.000094</td>
      <td>0.000116</td>
      <td>0.005313</td>
      <td>0.004763</td>
      <td>1.983665e-04</td>
      <td>-0.001564</td>
      <td>-0.004236</td>
      <td>-0.004945</td>
      <td>...</td>
      <td>0.000088</td>
      <td>-0.002956</td>
      <td>0.401716</td>
      <td>0.445662</td>
      <td>-0.678728</td>
      <td>NaN</td>
      <td>-1.002483e-03</td>
      <td>-1.590800e-05</td>
      <td>-7.769158e-05</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>MACD_Hist</th>
      <td>0.016419</td>
      <td>0.016423</td>
      <td>0.016427</td>
      <td>0.016430</td>
      <td>0.001419</td>
      <td>-0.001394</td>
      <td>-7.731055e-03</td>
      <td>0.016052</td>
      <td>0.013739</td>
      <td>0.010351</td>
      <td>...</td>
      <td>0.016750</td>
      <td>0.014923</td>
      <td>0.500952</td>
      <td>0.363086</td>
      <td>-0.267261</td>
      <td>NaN</td>
      <td>3.422603e-03</td>
      <td>-1.116062e-02</td>
      <td>-1.415654e-02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ATR</th>
      <td>0.435266</td>
      <td>0.436201</td>
      <td>0.434356</td>
      <td>0.435260</td>
      <td>0.219171</td>
      <td>0.008413</td>
      <td>1.207629e-01</td>
      <td>0.435383</td>
      <td>0.435575</td>
      <td>0.435733</td>
      <td>...</td>
      <td>0.435302</td>
      <td>0.435397</td>
      <td>0.014015</td>
      <td>0.013560</td>
      <td>-0.006166</td>
      <td>NaN</td>
      <td>-1.595105e-01</td>
      <td>-5.262123e-01</td>
      <td>-5.184004e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>HMA</th>
      <td>0.999992</td>
      <td>0.999993</td>
      <td>0.999993</td>
      <td>0.999995</td>
      <td>-0.269625</td>
      <td>-0.501317</td>
      <td>-1.634290e-03</td>
      <td>0.999998</td>
      <td>0.999986</td>
      <td>0.999967</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.999990</td>
      <td>0.008435</td>
      <td>0.004527</td>
      <td>-0.005118</td>
      <td>NaN</td>
      <td>-2.982959e-03</td>
      <td>-9.079576e-01</td>
      <td>-8.785151e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>KAMA</th>
      <td>0.999984</td>
      <td>0.999986</td>
      <td>0.999986</td>
      <td>0.999988</td>
      <td>-0.269640</td>
      <td>-0.501343</td>
      <td>-1.631369e-03</td>
      <td>0.999995</td>
      <td>0.999996</td>
      <td>0.999985</td>
      <td>...</td>
      <td>0.999990</td>
      <td>1.000000</td>
      <td>0.005699</td>
      <td>0.001784</td>
      <td>-0.001666</td>
      <td>NaN</td>
      <td>-3.040341e-03</td>
      <td>-9.079862e-01</td>
      <td>-8.785335e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>CMO</th>
      <td>0.008818</td>
      <td>0.008872</td>
      <td>0.008850</td>
      <td>0.008891</td>
      <td>0.001611</td>
      <td>-0.000189</td>
      <td>3.386966e-03</td>
      <td>0.007559</td>
      <td>0.005345</td>
      <td>0.003404</td>
      <td>...</td>
      <td>0.008435</td>
      <td>0.005699</td>
      <td>1.000000</td>
      <td>0.928003</td>
      <td>-0.596941</td>
      <td>NaN</td>
      <td>1.045545e-02</td>
      <td>-5.339926e-03</td>
      <td>-6.929066e-03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Z-Score</th>
      <td>0.005008</td>
      <td>0.005061</td>
      <td>0.005037</td>
      <td>0.005075</td>
      <td>0.002768</td>
      <td>0.001531</td>
      <td>4.049020e-03</td>
      <td>0.003713</td>
      <td>0.001700</td>
      <td>0.000208</td>
      <td>...</td>
      <td>0.004527</td>
      <td>0.001784</td>
      <td>0.928003</td>
      <td>1.000000</td>
      <td>-0.583927</td>
      <td>NaN</td>
      <td>8.308069e-03</td>
      <td>-1.821170e-03</td>
      <td>-2.514936e-03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>QStick</th>
      <td>-0.006230</td>
      <td>-0.005776</td>
      <td>-0.005794</td>
      <td>-0.005412</td>
      <td>-0.005404</td>
      <td>-0.003441</td>
      <td>-1.094864e-03</td>
      <td>-0.003784</td>
      <td>-0.001362</td>
      <td>-0.000031</td>
      <td>...</td>
      <td>-0.005118</td>
      <td>-0.001666</td>
      <td>-0.596941</td>
      <td>-0.583927</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>-3.694673e-03</td>
      <td>1.988225e-03</td>
      <td>3.423857e-03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>year</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>-0.002981</td>
      <td>-0.003177</td>
      <td>-0.002795</td>
      <td>-0.002983</td>
      <td>-0.127844</td>
      <td>-0.116746</td>
      <td>1.074958e-17</td>
      <td>-0.002985</td>
      <td>-0.002990</td>
      <td>-0.003002</td>
      <td>...</td>
      <td>-0.002983</td>
      <td>-0.003040</td>
      <td>0.010455</td>
      <td>0.008308</td>
      <td>-0.003695</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
      <td>7.739394e-04</td>
      <td>4.630311e-03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>month</th>
      <td>-0.907961</td>
      <td>-0.908124</td>
      <td>-0.907810</td>
      <td>-0.907961</td>
      <td>0.279641</td>
      <td>0.524728</td>
      <td>-1.447911e-15</td>
      <td>-0.907961</td>
      <td>-0.907960</td>
      <td>-0.907957</td>
      <td>...</td>
      <td>-0.907958</td>
      <td>-0.907986</td>
      <td>-0.005340</td>
      <td>-0.001821</td>
      <td>0.001988</td>
      <td>NaN</td>
      <td>7.739394e-04</td>
      <td>1.000000e+00</td>
      <td>9.664959e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>week_of_year</th>
      <td>-0.878520</td>
      <td>-0.878687</td>
      <td>-0.878363</td>
      <td>-0.878519</td>
      <td>0.270645</td>
      <td>0.515157</td>
      <td>-1.001371e-15</td>
      <td>-0.878516</td>
      <td>-0.878506</td>
      <td>-0.878492</td>
      <td>...</td>
      <td>-0.878515</td>
      <td>-0.878533</td>
      <td>-0.006929</td>
      <td>-0.002515</td>
      <td>0.003424</td>
      <td>NaN</td>
      <td>4.630311e-03</td>
      <td>9.664959e-01</td>
      <td>1.000000e+00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>time_delta</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 28 columns</p>
</div>



    
![png](data_analysis_files/data_analysis_23_11.png)
    


    
    Correlation Matrix (Year 2023):
    --------------------------------------------------



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>hour</th>
      <th>ema_5</th>
      <th>ema_15</th>
      <th>ema_30</th>
      <th>...</th>
      <th>HMA</th>
      <th>KAMA</th>
      <th>CMO</th>
      <th>Z-Score</th>
      <th>QStick</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>month</th>
      <th>week_of_year</th>
      <th>time_delta</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>open</th>
      <td>1.000000</td>
      <td>0.999992</td>
      <td>0.999992</td>
      <td>0.999989</td>
      <td>-0.318120</td>
      <td>-0.488418</td>
      <td>0.003511</td>
      <td>0.999956</td>
      <td>0.999855</td>
      <td>0.999702</td>
      <td>...</td>
      <td>0.999951</td>
      <td>0.999937</td>
      <td>0.019921</td>
      <td>0.018638</td>
      <td>-0.013873</td>
      <td>NaN</td>
      <td>0.001351</td>
      <td>0.599839</td>
      <td>0.564870</td>
      <td>0.000475</td>
    </tr>
    <tr>
      <th>high</th>
      <td>0.999992</td>
      <td>1.000000</td>
      <td>0.999984</td>
      <td>0.999993</td>
      <td>-0.316097</td>
      <td>-0.486667</td>
      <td>0.003822</td>
      <td>0.999960</td>
      <td>0.999859</td>
      <td>0.999706</td>
      <td>...</td>
      <td>0.999955</td>
      <td>0.999942</td>
      <td>0.019978</td>
      <td>0.018690</td>
      <td>-0.013057</td>
      <td>NaN</td>
      <td>0.001021</td>
      <td>0.599359</td>
      <td>0.564390</td>
      <td>0.000473</td>
    </tr>
    <tr>
      <th>low</th>
      <td>0.999992</td>
      <td>0.999984</td>
      <td>1.000000</td>
      <td>0.999993</td>
      <td>-0.320212</td>
      <td>-0.490183</td>
      <td>0.003199</td>
      <td>0.999960</td>
      <td>0.999859</td>
      <td>0.999705</td>
      <td>...</td>
      <td>0.999955</td>
      <td>0.999941</td>
      <td>0.020037</td>
      <td>0.018776</td>
      <td>-0.013067</td>
      <td>NaN</td>
      <td>0.001690</td>
      <td>0.600331</td>
      <td>0.565362</td>
      <td>0.000473</td>
    </tr>
    <tr>
      <th>close</th>
      <td>0.999989</td>
      <td>0.999993</td>
      <td>0.999993</td>
      <td>1.000000</td>
      <td>-0.318133</td>
      <td>-0.488395</td>
      <td>0.003513</td>
      <td>0.999967</td>
      <td>0.999866</td>
      <td>0.999713</td>
      <td>...</td>
      <td>0.999963</td>
      <td>0.999949</td>
      <td>0.020057</td>
      <td>0.018782</td>
      <td>-0.012390</td>
      <td>NaN</td>
      <td>0.001354</td>
      <td>0.599836</td>
      <td>0.564867</td>
      <td>0.000469</td>
    </tr>
    <tr>
      <th>Volume USDT</th>
      <td>-0.318120</td>
      <td>-0.316097</td>
      <td>-0.320212</td>
      <td>-0.318133</td>
      <td>1.000000</td>
      <td>0.939897</td>
      <td>0.057940</td>
      <td>-0.318087</td>
      <td>-0.317966</td>
      <td>-0.317830</td>
      <td>...</td>
      <td>-0.318115</td>
      <td>-0.318069</td>
      <td>-0.010835</td>
      <td>-0.013155</td>
      <td>0.014247</td>
      <td>NaN</td>
      <td>-0.076635</td>
      <td>-0.408955</td>
      <td>-0.406469</td>
      <td>-0.000837</td>
    </tr>
    <tr>
      <th>tradecount</th>
      <td>-0.488418</td>
      <td>-0.486667</td>
      <td>-0.490183</td>
      <td>-0.488395</td>
      <td>0.939897</td>
      <td>1.000000</td>
      <td>0.052775</td>
      <td>-0.488350</td>
      <td>-0.488232</td>
      <td>-0.488085</td>
      <td>...</td>
      <td>-0.488369</td>
      <td>-0.488353</td>
      <td>-0.010553</td>
      <td>-0.012977</td>
      <td>0.015416</td>
      <td>NaN</td>
      <td>-0.072557</td>
      <td>-0.534205</td>
      <td>-0.527959</td>
      <td>-0.001044</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.003511</td>
      <td>0.003822</td>
      <td>0.003199</td>
      <td>0.003513</td>
      <td>0.057940</td>
      <td>0.052775</td>
      <td>1.000000</td>
      <td>0.003555</td>
      <td>0.003669</td>
      <td>0.003853</td>
      <td>...</td>
      <td>0.003519</td>
      <td>0.003554</td>
      <td>-0.014987</td>
      <td>-0.018059</td>
      <td>0.001832</td>
      <td>NaN</td>
      <td>-0.000006</td>
      <td>0.000445</td>
      <td>0.000437</td>
      <td>-0.003491</td>
    </tr>
    <tr>
      <th>ema_5</th>
      <td>0.999956</td>
      <td>0.999960</td>
      <td>0.999960</td>
      <td>0.999967</td>
      <td>-0.318087</td>
      <td>-0.488350</td>
      <td>0.003555</td>
      <td>1.000000</td>
      <td>0.999948</td>
      <td>0.999816</td>
      <td>...</td>
      <td>0.999981</td>
      <td>0.999984</td>
      <td>0.017836</td>
      <td>0.016353</td>
      <td>-0.009404</td>
      <td>NaN</td>
      <td>0.001395</td>
      <td>0.599848</td>
      <td>0.564882</td>
      <td>0.000482</td>
    </tr>
    <tr>
      <th>ema_15</th>
      <td>0.999855</td>
      <td>0.999859</td>
      <td>0.999859</td>
      <td>0.999866</td>
      <td>-0.317966</td>
      <td>-0.488232</td>
      <td>0.003669</td>
      <td>0.999948</td>
      <td>1.000000</td>
      <td>0.999949</td>
      <td>...</td>
      <td>0.999879</td>
      <td>0.999942</td>
      <td>0.014083</td>
      <td>0.012796</td>
      <td>-0.004935</td>
      <td>NaN</td>
      <td>0.001497</td>
      <td>0.599866</td>
      <td>0.564907</td>
      <td>0.000494</td>
    </tr>
    <tr>
      <th>ema_30</th>
      <td>0.999702</td>
      <td>0.999706</td>
      <td>0.999705</td>
      <td>0.999713</td>
      <td>-0.317830</td>
      <td>-0.488085</td>
      <td>0.003853</td>
      <td>0.999816</td>
      <td>0.999949</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.999718</td>
      <td>0.999820</td>
      <td>0.010695</td>
      <td>0.010178</td>
      <td>-0.002462</td>
      <td>NaN</td>
      <td>0.001645</td>
      <td>0.599891</td>
      <td>0.564943</td>
      <td>0.000498</td>
    </tr>
    <tr>
      <th>ema_60</th>
      <td>0.999392</td>
      <td>0.999397</td>
      <td>0.999395</td>
      <td>0.999403</td>
      <td>-0.317607</td>
      <td>-0.487819</td>
      <td>0.004243</td>
      <td>0.999519</td>
      <td>0.999721</td>
      <td>0.999897</td>
      <td>...</td>
      <td>0.999402</td>
      <td>0.999534</td>
      <td>0.007069</td>
      <td>0.007977</td>
      <td>-0.000834</td>
      <td>NaN</td>
      <td>0.001927</td>
      <td>0.599940</td>
      <td>0.565014</td>
      <td>0.000499</td>
    </tr>
    <tr>
      <th>ema_100</th>
      <td>0.998974</td>
      <td>0.998979</td>
      <td>0.998977</td>
      <td>0.998985</td>
      <td>-0.317322</td>
      <td>-0.487476</td>
      <td>0.004750</td>
      <td>0.999107</td>
      <td>0.999347</td>
      <td>0.999608</td>
      <td>...</td>
      <td>0.998980</td>
      <td>0.999128</td>
      <td>0.004631</td>
      <td>0.006751</td>
      <td>-0.000069</td>
      <td>NaN</td>
      <td>0.002280</td>
      <td>0.600007</td>
      <td>0.565107</td>
      <td>0.000498</td>
    </tr>
    <tr>
      <th>ema_200</th>
      <td>0.997916</td>
      <td>0.997923</td>
      <td>0.997919</td>
      <td>0.997927</td>
      <td>-0.316721</td>
      <td>-0.486728</td>
      <td>0.005844</td>
      <td>0.998055</td>
      <td>0.998330</td>
      <td>0.998679</td>
      <td>...</td>
      <td>0.997920</td>
      <td>0.998082</td>
      <td>0.002054</td>
      <td>0.005597</td>
      <td>0.000589</td>
      <td>NaN</td>
      <td>0.003113</td>
      <td>0.600178</td>
      <td>0.565345</td>
      <td>0.000495</td>
    </tr>
    <tr>
      <th>WMA</th>
      <td>0.999890</td>
      <td>0.999894</td>
      <td>0.999894</td>
      <td>0.999901</td>
      <td>-0.318025</td>
      <td>-0.488285</td>
      <td>0.003606</td>
      <td>0.999976</td>
      <td>0.999988</td>
      <td>0.999890</td>
      <td>...</td>
      <td>0.999920</td>
      <td>0.999965</td>
      <td>0.015595</td>
      <td>0.014052</td>
      <td>-0.006065</td>
      <td>NaN</td>
      <td>0.001443</td>
      <td>0.599845</td>
      <td>0.564883</td>
      <td>0.000493</td>
    </tr>
    <tr>
      <th>MACD</th>
      <td>0.005016</td>
      <td>0.005000</td>
      <td>0.005035</td>
      <td>0.005023</td>
      <td>-0.010567</td>
      <td>-0.009643</td>
      <td>-0.016442</td>
      <td>0.002486</td>
      <td>-0.006248</td>
      <td>-0.016272</td>
      <td>...</td>
      <td>0.005835</td>
      <td>0.001459</td>
      <td>0.348782</td>
      <td>0.284268</td>
      <td>-0.287915</td>
      <td>NaN</td>
      <td>-0.013572</td>
      <td>-0.007621</td>
      <td>-0.008290</td>
      <td>-0.000492</td>
    </tr>
    <tr>
      <th>MACD_Signal</th>
      <td>0.000076</td>
      <td>0.000066</td>
      <td>0.000089</td>
      <td>0.000084</td>
      <td>-0.005386</td>
      <td>-0.003481</td>
      <td>0.001467</td>
      <td>-0.004792</td>
      <td>-0.012556</td>
      <td>-0.014610</td>
      <td>...</td>
      <td>-0.000038</td>
      <td>-0.005331</td>
      <td>0.224035</td>
      <td>0.272594</td>
      <td>-0.429228</td>
      <td>NaN</td>
      <td>-0.000521</td>
      <td>0.000023</td>
      <td>0.000014</td>
      <td>-0.001482</td>
    </tr>
    <tr>
      <th>MACD_Hist</th>
      <td>0.005293</td>
      <td>0.005279</td>
      <td>0.005309</td>
      <td>0.005299</td>
      <td>-0.009507</td>
      <td>-0.009127</td>
      <td>-0.017892</td>
      <td>0.004144</td>
      <td>-0.002672</td>
      <td>-0.012652</td>
      <td>...</td>
      <td>0.006197</td>
      <td>0.003225</td>
      <td>0.299239</td>
      <td>0.215568</td>
      <td>-0.170144</td>
      <td>NaN</td>
      <td>-0.014223</td>
      <td>-0.008086</td>
      <td>-0.008793</td>
      <td>-0.000055</td>
    </tr>
    <tr>
      <th>ATR</th>
      <td>0.001642</td>
      <td>0.003534</td>
      <td>-0.000309</td>
      <td>0.001598</td>
      <td>0.551255</td>
      <td>0.493743</td>
      <td>0.152359</td>
      <td>0.003003</td>
      <td>0.005381</td>
      <td>0.007427</td>
      <td>...</td>
      <td>0.002047</td>
      <td>0.002380</td>
      <td>-0.006086</td>
      <td>0.000102</td>
      <td>-0.018075</td>
      <td>NaN</td>
      <td>-0.156913</td>
      <td>-0.204647</td>
      <td>-0.203701</td>
      <td>0.000729</td>
    </tr>
    <tr>
      <th>HMA</th>
      <td>0.999951</td>
      <td>0.999955</td>
      <td>0.999955</td>
      <td>0.999963</td>
      <td>-0.318115</td>
      <td>-0.488369</td>
      <td>0.003519</td>
      <td>0.999981</td>
      <td>0.999879</td>
      <td>0.999718</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.999959</td>
      <td>0.019292</td>
      <td>0.017775</td>
      <td>-0.011819</td>
      <td>NaN</td>
      <td>0.001360</td>
      <td>0.599819</td>
      <td>0.564852</td>
      <td>0.000478</td>
    </tr>
    <tr>
      <th>KAMA</th>
      <td>0.999937</td>
      <td>0.999942</td>
      <td>0.999941</td>
      <td>0.999949</td>
      <td>-0.318069</td>
      <td>-0.488353</td>
      <td>0.003554</td>
      <td>0.999984</td>
      <td>0.999942</td>
      <td>0.999820</td>
      <td>...</td>
      <td>0.999959</td>
      <td>1.000000</td>
      <td>0.014923</td>
      <td>0.013060</td>
      <td>-0.005497</td>
      <td>NaN</td>
      <td>0.001391</td>
      <td>0.599916</td>
      <td>0.564950</td>
      <td>0.000494</td>
    </tr>
    <tr>
      <th>CMO</th>
      <td>0.019921</td>
      <td>0.019978</td>
      <td>0.020037</td>
      <td>0.020057</td>
      <td>-0.010835</td>
      <td>-0.010553</td>
      <td>-0.014987</td>
      <td>0.017836</td>
      <td>0.014083</td>
      <td>0.010695</td>
      <td>...</td>
      <td>0.019292</td>
      <td>0.014923</td>
      <td>1.000000</td>
      <td>0.914051</td>
      <td>-0.538627</td>
      <td>NaN</td>
      <td>-0.007100</td>
      <td>0.006791</td>
      <td>0.004212</td>
      <td>-0.002108</td>
    </tr>
    <tr>
      <th>Z-Score</th>
      <td>0.018638</td>
      <td>0.018690</td>
      <td>0.018776</td>
      <td>0.018782</td>
      <td>-0.013155</td>
      <td>-0.012977</td>
      <td>-0.018059</td>
      <td>0.016353</td>
      <td>0.012796</td>
      <td>0.010178</td>
      <td>...</td>
      <td>0.017775</td>
      <td>0.013060</td>
      <td>0.914051</td>
      <td>1.000000</td>
      <td>-0.566454</td>
      <td>NaN</td>
      <td>-0.002726</td>
      <td>0.008593</td>
      <td>0.006116</td>
      <td>-0.002321</td>
    </tr>
    <tr>
      <th>QStick</th>
      <td>-0.013873</td>
      <td>-0.013057</td>
      <td>-0.013067</td>
      <td>-0.012390</td>
      <td>0.014247</td>
      <td>0.015416</td>
      <td>0.001832</td>
      <td>-0.009404</td>
      <td>-0.004935</td>
      <td>-0.002462</td>
      <td>...</td>
      <td>-0.011819</td>
      <td>-0.005497</td>
      <td>-0.538627</td>
      <td>-0.566454</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.002085</td>
      <td>-0.005607</td>
      <td>-0.005089</td>
      <td>0.001689</td>
    </tr>
    <tr>
      <th>year</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>0.001351</td>
      <td>0.001021</td>
      <td>0.001690</td>
      <td>0.001354</td>
      <td>-0.076635</td>
      <td>-0.072557</td>
      <td>-0.000006</td>
      <td>0.001395</td>
      <td>0.001497</td>
      <td>0.001645</td>
      <td>...</td>
      <td>0.001360</td>
      <td>0.001391</td>
      <td>-0.007100</td>
      <td>-0.002726</td>
      <td>0.002085</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.020340</td>
      <td>0.018098</td>
      <td>-0.001724</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.599839</td>
      <td>0.599359</td>
      <td>0.600331</td>
      <td>0.599836</td>
      <td>-0.408955</td>
      <td>-0.534205</td>
      <td>0.000445</td>
      <td>0.599848</td>
      <td>0.599866</td>
      <td>0.599891</td>
      <td>...</td>
      <td>0.599819</td>
      <td>0.599916</td>
      <td>0.006791</td>
      <td>0.008593</td>
      <td>-0.005607</td>
      <td>NaN</td>
      <td>0.020340</td>
      <td>1.000000</td>
      <td>0.966455</td>
      <td>0.001712</td>
    </tr>
    <tr>
      <th>week_of_year</th>
      <td>0.564870</td>
      <td>0.564390</td>
      <td>0.565362</td>
      <td>0.564867</td>
      <td>-0.406469</td>
      <td>-0.527959</td>
      <td>0.000437</td>
      <td>0.564882</td>
      <td>0.564907</td>
      <td>0.564943</td>
      <td>...</td>
      <td>0.564852</td>
      <td>0.564950</td>
      <td>0.004212</td>
      <td>0.006116</td>
      <td>-0.005089</td>
      <td>NaN</td>
      <td>0.018098</td>
      <td>0.966455</td>
      <td>1.000000</td>
      <td>0.001965</td>
    </tr>
    <tr>
      <th>time_delta</th>
      <td>0.000475</td>
      <td>0.000473</td>
      <td>0.000473</td>
      <td>0.000469</td>
      <td>-0.000837</td>
      <td>-0.001044</td>
      <td>-0.003491</td>
      <td>0.000482</td>
      <td>0.000494</td>
      <td>0.000498</td>
      <td>...</td>
      <td>0.000478</td>
      <td>0.000494</td>
      <td>-0.002108</td>
      <td>-0.002321</td>
      <td>0.001689</td>
      <td>NaN</td>
      <td>-0.001724</td>
      <td>0.001712</td>
      <td>0.001965</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>28 rows × 28 columns</p>
</div>



    
![png](data_analysis_files/data_analysis_23_14.png)
    



```python
# Distributions and Outliers
print("\n" + "="*50)
print("DISTRIBUTIONS AND OUTLIERS")
print("="*50)

# Entire dataset distributions
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    print(f"\nDistribution of {col} (Entire Dataset):")
    plt.figure(figsize=(20, 10))
    sns.histplot(df[col], kde=True, bins=50, color='blue')
    plt.title(f"Distribution of {col} (Entire Dataset)")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Clear the plot after display
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure completely to free memory

# Per year distributions
for year, yearly_df in yearly_data.items():
    for col in numeric_columns:
        print(f"\nDistribution of {col} (Year {year}):")
        plt.figure(figsize=(20, 10))
        sns.histplot(yearly_df[col], kde=True, bins=50, color='green')
        plt.title(f"Distribution of {col} (Year {year})")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

        # Clear the plot after display
        plt.clf()  # Clear the current figure
        plt.close()  # Close the figure completely to free memory

        # Boxplot for outlier detection
        print(f"\nBoxplot of {col} (Year {year}):")
        plt.figure(figsize=(20, 10))
        sns.boxplot(x=yearly_df[col], color='orange')
        plt.title(f"Boxplot of {col} (Year {year})")
        plt.xlabel(col)
        plt.grid(True)
        plt.show()

        # Clear the plot after display
        plt.clf()  # Clear the current figure
        plt.close()  # Close the figure completely to free memory
```

    
    ==================================================
    DISTRIBUTIONS AND OUTLIERS
    ==================================================
    
    Distribution of open (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_1.png)
    


    
    Distribution of high (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_3.png)
    


    
    Distribution of low (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_5.png)
    


    
    Distribution of close (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_7.png)
    


    
    Distribution of Volume USDT (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_9.png)
    


    
    Distribution of tradecount (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_11.png)
    


    
    Distribution of hour (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_13.png)
    


    
    Distribution of ema_5 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_15.png)
    


    
    Distribution of ema_15 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_17.png)
    


    
    Distribution of ema_30 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_19.png)
    


    
    Distribution of ema_60 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_21.png)
    


    
    Distribution of ema_100 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_23.png)
    


    
    Distribution of ema_200 (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_25.png)
    


    
    Distribution of WMA (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_27.png)
    


    
    Distribution of MACD (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_29.png)
    


    
    Distribution of MACD_Signal (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_31.png)
    


    
    Distribution of MACD_Hist (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_33.png)
    


    
    Distribution of ATR (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_35.png)
    


    
    Distribution of HMA (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_37.png)
    


    
    Distribution of KAMA (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_39.png)
    


    
    Distribution of CMO (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_41.png)
    


    
    Distribution of Z-Score (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_43.png)
    


    
    Distribution of QStick (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_45.png)
    


    
    Distribution of year (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_47.png)
    


    
    Distribution of day_of_week (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_49.png)
    


    
    Distribution of month (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_51.png)
    


    
    Distribution of week_of_year (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_53.png)
    


    
    Distribution of time_delta (Entire Dataset):



    
![png](data_analysis_files/data_analysis_24_55.png)
    


    
    Distribution of open (Year 2020):



    
![png](data_analysis_files/data_analysis_24_57.png)
    


    
    Boxplot of open (Year 2020):



    
![png](data_analysis_files/data_analysis_24_59.png)
    


    
    Distribution of high (Year 2020):



    
![png](data_analysis_files/data_analysis_24_61.png)
    


    
    Boxplot of high (Year 2020):



    
![png](data_analysis_files/data_analysis_24_63.png)
    


    
    Distribution of low (Year 2020):



    
![png](data_analysis_files/data_analysis_24_65.png)
    


    
    Boxplot of low (Year 2020):



    
![png](data_analysis_files/data_analysis_24_67.png)
    


    
    Distribution of close (Year 2020):



    
![png](data_analysis_files/data_analysis_24_69.png)
    


    
    Boxplot of close (Year 2020):



    
![png](data_analysis_files/data_analysis_24_71.png)
    


    
    Distribution of Volume USDT (Year 2020):



    
![png](data_analysis_files/data_analysis_24_73.png)
    


    
    Boxplot of Volume USDT (Year 2020):



    
![png](data_analysis_files/data_analysis_24_75.png)
    


    
    Distribution of tradecount (Year 2020):



    
![png](data_analysis_files/data_analysis_24_77.png)
    


    
    Boxplot of tradecount (Year 2020):



    
![png](data_analysis_files/data_analysis_24_79.png)
    


    
    Distribution of hour (Year 2020):



    
![png](data_analysis_files/data_analysis_24_81.png)
    


    
    Boxplot of hour (Year 2020):



    
![png](data_analysis_files/data_analysis_24_83.png)
    


    
    Distribution of ema_5 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_85.png)
    


    
    Boxplot of ema_5 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_87.png)
    


    
    Distribution of ema_15 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_89.png)
    


    
    Boxplot of ema_15 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_91.png)
    


    
    Distribution of ema_30 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_93.png)
    


    
    Boxplot of ema_30 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_95.png)
    


    
    Distribution of ema_60 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_97.png)
    


    
    Boxplot of ema_60 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_99.png)
    


    
    Distribution of ema_100 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_101.png)
    


    
    Boxplot of ema_100 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_103.png)
    


    
    Distribution of ema_200 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_105.png)
    


    
    Boxplot of ema_200 (Year 2020):



    
![png](data_analysis_files/data_analysis_24_107.png)
    


    
    Distribution of WMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_109.png)
    


    
    Boxplot of WMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_111.png)
    


    
    Distribution of MACD (Year 2020):



    
![png](data_analysis_files/data_analysis_24_113.png)
    


    
    Boxplot of MACD (Year 2020):



    
![png](data_analysis_files/data_analysis_24_115.png)
    


    
    Distribution of MACD_Signal (Year 2020):



    
![png](data_analysis_files/data_analysis_24_117.png)
    


    
    Boxplot of MACD_Signal (Year 2020):



    
![png](data_analysis_files/data_analysis_24_119.png)
    


    
    Distribution of MACD_Hist (Year 2020):



    
![png](data_analysis_files/data_analysis_24_121.png)
    


    
    Boxplot of MACD_Hist (Year 2020):



    
![png](data_analysis_files/data_analysis_24_123.png)
    


    
    Distribution of ATR (Year 2020):



    
![png](data_analysis_files/data_analysis_24_125.png)
    


    
    Boxplot of ATR (Year 2020):



    
![png](data_analysis_files/data_analysis_24_127.png)
    


    
    Distribution of HMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_129.png)
    


    
    Boxplot of HMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_131.png)
    


    
    Distribution of KAMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_133.png)
    


    
    Boxplot of KAMA (Year 2020):



    
![png](data_analysis_files/data_analysis_24_135.png)
    


    
    Distribution of CMO (Year 2020):



    
![png](data_analysis_files/data_analysis_24_137.png)
    


    
    Boxplot of CMO (Year 2020):



    
![png](data_analysis_files/data_analysis_24_139.png)
    


    
    Distribution of Z-Score (Year 2020):



    
![png](data_analysis_files/data_analysis_24_141.png)
    


    
    Boxplot of Z-Score (Year 2020):



    
![png](data_analysis_files/data_analysis_24_143.png)
    


    
    Distribution of QStick (Year 2020):



    
![png](data_analysis_files/data_analysis_24_145.png)
    


    
    Boxplot of QStick (Year 2020):



    
![png](data_analysis_files/data_analysis_24_147.png)
    


    
    Distribution of year (Year 2020):



    
![png](data_analysis_files/data_analysis_24_149.png)
    


    
    Boxplot of year (Year 2020):



    
![png](data_analysis_files/data_analysis_24_151.png)
    


    
    Distribution of day_of_week (Year 2020):



    
![png](data_analysis_files/data_analysis_24_153.png)
    


    
    Boxplot of day_of_week (Year 2020):



    
![png](data_analysis_files/data_analysis_24_155.png)
    


    
    Distribution of month (Year 2020):



    
![png](data_analysis_files/data_analysis_24_157.png)
    


    
    Boxplot of month (Year 2020):



    
![png](data_analysis_files/data_analysis_24_159.png)
    


    
    Distribution of week_of_year (Year 2020):



    
![png](data_analysis_files/data_analysis_24_161.png)
    


    
    Boxplot of week_of_year (Year 2020):



    
![png](data_analysis_files/data_analysis_24_163.png)
    


    
    Distribution of time_delta (Year 2020):



    
![png](data_analysis_files/data_analysis_24_165.png)
    


    
    Boxplot of time_delta (Year 2020):



    
![png](data_analysis_files/data_analysis_24_167.png)
    


    
    Distribution of open (Year 2021):



    
![png](data_analysis_files/data_analysis_24_169.png)
    


    
    Boxplot of open (Year 2021):



    
![png](data_analysis_files/data_analysis_24_171.png)
    


    
    Distribution of high (Year 2021):



    
![png](data_analysis_files/data_analysis_24_173.png)
    


    
    Boxplot of high (Year 2021):



    
![png](data_analysis_files/data_analysis_24_175.png)
    


    
    Distribution of low (Year 2021):



    
![png](data_analysis_files/data_analysis_24_177.png)
    


    
    Boxplot of low (Year 2021):



    
![png](data_analysis_files/data_analysis_24_179.png)
    


    
    Distribution of close (Year 2021):



    
![png](data_analysis_files/data_analysis_24_181.png)
    


    
    Boxplot of close (Year 2021):



    
![png](data_analysis_files/data_analysis_24_183.png)
    


    
    Distribution of Volume USDT (Year 2021):



    
![png](data_analysis_files/data_analysis_24_185.png)
    


    
    Boxplot of Volume USDT (Year 2021):



    
![png](data_analysis_files/data_analysis_24_187.png)
    


    
    Distribution of tradecount (Year 2021):



    
![png](data_analysis_files/data_analysis_24_189.png)
    


    
    Boxplot of tradecount (Year 2021):



    
![png](data_analysis_files/data_analysis_24_191.png)
    


    
    Distribution of hour (Year 2021):



    
![png](data_analysis_files/data_analysis_24_193.png)
    


    
    Boxplot of hour (Year 2021):



    
![png](data_analysis_files/data_analysis_24_195.png)
    


    
    Distribution of ema_5 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_197.png)
    


    
    Boxplot of ema_5 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_199.png)
    


    
    Distribution of ema_15 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_201.png)
    


    
    Boxplot of ema_15 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_203.png)
    


    
    Distribution of ema_30 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_205.png)
    


    
    Boxplot of ema_30 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_207.png)
    


    
    Distribution of ema_60 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_209.png)
    


    
    Boxplot of ema_60 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_211.png)
    


    
    Distribution of ema_100 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_213.png)
    


    
    Boxplot of ema_100 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_215.png)
    


    
    Distribution of ema_200 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_217.png)
    


    
    Boxplot of ema_200 (Year 2021):



    
![png](data_analysis_files/data_analysis_24_219.png)
    


    
    Distribution of WMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_221.png)
    


    
    Boxplot of WMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_223.png)
    


    
    Distribution of MACD (Year 2021):



    
![png](data_analysis_files/data_analysis_24_225.png)
    


    
    Boxplot of MACD (Year 2021):



    
![png](data_analysis_files/data_analysis_24_227.png)
    


    
    Distribution of MACD_Signal (Year 2021):



    
![png](data_analysis_files/data_analysis_24_229.png)
    


    
    Boxplot of MACD_Signal (Year 2021):



    
![png](data_analysis_files/data_analysis_24_231.png)
    


    
    Distribution of MACD_Hist (Year 2021):



    
![png](data_analysis_files/data_analysis_24_233.png)
    


    
    Boxplot of MACD_Hist (Year 2021):



    
![png](data_analysis_files/data_analysis_24_235.png)
    


    
    Distribution of ATR (Year 2021):



    
![png](data_analysis_files/data_analysis_24_237.png)
    


    
    Boxplot of ATR (Year 2021):



    
![png](data_analysis_files/data_analysis_24_239.png)
    


    
    Distribution of HMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_241.png)
    


    
    Boxplot of HMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_243.png)
    


    
    Distribution of KAMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_245.png)
    


    
    Boxplot of KAMA (Year 2021):



    
![png](data_analysis_files/data_analysis_24_247.png)
    


    
    Distribution of CMO (Year 2021):



    
![png](data_analysis_files/data_analysis_24_249.png)
    


    
    Boxplot of CMO (Year 2021):



    
![png](data_analysis_files/data_analysis_24_251.png)
    


    
    Distribution of Z-Score (Year 2021):



    
![png](data_analysis_files/data_analysis_24_253.png)
    


    
    Boxplot of Z-Score (Year 2021):



    
![png](data_analysis_files/data_analysis_24_255.png)
    


    
    Distribution of QStick (Year 2021):



    
![png](data_analysis_files/data_analysis_24_257.png)
    


    
    Boxplot of QStick (Year 2021):



    
![png](data_analysis_files/data_analysis_24_259.png)
    


    
    Distribution of year (Year 2021):



    
![png](data_analysis_files/data_analysis_24_261.png)
    


    
    Boxplot of year (Year 2021):



    
![png](data_analysis_files/data_analysis_24_263.png)
    


    
    Distribution of day_of_week (Year 2021):



    
![png](data_analysis_files/data_analysis_24_265.png)
    


    
    Boxplot of day_of_week (Year 2021):



    
![png](data_analysis_files/data_analysis_24_267.png)
    


    
    Distribution of month (Year 2021):



    
![png](data_analysis_files/data_analysis_24_269.png)
    


    
    Boxplot of month (Year 2021):



    
![png](data_analysis_files/data_analysis_24_271.png)
    


    
    Distribution of week_of_year (Year 2021):



    
![png](data_analysis_files/data_analysis_24_273.png)
    


    
    Boxplot of week_of_year (Year 2021):



    
![png](data_analysis_files/data_analysis_24_275.png)
    


    
    Distribution of time_delta (Year 2021):



    
![png](data_analysis_files/data_analysis_24_277.png)
    


    
    Boxplot of time_delta (Year 2021):



    
![png](data_analysis_files/data_analysis_24_279.png)
    


    
    Distribution of open (Year 2022):



    
![png](data_analysis_files/data_analysis_24_281.png)
    


    
    Boxplot of open (Year 2022):



    
![png](data_analysis_files/data_analysis_24_283.png)
    


    
    Distribution of high (Year 2022):



    
![png](data_analysis_files/data_analysis_24_285.png)
    


    
    Boxplot of high (Year 2022):



    
![png](data_analysis_files/data_analysis_24_287.png)
    


    
    Distribution of low (Year 2022):



    
![png](data_analysis_files/data_analysis_24_289.png)
    


    
    Boxplot of low (Year 2022):



    
![png](data_analysis_files/data_analysis_24_291.png)
    


    
    Distribution of close (Year 2022):



    
![png](data_analysis_files/data_analysis_24_293.png)
    


    
    Boxplot of close (Year 2022):



    
![png](data_analysis_files/data_analysis_24_295.png)
    


    
    Distribution of Volume USDT (Year 2022):



    
![png](data_analysis_files/data_analysis_24_297.png)
    


    
    Boxplot of Volume USDT (Year 2022):



    
![png](data_analysis_files/data_analysis_24_299.png)
    


    
    Distribution of tradecount (Year 2022):



    
![png](data_analysis_files/data_analysis_24_301.png)
    


    
    Boxplot of tradecount (Year 2022):



    
![png](data_analysis_files/data_analysis_24_303.png)
    


    
    Distribution of hour (Year 2022):



    
![png](data_analysis_files/data_analysis_24_305.png)
    


    
    Boxplot of hour (Year 2022):



    
![png](data_analysis_files/data_analysis_24_307.png)
    


    
    Distribution of ema_5 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_309.png)
    


    
    Boxplot of ema_5 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_311.png)
    


    
    Distribution of ema_15 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_313.png)
    


    
    Boxplot of ema_15 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_315.png)
    


    
    Distribution of ema_30 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_317.png)
    


    
    Boxplot of ema_30 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_319.png)
    


    
    Distribution of ema_60 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_321.png)
    


    
    Boxplot of ema_60 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_323.png)
    


    
    Distribution of ema_100 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_325.png)
    


    
    Boxplot of ema_100 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_327.png)
    


    
    Distribution of ema_200 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_329.png)
    


    
    Boxplot of ema_200 (Year 2022):



    
![png](data_analysis_files/data_analysis_24_331.png)
    


    
    Distribution of WMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_333.png)
    


    
    Boxplot of WMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_335.png)
    


    
    Distribution of MACD (Year 2022):



    
![png](data_analysis_files/data_analysis_24_337.png)
    


    
    Boxplot of MACD (Year 2022):



    
![png](data_analysis_files/data_analysis_24_339.png)
    


    
    Distribution of MACD_Signal (Year 2022):



    
![png](data_analysis_files/data_analysis_24_341.png)
    


    
    Boxplot of MACD_Signal (Year 2022):



    
![png](data_analysis_files/data_analysis_24_343.png)
    


    
    Distribution of MACD_Hist (Year 2022):



    
![png](data_analysis_files/data_analysis_24_345.png)
    


    
    Boxplot of MACD_Hist (Year 2022):



    
![png](data_analysis_files/data_analysis_24_347.png)
    


    
    Distribution of ATR (Year 2022):



    
![png](data_analysis_files/data_analysis_24_349.png)
    


    
    Boxplot of ATR (Year 2022):



    
![png](data_analysis_files/data_analysis_24_351.png)
    


    
    Distribution of HMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_353.png)
    


    
    Boxplot of HMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_355.png)
    


    
    Distribution of KAMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_357.png)
    


    
    Boxplot of KAMA (Year 2022):



    
![png](data_analysis_files/data_analysis_24_359.png)
    


    
    Distribution of CMO (Year 2022):



    
![png](data_analysis_files/data_analysis_24_361.png)
    


    
    Boxplot of CMO (Year 2022):



    
![png](data_analysis_files/data_analysis_24_363.png)
    


    
    Distribution of Z-Score (Year 2022):



    
![png](data_analysis_files/data_analysis_24_365.png)
    


    
    Boxplot of Z-Score (Year 2022):



    
![png](data_analysis_files/data_analysis_24_367.png)
    


    
    Distribution of QStick (Year 2022):



    
![png](data_analysis_files/data_analysis_24_369.png)
    


    
    Boxplot of QStick (Year 2022):



    
![png](data_analysis_files/data_analysis_24_371.png)
    


    
    Distribution of year (Year 2022):



    
![png](data_analysis_files/data_analysis_24_373.png)
    


    
    Boxplot of year (Year 2022):



    
![png](data_analysis_files/data_analysis_24_375.png)
    


    
    Distribution of day_of_week (Year 2022):



    
![png](data_analysis_files/data_analysis_24_377.png)
    


    
    Boxplot of day_of_week (Year 2022):



    
![png](data_analysis_files/data_analysis_24_379.png)
    


    
    Distribution of month (Year 2022):



    
![png](data_analysis_files/data_analysis_24_381.png)
    


    
    Boxplot of month (Year 2022):



    
![png](data_analysis_files/data_analysis_24_383.png)
    


    
    Distribution of week_of_year (Year 2022):



    
![png](data_analysis_files/data_analysis_24_385.png)
    


    
    Boxplot of week_of_year (Year 2022):



    
![png](data_analysis_files/data_analysis_24_387.png)
    


    
    Distribution of time_delta (Year 2022):



    
![png](data_analysis_files/data_analysis_24_389.png)
    


    
    Boxplot of time_delta (Year 2022):



    
![png](data_analysis_files/data_analysis_24_391.png)
    


    
    Distribution of open (Year 2023):



    
![png](data_analysis_files/data_analysis_24_393.png)
    


    
    Boxplot of open (Year 2023):



    
![png](data_analysis_files/data_analysis_24_395.png)
    


    
    Distribution of high (Year 2023):



    
![png](data_analysis_files/data_analysis_24_397.png)
    


    
    Boxplot of high (Year 2023):



    
![png](data_analysis_files/data_analysis_24_399.png)
    


    
    Distribution of low (Year 2023):



    
![png](data_analysis_files/data_analysis_24_401.png)
    


    
    Boxplot of low (Year 2023):



    
![png](data_analysis_files/data_analysis_24_403.png)
    


    
    Distribution of close (Year 2023):



    
![png](data_analysis_files/data_analysis_24_405.png)
    


    
    Boxplot of close (Year 2023):



    
![png](data_analysis_files/data_analysis_24_407.png)
    


    
    Distribution of Volume USDT (Year 2023):



    
![png](data_analysis_files/data_analysis_24_409.png)
    


    
    Boxplot of Volume USDT (Year 2023):



    
![png](data_analysis_files/data_analysis_24_411.png)
    


    
    Distribution of tradecount (Year 2023):



    
![png](data_analysis_files/data_analysis_24_413.png)
    


    
    Boxplot of tradecount (Year 2023):



    
![png](data_analysis_files/data_analysis_24_415.png)
    


    
    Distribution of hour (Year 2023):



    
![png](data_analysis_files/data_analysis_24_417.png)
    


    
    Boxplot of hour (Year 2023):



    
![png](data_analysis_files/data_analysis_24_419.png)
    


    
    Distribution of ema_5 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_421.png)
    


    
    Boxplot of ema_5 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_423.png)
    


    
    Distribution of ema_15 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_425.png)
    


    
    Boxplot of ema_15 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_427.png)
    


    
    Distribution of ema_30 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_429.png)
    


    
    Boxplot of ema_30 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_431.png)
    


    
    Distribution of ema_60 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_433.png)
    


    
    Boxplot of ema_60 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_435.png)
    


    
    Distribution of ema_100 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_437.png)
    


    
    Boxplot of ema_100 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_439.png)
    


    
    Distribution of ema_200 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_441.png)
    


    
    Boxplot of ema_200 (Year 2023):



    
![png](data_analysis_files/data_analysis_24_443.png)
    


    
    Distribution of WMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_445.png)
    


    
    Boxplot of WMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_447.png)
    


    
    Distribution of MACD (Year 2023):



    
![png](data_analysis_files/data_analysis_24_449.png)
    


    
    Boxplot of MACD (Year 2023):



    
![png](data_analysis_files/data_analysis_24_451.png)
    


    
    Distribution of MACD_Signal (Year 2023):



    
![png](data_analysis_files/data_analysis_24_453.png)
    


    
    Boxplot of MACD_Signal (Year 2023):



    
![png](data_analysis_files/data_analysis_24_455.png)
    


    
    Distribution of MACD_Hist (Year 2023):



    
![png](data_analysis_files/data_analysis_24_457.png)
    


    
    Boxplot of MACD_Hist (Year 2023):



    
![png](data_analysis_files/data_analysis_24_459.png)
    


    
    Distribution of ATR (Year 2023):



    
![png](data_analysis_files/data_analysis_24_461.png)
    


    
    Boxplot of ATR (Year 2023):



    
![png](data_analysis_files/data_analysis_24_463.png)
    


    
    Distribution of HMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_465.png)
    


    
    Boxplot of HMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_467.png)
    


    
    Distribution of KAMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_469.png)
    


    
    Boxplot of KAMA (Year 2023):



    
![png](data_analysis_files/data_analysis_24_471.png)
    


    
    Distribution of CMO (Year 2023):



    
![png](data_analysis_files/data_analysis_24_473.png)
    


    
    Boxplot of CMO (Year 2023):



    
![png](data_analysis_files/data_analysis_24_475.png)
    


    
    Distribution of Z-Score (Year 2023):



    
![png](data_analysis_files/data_analysis_24_477.png)
    


    
    Boxplot of Z-Score (Year 2023):



    
![png](data_analysis_files/data_analysis_24_479.png)
    


    
    Distribution of QStick (Year 2023):



    
![png](data_analysis_files/data_analysis_24_481.png)
    


    
    Boxplot of QStick (Year 2023):



    
![png](data_analysis_files/data_analysis_24_483.png)
    


    
    Distribution of year (Year 2023):



    
![png](data_analysis_files/data_analysis_24_485.png)
    


    
    Boxplot of year (Year 2023):



    
![png](data_analysis_files/data_analysis_24_487.png)
    


    
    Distribution of day_of_week (Year 2023):



    
![png](data_analysis_files/data_analysis_24_489.png)
    


    
    Boxplot of day_of_week (Year 2023):



    
![png](data_analysis_files/data_analysis_24_491.png)
    


    
    Distribution of month (Year 2023):



    
![png](data_analysis_files/data_analysis_24_493.png)
    


    
    Boxplot of month (Year 2023):



    
![png](data_analysis_files/data_analysis_24_495.png)
    


    
    Distribution of week_of_year (Year 2023):



    
![png](data_analysis_files/data_analysis_24_497.png)
    


    
    Boxplot of week_of_year (Year 2023):



    
![png](data_analysis_files/data_analysis_24_499.png)
    


    
    Distribution of time_delta (Year 2023):



    
![png](data_analysis_files/data_analysis_24_501.png)
    


    
    Boxplot of time_delta (Year 2023):



    
![png](data_analysis_files/data_analysis_24_503.png)
    



```python
# Apply the tests to the entire dataset
normality_results = perform_normality_tests(df)
```

    Skipping column 'time_delta' (Subsample has zero range).



```python
# Display results
print("\nNormality Test Results (Entire Dataset):")
display(normality_results)
```

    
    Normality Test Results (Entire Dataset):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Shapiro-Wilk</th>
      <th>Shapiro p-value</th>
      <th>Kolmogorov-Smirnov</th>
      <th>KS p-value</th>
      <th>Anderson-Darling</th>
      <th>AD Statistic</th>
      <th>AD Critical Value (5%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>Not Normal</td>
      <td>2.575203e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25845.062912</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>Not Normal</td>
      <td>2.460729e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25880.829540</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Not Normal</td>
      <td>2.696461e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25809.164291</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>Not Normal</td>
      <td>2.569391e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25845.035856</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>Not Normal</td>
      <td>1.466383e-79</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>228224.137004</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>Not Normal</td>
      <td>5.617090e-75</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>219363.555472</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>Not Normal</td>
      <td>1.659614e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>24310.483564</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>Not Normal</td>
      <td>2.568229e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25844.564911</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>Not Normal</td>
      <td>2.569860e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25843.551686</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>Not Normal</td>
      <td>2.562198e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25842.028230</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>Not Normal</td>
      <td>2.520670e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25838.937670</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>Not Normal</td>
      <td>2.461158e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25834.677005</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>Not Normal</td>
      <td>2.337066e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25823.512000</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>Not Normal</td>
      <td>2.567995e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25844.615046</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>Not Normal</td>
      <td>9.361082e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>216012.572954</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>Not Normal</td>
      <td>3.444452e-64</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>206394.860400</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>Not Normal</td>
      <td>2.924564e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>216804.063523</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>Not Normal</td>
      <td>2.206792e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>121742.285829</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>Not Normal</td>
      <td>2.571053e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25845.394148</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>Not Normal</td>
      <td>2.558699e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>25844.151149</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>Not Normal</td>
      <td>4.971199e-05</td>
      <td>Not Normal</td>
      <td>1.034612e-282</td>
      <td>Not Normal</td>
      <td>800.371958</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>Not Normal</td>
      <td>5.200202e-19</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>8705.955604</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>Not Normal</td>
      <td>4.604842e-66</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>135965.618845</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>year</td>
      <td>Not Normal</td>
      <td>1.327414e-54</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>97061.784867</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>Not Normal</td>
      <td>7.127199e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>47029.065150</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>25</th>
      <td>month</td>
      <td>Not Normal</td>
      <td>3.532288e-39</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>26789.596481</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>26</th>
      <td>week_of_year</td>
      <td>Not Normal</td>
      <td>4.102543e-35</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>19025.075483</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Perform the tests per year
for year, yearly_df in yearly_data.items():
    print(f"\nNormality Test Results (Year {year}):")
    yearly_normality_results = perform_normality_tests(yearly_df)
    display(yearly_normality_results)  # Display yearly results
```

    
    Normality Test Results (Year 2020):
    Skipping column 'year' due to zero range (all values are identical).
    Skipping column 'time_delta' (Subsample has zero range).



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Shapiro-Wilk</th>
      <th>Shapiro p-value</th>
      <th>Kolmogorov-Smirnov</th>
      <th>KS p-value</th>
      <th>Anderson-Darling</th>
      <th>AD Statistic</th>
      <th>AD Critical Value (5%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>Not Normal</td>
      <td>6.503383e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34494.569497</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>Not Normal</td>
      <td>6.006234e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34558.028417</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Not Normal</td>
      <td>7.246887e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34430.344665</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>Not Normal</td>
      <td>6.632992e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34495.312135</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>Not Normal</td>
      <td>4.413268e-86</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>83993.861945</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>Not Normal</td>
      <td>8.681307e-79</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>51499.044650</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>Not Normal</td>
      <td>4.678985e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6417.938184</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>Not Normal</td>
      <td>6.458003e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34497.629451</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>Not Normal</td>
      <td>6.406617e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34502.886399</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>Not Normal</td>
      <td>6.362745e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34510.578932</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>Not Normal</td>
      <td>6.227882e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34525.786529</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>Not Normal</td>
      <td>6.038140e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34545.916799</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>Not Normal</td>
      <td>5.557113e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34595.685885</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>Not Normal</td>
      <td>6.674850e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34498.255980</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>Not Normal</td>
      <td>2.560695e-67</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>29018.801758</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>Not Normal</td>
      <td>1.110792e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>26652.968688</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>Not Normal</td>
      <td>7.953148e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>29293.979421</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>Not Normal</td>
      <td>8.960183e-75</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>44361.447603</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>Not Normal</td>
      <td>6.735729e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34493.821384</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>Not Normal</td>
      <td>7.449397e-60</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>34482.106929</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>Not Normal</td>
      <td>4.248576e-10</td>
      <td>Not Normal</td>
      <td>1.583032e-137</td>
      <td>Not Normal</td>
      <td>368.639396</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>Not Normal</td>
      <td>3.157672e-21</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>2200.514305</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>Not Normal</td>
      <td>4.059196e-70</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>29267.999709</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>day_of_week</td>
      <td>Not Normal</td>
      <td>1.419726e-45</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>12312.928961</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>month</td>
      <td>Not Normal</td>
      <td>6.400202e-41</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>7974.417051</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>25</th>
      <td>week_of_year</td>
      <td>Not Normal</td>
      <td>5.035630e-37</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>5920.030179</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>
</div>


    
    Normality Test Results (Year 2021):
    Skipping column 'year' due to zero range (all values are identical).
    Skipping column 'time_delta' (Subsample has zero range).



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Shapiro-Wilk</th>
      <th>Shapiro p-value</th>
      <th>Kolmogorov-Smirnov</th>
      <th>KS p-value</th>
      <th>Anderson-Darling</th>
      <th>AD Statistic</th>
      <th>AD Critical Value (5%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>Not Normal</td>
      <td>8.307602e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6720.307935</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>Not Normal</td>
      <td>7.895858e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6718.799780</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Not Normal</td>
      <td>8.163437e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6721.300707</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>Not Normal</td>
      <td>8.177442e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6720.323549</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>Not Normal</td>
      <td>1.992431e-81</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>55635.985051</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>Not Normal</td>
      <td>1.872780e-78</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>45400.866875</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>Not Normal</td>
      <td>1.776376e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6366.527194</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>Not Normal</td>
      <td>7.825670e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6723.008774</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>Not Normal</td>
      <td>7.302454e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6727.950694</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>Not Normal</td>
      <td>7.003014e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6734.778542</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>Not Normal</td>
      <td>6.711057e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6747.371282</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>Not Normal</td>
      <td>6.455066e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6763.107775</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>Not Normal</td>
      <td>5.686472e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6800.178243</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>Not Normal</td>
      <td>7.446859e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6724.170647</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>Not Normal</td>
      <td>3.258864e-49</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31611.079369</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>Not Normal</td>
      <td>4.716352e-49</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>28749.194156</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>Not Normal</td>
      <td>4.291401e-50</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31871.542263</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>Not Normal</td>
      <td>1.212660e-74</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>27271.715868</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>Not Normal</td>
      <td>8.318309e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6719.057606</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>Not Normal</td>
      <td>7.152036e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6732.885710</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>Not Normal</td>
      <td>1.118497e-09</td>
      <td>Not Normal</td>
      <td>9.792251e-88</td>
      <td>Not Normal</td>
      <td>242.309938</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>Not Normal</td>
      <td>3.132727e-23</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>2435.517890</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>Not Normal</td>
      <td>5.002206e-53</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9985.095997</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>day_of_week</td>
      <td>Not Normal</td>
      <td>2.976386e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>12373.233318</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>month</td>
      <td>Not Normal</td>
      <td>2.073779e-40</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>7891.463810</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>25</th>
      <td>week_of_year</td>
      <td>Not Normal</td>
      <td>1.455181e-36</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>5892.313352</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>
</div>


    
    Normality Test Results (Year 2022):
    Skipping column 'year' due to zero range (all values are identical).
    Skipping column 'time_delta' due to zero range (all values are identical).



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Shapiro-Wilk</th>
      <th>Shapiro p-value</th>
      <th>Kolmogorov-Smirnov</th>
      <th>KS p-value</th>
      <th>Anderson-Darling</th>
      <th>AD Statistic</th>
      <th>AD Critical Value (5%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>Not Normal</td>
      <td>5.251834e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31141.633474</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>Not Normal</td>
      <td>5.292743e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31130.039393</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Not Normal</td>
      <td>5.266839e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31153.159205</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>Not Normal</td>
      <td>5.317889e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31141.769412</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>Not Normal</td>
      <td>2.866764e-76</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>46975.011897</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>Not Normal</td>
      <td>3.158713e-68</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>30469.433593</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>Not Normal</td>
      <td>1.575762e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6402.560859</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>Not Normal</td>
      <td>5.286027e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31143.927998</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>Not Normal</td>
      <td>5.218903e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31148.232990</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>Not Normal</td>
      <td>5.143830e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31154.573067</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>Not Normal</td>
      <td>5.019799e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31167.163977</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>Not Normal</td>
      <td>4.875729e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31183.758946</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>Not Normal</td>
      <td>4.568495e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31224.440362</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>Not Normal</td>
      <td>5.255654e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31144.817360</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>Not Normal</td>
      <td>1.807249e-52</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31081.978148</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>Not Normal</td>
      <td>1.782062e-53</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>28791.400123</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>Not Normal</td>
      <td>1.290922e-54</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31209.919333</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>Not Normal</td>
      <td>4.000186e-63</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>21336.133929</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>Not Normal</td>
      <td>5.325232e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31140.128322</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>Not Normal</td>
      <td>5.228559e-55</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>31147.437624</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>Not Normal</td>
      <td>1.012032e-09</td>
      <td>Not Normal</td>
      <td>9.311448e-79</td>
      <td>Not Normal</td>
      <td>260.058105</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>Not Normal</td>
      <td>7.551051e-22</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>2174.199879</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>Not Normal</td>
      <td>1.732052e-57</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>21018.175567</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>day_of_week</td>
      <td>Not Normal</td>
      <td>2.786744e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>12491.714232</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>month</td>
      <td>Not Normal</td>
      <td>2.560439e-41</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>7897.674491</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>25</th>
      <td>week_of_year</td>
      <td>Not Normal</td>
      <td>8.128024e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>5978.782966</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>
</div>


    
    Normality Test Results (Year 2023):
    Skipping column 'year' due to zero range (all values are identical).
    Skipping column 'time_delta' (Subsample has zero range).



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Shapiro-Wilk</th>
      <th>Shapiro p-value</th>
      <th>Kolmogorov-Smirnov</th>
      <th>KS p-value</th>
      <th>Anderson-Darling</th>
      <th>AD Statistic</th>
      <th>AD Critical Value (5%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>open</td>
      <td>Not Normal</td>
      <td>9.719478e-47</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9772.514741</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>high</td>
      <td>Not Normal</td>
      <td>9.829771e-47</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9759.706847</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>2</th>
      <td>low</td>
      <td>Not Normal</td>
      <td>9.726019e-47</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9786.102076</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>3</th>
      <td>close</td>
      <td>Not Normal</td>
      <td>9.714316e-47</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9772.276757</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Volume USDT</td>
      <td>Not Normal</td>
      <td>1.203408e-79</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>56209.211019</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tradecount</td>
      <td>Not Normal</td>
      <td>9.761521e-75</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>50368.574385</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hour</td>
      <td>Not Normal</td>
      <td>8.952761e-38</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>5123.535336</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>Not Normal</td>
      <td>1.426915e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9771.168650</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>Not Normal</td>
      <td>1.670649e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9767.962540</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>Not Normal</td>
      <td>1.488314e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9762.931719</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>Not Normal</td>
      <td>1.480350e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9752.574652</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>Not Normal</td>
      <td>1.655552e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9738.559052</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>Not Normal</td>
      <td>2.305356e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9701.689693</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>Not Normal</td>
      <td>1.714020e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9769.592196</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>Not Normal</td>
      <td>3.760909e-92</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>59302.380635</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>Not Normal</td>
      <td>1.224228e-94</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>55852.837957</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>Not Normal</td>
      <td>6.947743e-83</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>59741.736433</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>Not Normal</td>
      <td>1.061258e-81</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>21871.697745</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>Not Normal</td>
      <td>1.006520e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9770.402837</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>Not Normal</td>
      <td>1.194080e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9779.585249</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>Not Normal</td>
      <td>2.383260e-04</td>
      <td>Not Normal</td>
      <td>1.070476e-45</td>
      <td>Not Normal</td>
      <td>116.452610</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>Not Normal</td>
      <td>4.974417e-20</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>1923.163190</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>Not Normal</td>
      <td>9.983160e-62</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>17866.784388</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>23</th>
      <td>day_of_week</td>
      <td>Not Normal</td>
      <td>8.151252e-46</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>9856.451421</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>24</th>
      <td>month</td>
      <td>Not Normal</td>
      <td>9.477601e-41</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>6761.736973</td>
      <td>0.787</td>
    </tr>
    <tr>
      <th>25</th>
      <td>week_of_year</td>
      <td>Not Normal</td>
      <td>2.778185e-35</td>
      <td>Not Normal</td>
      <td>0.000000e+00</td>
      <td>Not Normal</td>
      <td>4489.834044</td>
      <td>0.787</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Time Series and Seasonality
print("\n" + "="*50)
print("TIME SERIES ANALYSIS AND SEASONALITY")
print("="*50)

# Monthly averages for the entire dataset
print("Monthly Average Close Prices (Entire Dataset):")
monthly_avg = df.groupby(df['date'].dt.to_period("M"))['close'].mean()
monthly_avg.plot(figsize=(20, 10), title="Monthly Average Close Prices (Entire Dataset)", ylabel="Average Close Price")
plt.show()

# Clear the plot after display
plt.clf()  # Clear the current figure
plt.close()  # Close the figure completely to free memory

# Per year monthly averages# Apply the tests to the entire dataset
normality_results = perform_normality_tests(df)
for year, yearly_df in yearly_data.items():
    print(f"\nMonthly Average Close Prices (Year {year}):")
    monthly_avg_yearly = yearly_df.groupby(yearly_df['date'].dt.to_period("M"))['close'].mean()
    monthly_avg_yearly.plot(figsize=(20, 10), title=f"Monthly Average Close Prices (Year {year})", ylabel="Average Close Price")
    plt.show()

    # Clear the plot after display
    plt.clf()  # Clear the current figure
    plt.close()  # Close the figure completely to free memory

# Rolling averages for smoothing
print("\nRolling Average (30-Day) for Close Prices (Entire Dataset):")
df['close_rolling'] = df['close'].rolling(window=30).mean()
plt.figure(figsize=(20, 10))
plt.plot(df['date'], df['close'], label="Actual Close Price", alpha=0.5)
plt.plot(df['date'], df['close_rolling'], label="30-Day Rolling Average", color="red")
plt.title("30-Day Rolling Average of Close Prices (Entire Dataset)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Clear the plot after display
plt.clf()  # Clear the current figure
plt.close()  # Close the figure completely to free memory
```

    
    ==================================================
    TIME SERIES ANALYSIS AND SEASONALITY
    ==================================================
    Monthly Average Close Prices (Entire Dataset):



    
![png](data_analysis_files/data_analysis_28_1.png)
    


    Skipping column 'time_delta' (Subsample has zero range).
    
    Monthly Average Close Prices (Year 2020):



    
![png](data_analysis_files/data_analysis_28_3.png)
    


    
    Monthly Average Close Prices (Year 2021):



    
![png](data_analysis_files/data_analysis_28_5.png)
    


    
    Monthly Average Close Prices (Year 2022):



    
![png](data_analysis_files/data_analysis_28_7.png)
    


    
    Monthly Average Close Prices (Year 2023):



    
![png](data_analysis_files/data_analysis_28_9.png)
    


    
    Rolling Average (30-Day) for Close Prices (Entire Dataset):



    
![png](data_analysis_files/data_analysis_28_11.png)
    



```python
# Data Quality and Missing Values
print("\n" + "="*50)
print("DATA QUALITY AND MISSING VALUES")
print("="*50)

print("\nMissing Values (Entire Dataset):")
print("-"*50)
print(df.isna().sum())

# Missing values heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(df.isna(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap (Entire Dataset)")
plt.show()

# Clear the plot after display
plt.clf()  # Clear the current figure
plt.close()  # Close the figure completely to free memory

# Per year missing values summary
for year, yearly_df in yearly_data.items():
    print(f"\nMissing Values Summary (Year {year}):")
    print("-"*50)
    print(yearly_df.isna().sum())
```

    
    ==================================================
    DATA QUALITY AND MISSING VALUES
    ==================================================
    
    Missing Values (Entire Dataset):
    --------------------------------------------------
    date              0
    symbol            0
    open              0
    high              0
    low               0
    close             0
    Volume USDT       0
    tradecount        0
    token             0
    hour              0
    day               0
    ema_5             0
    ema_15            0
    ema_30            0
    ema_60            0
    ema_100           0
    ema_200           0
    WMA               0
    MACD              0
    MACD_Signal       0
    MACD_Hist         0
    ATR               0
    HMA               0
    KAMA              0
    CMO               0
    Z-Score           0
    QStick            0
    year              0
    day_of_week       0
    month             0
    week_of_year      0
    time_delta        0
    close_rolling    29
    dtype: int64



    
![png](data_analysis_files/data_analysis_29_1.png)
    


    
    Missing Values Summary (Year 2020):
    --------------------------------------------------
    date            0
    symbol          0
    open            0
    high            0
    low             0
    close           0
    Volume USDT     0
    tradecount      0
    token           0
    hour            0
    day             0
    ema_5           0
    ema_15          0
    ema_30          0
    ema_60          0
    ema_100         0
    ema_200         0
    WMA             0
    MACD            0
    MACD_Signal     0
    MACD_Hist       0
    ATR             0
    HMA             0
    KAMA            0
    CMO             0
    Z-Score         0
    QStick          0
    year            0
    day_of_week     0
    month           0
    week_of_year    0
    time_delta      0
    dtype: int64
    
    Missing Values Summary (Year 2021):
    --------------------------------------------------
    date            0
    symbol          0
    open            0
    high            0
    low             0
    close           0
    Volume USDT     0
    tradecount      0
    token           0
    hour            0
    day             0
    ema_5           0
    ema_15          0
    ema_30          0
    ema_60          0
    ema_100         0
    ema_200         0
    WMA             0
    MACD            0
    MACD_Signal     0
    MACD_Hist       0
    ATR             0
    HMA             0
    KAMA            0
    CMO             0
    Z-Score         0
    QStick          0
    year            0
    day_of_week     0
    month           0
    week_of_year    0
    time_delta      0
    dtype: int64
    
    Missing Values Summary (Year 2022):
    --------------------------------------------------
    date            0
    symbol          0
    open            0
    high            0
    low             0
    close           0
    Volume USDT     0
    tradecount      0
    token           0
    hour            0
    day             0
    ema_5           0
    ema_15          0
    ema_30          0
    ema_60          0
    ema_100         0
    ema_200         0
    WMA             0
    MACD            0
    MACD_Signal     0
    MACD_Hist       0
    ATR             0
    HMA             0
    KAMA            0
    CMO             0
    Z-Score         0
    QStick          0
    year            0
    day_of_week     0
    month           0
    week_of_year    0
    time_delta      0
    dtype: int64
    
    Missing Values Summary (Year 2023):
    --------------------------------------------------
    date            0
    symbol          0
    open            0
    high            0
    low             0
    close           0
    Volume USDT     0
    tradecount      0
    token           0
    hour            0
    day             0
    ema_5           0
    ema_15          0
    ema_30          0
    ema_60          0
    ema_100         0
    ema_200         0
    WMA             0
    MACD            0
    MACD_Signal     0
    MACD_Hist       0
    ATR             0
    HMA             0
    KAMA            0
    CMO             0
    Z-Score         0
    QStick          0
    year            0
    day_of_week     0
    month           0
    week_of_year    0
    time_delta      0
    dtype: int64



```python
# Key Observations and Insights
print("\n" + "="*50)
print("KEY OBSERVATIONS AND INSIGHTS")
print("="*50)

# Entire dataset key observations
print("\nKey Observations for the Entire Dataset:")
print("-"*50)
print(f"Highest Close Price: {df['close'].max()} on {df.loc[df['close'].idxmax(), 'date']}")
print(f"Lowest Close Price: {df['close'].min()} on {df.loc[df['close'].idxmin(), 'date']}")
print(f"Highest Volume: {df['Volume USDT'].max()} on {df.loc[df['Volume USDT'].idxmax(), 'date']}")
print(f"Most Active Day (Trades): {df['tradecount'].max()} on {df.loc[df['tradecount'].idxmax(), 'date']}")

# Per year key observations
for year, yearly_df in yearly_data.items():
    print(f"\nKey Observations for Year {year}:")
    print("-"*50)
    print(f"Highest Close Price: {yearly_df['close'].max()} on {yearly_df.loc[yearly_df['close'].idxmax(), 'date']}")
    print(f"Lowest Close Price: {yearly_df['close'].min()} on {yearly_df.loc[yearly_df['close'].idxmin(), 'date']}")
    print(f"Highest Volume: {yearly_df['Volume USDT'].max()} on {yearly_df.loc[yearly_df['Volume USDT'].idxmax(), 'date']}")
    print(f"Most Active Day (Trades): {yearly_df['tradecount'].max()} on {yearly_df.loc[yearly_df['tradecount'].idxmax(), 'date']}")
```

    
    ==================================================
    KEY OBSERVATIONS AND INSIGHTS
    ==================================================
    
    Key Observations for the Entire Dataset:
    --------------------------------------------------
    Highest Close Price: 69000.0 on 2021-11-10 14:16:00
    Lowest Close Price: 3706.96 on 2020-03-13 02:16:00
    Highest Volume: 160935802 on 2020-08-02 04:38:00
    Most Active Day (Trades): 107315 on 2023-03-14 12:30:00
    
    Key Observations for Year 2020:
    --------------------------------------------------
    Highest Close Price: 29293.14 on 2020-12-31 00:20:00
    Lowest Close Price: 3706.96 on 2020-03-13 02:16:00
    Highest Volume: 160935802 on 2020-08-02 04:38:00
    Most Active Day (Trades): 29025 on 2020-06-24 10:38:00
    
    Key Observations for Year 2021:
    --------------------------------------------------
    Highest Close Price: 69000.0 on 2021-11-10 14:16:00
    Lowest Close Price: 28235.47 on 2021-01-04 10:16:00
    Highest Volume: 115549406 on 2021-11-03 18:00:00
    Most Active Day (Trades): 55181 on 2021-11-03 18:00:00
    
    Key Observations for Year 2022:
    --------------------------------------------------
    Highest Close Price: 48164.32 on 2022-03-28 19:33:00
    Lowest Close Price: 15513.84 on 2022-11-21 21:51:00
    Highest Volume: 97391127 on 2022-05-30 21:17:00
    Most Active Day (Trades): 69114 on 2022-12-13 13:30:00
    
    Key Observations for Year 2023:
    --------------------------------------------------
    Highest Close Price: 31798.0 on 2023-07-13 19:31:00
    Lowest Close Price: 16505.87 on 2023-01-01 09:07:00
    Highest Volume: 145955668 on 2023-03-14 12:30:00
    Most Active Day (Trades): 107315 on 2023-03-14 12:30:00

