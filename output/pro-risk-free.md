```python
import builtins
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
```


```python
# Output path
output_path = '../export/simulators/pro_risk_free_trader/'
```


```python
# Run the input processing notebook to prepare input
%run "../helpers/data-processing.ipynb"
```

    Missing values in the dataset



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
      <th>Missing Count</th>
      <th>Missing Percentage</th>
      <th>Action</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>date</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>1</th>
      <td>open</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>2</th>
      <td>high</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>3</th>
      <td>low</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>4</th>
      <td>close</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Volume USDT</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>6</th>
      <td>tradecount</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ema_5</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ema_15</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ema_30</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ema_60</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ema_100</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ema_200</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>13</th>
      <td>WMA</td>
      <td>13</td>
      <td>0.000651</td>
      <td>Filled with median (26752.13)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MACD</td>
      <td>25</td>
      <td>0.001252</td>
      <td>Filled with median (-0.08)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MACD_Signal</td>
      <td>33</td>
      <td>0.001652</td>
      <td>Filled with median (0.00)</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MACD_Hist</td>
      <td>33</td>
      <td>0.001652</td>
      <td>Filled with median (-0.10)</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ATR</td>
      <td>14</td>
      <td>0.000701</td>
      <td>Filled with median (25.15)</td>
    </tr>
    <tr>
      <th>18</th>
      <td>HMA</td>
      <td>11</td>
      <td>0.000551</td>
      <td>Filled with median (26751.02)</td>
    </tr>
    <tr>
      <th>19</th>
      <td>KAMA</td>
      <td>9</td>
      <td>0.000451</td>
      <td>Filled with median (26751.57)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CMO</td>
      <td>14</td>
      <td>0.000701</td>
      <td>Filled with median (-0.12)</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Z-Score</td>
      <td>154</td>
      <td>0.007711</td>
      <td>Filled with median (-0.01)</td>
    </tr>
    <tr>
      <th>22</th>
      <td>QStick</td>
      <td>9</td>
      <td>0.000451</td>
      <td>Filled with median (0.01)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>hour</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
    <tr>
      <th>24</th>
      <td>day_of_week</td>
      <td>0</td>
      <td>0.000000</td>
      <td>No Missing Values</td>
    </tr>
  </tbody>
</table>
</div>


    Analyzing feature correlations...
    
    The following features were dropped due to high correlation (threshold: 90%):
    - ema_100
    - high
    - close
    - low
    - ema_200
    - MACD_Hist
    - ema_30
    - Z-Score
    - open
    - HMA
    - ema_60
    - ema_5
    - ema_15
    - WMA
    - KAMA
    
    Performing feature selection using RandomForestClassifier...
    
    Cross-validation accuracy scores: [1.      0.99982 0.99994]
    
    Mean accuracy: 0.9999
    
    The following features were selected based on feature importance:
    - date
    - price
    - Volume USDT
    - tradecount
    - MACD
    - MACD_Signal
    - ATR
    - CMO
    - QStick
    - price_change_ratio
    - high_low_spread
    
    Feature selection process completed.
    
    Shape of X: (1997210, 11)



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
      <th>price</th>
      <th>Volume USDT</th>
      <th>tradecount</th>
      <th>MACD</th>
      <th>MACD_Signal</th>
      <th>ATR</th>
      <th>CMO</th>
      <th>QStick</th>
      <th>price_change_ratio</th>
      <th>high_low_spread</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.577837e+09</td>
      <td>7180.720</td>
      <td>509146.0</td>
      <td>140.0</td>
      <td>0.728704</td>
      <td>-0.152219</td>
      <td>4.684925</td>
      <td>4.193879</td>
      <td>0.120</td>
      <td>0.000000</td>
      <td>3.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.577837e+09</td>
      <td>7178.470</td>
      <td>713540.0</td>
      <td>148.0</td>
      <td>0.736887</td>
      <td>-0.182091</td>
      <td>4.698380</td>
      <td>0.859360</td>
      <td>0.528</td>
      <td>-0.000313</td>
      <td>3.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.577837e+09</td>
      <td>7179.440</td>
      <td>497793.0</td>
      <td>104.0</td>
      <td>0.846578</td>
      <td>-0.117923</td>
      <td>4.609025</td>
      <td>11.466626</td>
      <td>0.493</td>
      <td>0.000135</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.577837e+09</td>
      <td>7177.175</td>
      <td>698627.0</td>
      <td>193.0</td>
      <td>0.650488</td>
      <td>-0.343494</td>
      <td>4.398181</td>
      <td>-7.962104</td>
      <td>-0.425</td>
      <td>-0.000315</td>
      <td>6.16</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.577837e+09</td>
      <td>7175.160</td>
      <td>241980.0</td>
      <td>124.0</td>
      <td>0.987398</td>
      <td>-0.092457</td>
      <td>4.262656</td>
      <td>-6.795307</td>
      <td>-0.131</td>
      <td>-0.000281</td>
      <td>3.86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1997205</th>
      <td>1.698019e+09</td>
      <td>29966.285</td>
      <td>482950.0</td>
      <td>635.0</td>
      <td>-4056.925846</td>
      <td>-1909.922487</td>
      <td>926.278065</td>
      <td>-97.689989</td>
      <td>13.189</td>
      <td>-0.000401</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>1997206</th>
      <td>1.698019e+09</td>
      <td>29970.500</td>
      <td>169682.0</td>
      <td>450.0</td>
      <td>-3657.565528</td>
      <td>-1988.042791</td>
      <td>996.882531</td>
      <td>-97.688910</td>
      <td>24.399</td>
      <td>0.000141</td>
      <td>5.83</td>
    </tr>
    <tr>
      <th>1997207</th>
      <td>1.698019e+09</td>
      <td>29975.100</td>
      <td>111271.0</td>
      <td>303.0</td>
      <td>-3095.229187</td>
      <td>-1922.717147</td>
      <td>1072.856572</td>
      <td>-97.688328</td>
      <td>32.045</td>
      <td>0.000153</td>
      <td>3.40</td>
    </tr>
    <tr>
      <th>1997208</th>
      <td>1.698019e+09</td>
      <td>29980.890</td>
      <td>169741.0</td>
      <td>631.0</td>
      <td>-2332.807178</td>
      <td>-1640.974425</td>
      <td>1154.492462</td>
      <td>-97.687019</td>
      <td>22.669</td>
      <td>0.000193</td>
      <td>8.21</td>
    </tr>
    <tr>
      <th>1997209</th>
      <td>1.698019e+09</td>
      <td>29988.730</td>
      <td>321595.0</td>
      <td>861.0</td>
      <td>-1326.581600</td>
      <td>-1044.992454</td>
      <td>1242.094190</td>
      <td>-97.685909</td>
      <td>18.319</td>
      <td>0.000261</td>
      <td>12.47</td>
    </tr>
  </tbody>
</table>
<p>1997210 rows × 11 columns</p>
</div>


    
    Shape of y: (1997210,)



    0    1
    1    0
    2    1
    3    0
    4    0
    Name: price_direction, dtype: int64



```python
# Ensure the features and target align
prices = features['price'].values   # Price column
predictions = target.values        # Binary predictions from target
```


```python
# Parameters
rolling_window = 60  # Larger rolling window for professional traders
train_window = 120  # Training window for decision tree
buy_fee = 0.0025  # 0.25% buy fee
sell_fee = 0.004  # 0.40% sell fee
lambda_risk = 0.5  # Risk penalty factor for conservative traders
risk_free_rate = 0.0002  # Assumed risk-free rate for Sharpe ratio calculation
```


```python
# Portfolio values
initial_capital = 10000.0  # USD
usd_balance = initial_capital
btc_balance = 0.0
```


```python
# Initialize tracking
usd_balances = []
btc_balances = []
actions = []
trade_percentages = []
```


```python
# Indicators
indicators = ['price_change_ratio', 'Volume USDT', 'tradecount', 'MACD', 'ATR', 'CMO', 'high_low_spread']

# Initialize export and scalers
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
scaler = MinMaxScaler()
```


```python
# Precompute rolling returns and volatility
rolling_returns = pd.Series(prices).pct_change().rolling(rolling_window).mean().fillna(0).values
rolling_volatility = pd.Series(prices).pct_change().rolling(rolling_window).std().fillna(0).values
```


```python
# Simulate trading
for t in tqdm(range(len(prices)), desc="Processing Rows", unit="rows"):
    # Default action is to hold
    action = 'None'
    trade_percentage = 0.0

    # Skip early rows where rolling calculations are not possible
    if t < max(rolling_window, train_window):
        usd_balances.append(usd_balance)
        btc_balances.append(btc_balance)
        actions.append(action)
        trade_percentages.append(trade_percentage)
        continue

    # Prepare training input using rolling window
    train_data = features.iloc[t - train_window:t][indicators].values
    train_target = target.iloc[t - train_window:t].values

    # Scale the training input
    scaled_train_data = scaler.fit_transform(train_data)  # Fit MinMaxScaler on the rolling window input
    tree_model.fit(scaled_train_data, train_target)

    # Predict action for the current row
    current_data = features.iloc[[t]][indicators].values
    scaled_current_data = scaler.transform(current_data)  # Transform current input using the same fitted scaler
    predicted_action = tree_model.predict(scaled_current_data)[0]
    predicted_proba = tree_model.predict_proba(scaled_current_data)
    buy_proba, sell_proba = (
        predicted_proba[0][1] if predicted_proba.shape[1] > 1 else 0.0,
        predicted_proba[0][0] if predicted_proba.shape[1] > 1 else 1.0,
    )

    # Risk-free conservative trader logic
    expected_return = rolling_returns[t]
    volatility = rolling_volatility[t]
    sharpe_ratio = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Profit term calculation
    profit_term = expected_return - (lambda_risk * volatility)

    # Determine action and trade percentage
    if predicted_action == 1 and usd_balance > 1e-3 and buy_proba > 0.6 and profit_term > 0:
        trade_percentage = min(0.1, usd_balance / prices[t])  # Max 10% of USD balance
        usd_spent = trade_percentage * usd_balance
        btc_bought = (usd_spent * (1 - buy_fee)) / prices[t]
        usd_balance -= usd_spent
        btc_balance += btc_bought
        action = 'Buy'

    elif predicted_action == 0 and btc_balance > 1e-6 and sell_proba > 0.6 and profit_term > 0:
        trade_percentage = min(0.1, btc_balance)  # Max 10% of BTC balance
        btc_to_sell = trade_percentage * btc_balance
        usd_gained = btc_to_sell * prices[t] * (1 - sell_fee)
        btc_balance -= btc_to_sell
        usd_balance += usd_gained
        action = 'Sell'

    # Record balances and actions
    usd_balances.append(usd_balance)
    btc_balances.append(btc_balance)
    actions.append(action)
    trade_percentages.append(trade_percentage)
```

    Processing Rows: 100%|██████████| 1997210/1997210 [37:34<00:00, 885.73rows/s]



```python
# Ensure lengths match
assert len(usd_balances) == len(prices), "USD balances length mismatch!"
assert len(btc_balances) == len(prices), "BTC balances length mismatch!"
assert len(actions) == len(prices), "Actions length mismatch!"
assert len(trade_percentages) == len(prices), "Trade percentages length mismatch!"
```


```python
# Export results to a DataFrame
builtins.data = pro_conservative_trader_df = pd.DataFrame({
    'prices': prices,
    'USD_Balance': usd_balances,
    'BTC_Balance': btc_balances,
    'Action': actions,
    'Trade_Percentage': trade_percentages,
    'Rolling_Returns': rolling_returns,
    'Rolling_Volatility': rolling_volatility,
})
```


```python
%run "../helpers/trades.ipynb"
```

    Trading Log:



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
      <th>prices</th>
      <th>USD_Balance</th>
      <th>BTC_Balance</th>
      <th>Action</th>
      <th>Trade_Percentage</th>
      <th>Rolling_Returns</th>
      <th>Rolling_Volatility</th>
      <th>Total_Capital</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7180.720</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7178.470</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7179.440</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7177.175</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7175.160</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1997205</th>
      <td>29966.285</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>None</td>
      <td>0.0</td>
      <td>-0.000015</td>
      <td>0.000337</td>
      <td>28709.555232</td>
    </tr>
    <tr>
      <th>1997206</th>
      <td>29970.500</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>None</td>
      <td>0.0</td>
      <td>-0.000014</td>
      <td>0.000337</td>
      <td>28712.616323</td>
    </tr>
    <tr>
      <th>1997207</th>
      <td>29975.100</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>None</td>
      <td>0.0</td>
      <td>-0.000015</td>
      <td>0.000337</td>
      <td>28715.957016</td>
    </tr>
    <tr>
      <th>1997208</th>
      <td>29980.890</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>None</td>
      <td>0.0</td>
      <td>-0.000024</td>
      <td>0.000322</td>
      <td>28720.161932</td>
    </tr>
    <tr>
      <th>1997209</th>
      <td>29988.730</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>None</td>
      <td>0.0</td>
      <td>-0.000028</td>
      <td>0.000318</td>
      <td>28725.855634</td>
    </tr>
  </tbody>
</table>
<p>1997210 rows × 8 columns</p>
</div>


    Processing Rows: 100%|██████████| 1997210/1997210 [00:38<00:00, 52477.76rows/s]


    Final Portfolio Status:
      USD Balance: $6946.91
      BTC Balance: 0.726238 BTC
      BTC Value (in USD at last price): $21778.94
      Total Portfolio Value (USD): $28725.86
      Profit/Loss: 187.26%
      Total Trades Executed: 1320
        Buy Trades: 970
        Sell Trades: 350



    
![png](pro-risk-free_files/pro-risk-free_12_4.png)
    



    
![png](pro-risk-free_files/pro-risk-free_12_5.png)
    



    
![png](pro-risk-free_files/pro-risk-free_12_6.png)
    



    
![png](pro-risk-free_files/pro-risk-free_12_7.png)
    



```python
%run "../helpers/testing.ipynb"
```

    Data Leakage Check
    
    Data alignment check passed.
    Correlation between predictions and future price changes:
                         predictions  future_price_change
    predictions             1.000000             0.250137
    future_price_change     0.250137             1.000000
    
    
    Feature Importance Analysis
    
    Feature Importances:
                   Feature  Importance
    9   price_change_ratio    0.941433
    8               QStick    0.022292
    7                  CMO    0.009655
    5          MACD_Signal    0.008473
    10     high_low_spread    0.005913
    6                  ATR    0.002730
    4                 MACD    0.002276
    2          Volume USDT    0.002252
    3           tradecount    0.001853
    0                 date    0.001638
    1                price    0.001486
    Permutation Importances:
                   Feature  Importance
    9   price_change_ratio    0.499862
    8               QStick    0.000049
    7                  CMO    0.000046
    5          MACD_Signal    0.000040
    10     high_low_spread    0.000036
    4                 MACD    0.000034
    0                 date    0.000027
    2          Volume USDT    0.000026
    6                  ATR    0.000026
    3           tradecount    0.000023
    1                price    0.000020
    
    
    Risk-Reward Dynamics
    
    Profit/Loss Distribution:
    count    1.997209e+06
    mean     9.376012e-03
    std      1.325444e+01
    min     -7.460352e+02
    25%     -3.763797e+00
    50%      0.000000e+00
    75%      3.764564e+00
    max      6.443523e+02
    dtype: float64
    Sharpe Ratio: -7.54
    
    
    Trading Logic Validation
    
    Profit/Loss Distribution:
    count    1.997209e+06
    mean     9.396710e-03
    std      9.009876e+00
    min     -8.557609e+02
    25%     -2.587775e+00
    50%      0.000000e+00
    75%      2.586851e+00
    max      7.609706e+02
    dtype: float64
    Sharpe Ratio: -11.10


    Processing Rows: 100%|██████████| 1997210/1997210 [00:38<00:00, 51752.16rows/s]


    
    
    Sensitivity Analysis
    



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
      <th>Initial USD Balance</th>
      <th>Initial BTC Balance</th>
      <th>Maker Fee</th>
      <th>Taker Fee</th>
      <th>Final USD Balance</th>
      <th>Final BTC Balance</th>
      <th>Total Portfolio Value (USD)</th>
      <th>Profit/Loss (%)</th>
      <th>Buy Trades</th>
      <th>Sell Trades</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10000.0</td>
      <td>0.0</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>6946.913065</td>
      <td>0.726238</td>
      <td>28725.855634</td>
      <td>187.258556</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000.0</td>
      <td>0.0</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>7321.496972</td>
      <td>0.764787</td>
      <td>30256.476016</td>
      <td>202.564760</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000.0</td>
      <td>0.0</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>5934.308777</td>
      <td>0.622182</td>
      <td>24592.760009</td>
      <td>145.927600</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>13893.826129</td>
      <td>1.452475</td>
      <td>57451.711268</td>
      <td>187.258556</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>14642.993943</td>
      <td>1.529573</td>
      <td>60512.952032</td>
      <td>202.564760</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>11868.617554</td>
      <td>1.244364</td>
      <td>49185.520018</td>
      <td>145.927600</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>6590.902443</td>
      <td>0.689020</td>
      <td>27253.732747</td>
      <td>445.074655</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>6944.338908</td>
      <td>0.725390</td>
      <td>28697.850240</td>
      <td>473.957005</td>
      <td>970</td>
      <td>350</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>5633.615210</td>
      <td>0.590656</td>
      <td>23346.635986</td>
      <td>366.932720</td>
      <td>970</td>
      <td>350</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](pro-risk-free_files/pro-risk-free_13_4.png)
    

