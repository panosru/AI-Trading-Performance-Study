```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
import builtins
import os
```

    2025-01-18 17:40:34.121854: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
# Set intra-op threads (for parallelizing within a single operation)
tf.config.threading.set_intra_op_parallelism_threads(24)  # Use all 24 threads

# Set inter-op threads (for parallelizing across independent operations)
tf.config.threading.set_inter_op_parallelism_threads(2)   # Adjust based on your workload

# Verify the settings
print("Intra-op threads:", tf.config.threading.get_intra_op_parallelism_threads())
print("Inter-op threads:", tf.config.threading.get_inter_op_parallelism_threads())
```

    Intra-op threads: 24
    Inter-op threads: 2



```python
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
```


```python
# Output path
output_path = '../export/mlp/'
```


```python
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
    - ema_5
    - high
    - ema_100
    - close
    - HMA
    - Z-Score
    - ema_60
    - ema_15
    - WMA
    - MACD_Hist
    - ema_30
    - ema_200
    - open
    - KAMA
    - low
    
    Performing feature selection using RandomForestClassifier...
    
    Cross-validation accuracy scores: [0.99994 1.      1.     ]
    
    Mean accuracy: 1.0000
    
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
def trading_loss(y_true, y_pred):
    """
    Custom loss function to penalize frequent trades.
    Penalizes predictions close to 0.5 (indicating uncertainty).
    """
    # Binary crossentropy loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Trade penalty: higher penalty for predictions close to 0.5
    trade_penalty = tf.abs(y_pred - 0.5)  # Confidence score: smaller for values near 0.5
    penalty_factor = 0.1  # Adjust the trade penalty factor (tune as needed)
    total_loss = bce_loss + penalty_factor * (1 - trade_penalty)  # Higher penalty near 0.5

    return tf.reduce_mean(total_loss)
```


```python
# Exclude date feature
X = X[:, 1:]
```


```python
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```


```python
# Reserve the last 15% as the test set
test_size = int(0.15 * len(X_scaled))
X_test, y_test = X_scaled[-test_size:], y[-test_size:]
X_train_val, y_train_val = X_scaled[:-test_size], y[:-test_size]
```


```python
# Time-based split using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Placeholder to store metrics for each split
split_metrics = []

for i, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
    print(f"\nSplit {i + 1}/{tscv.n_splits}")

    # Split the data
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Balance the classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Build the MLP model (one model per split if necessary)
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        GaussianNoise(0.01),  # Add noise to the input layer
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=trading_loss,
        metrics=['accuracy']
    )

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model with class weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy for Split {i + 1}: {val_accuracy:.4f}")
    print(f"Validation Loss for Split {i + 1}: {val_loss:.4f}")

    # Save metrics for analysis
    split_metrics.append({
        "split": i + 1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

# Aggregate results across splits
mean_val_accuracy = np.mean([m["val_accuracy"] for m in split_metrics])
mean_val_loss = np.mean([m["val_loss"] for m in split_metrics])

print("\n--- Cross-Validation Results ---")
print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f}")
print(f"Mean Validation Loss: {mean_val_loss:.4f}")
```

    
    Split 1/5
    Epoch 1/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 2ms/step - accuracy: 0.5590 - loss: 0.8542 - val_accuracy: 0.6810 - val_loss: 0.7203
    Epoch 2/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.6402 - loss: 0.7243 - val_accuracy: 0.7812 - val_loss: 0.7167
    Epoch 3/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 2ms/step - accuracy: 0.6435 - loss: 0.7131 - val_accuracy: 0.5586 - val_loss: 0.7369
    Epoch 4/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.6446 - loss: 0.7089 - val_accuracy: 0.5140 - val_loss: 0.7529
    Epoch 5/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.6437 - loss: 0.7086 - val_accuracy: 0.5282 - val_loss: 0.7622
    Epoch 6/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.6448 - loss: 0.7068 - val_accuracy: 0.5938 - val_loss: 0.7207
    Epoch 7/10
    [1m2211/2211[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m4s[0m 2ms/step - accuracy: 0.6438 - loss: 0.7063 - val_accuracy: 0.5448 - val_loss: 0.7570
    Validation Accuracy for Split 1: 0.7812
    Validation Loss for Split 1: 0.7167
    
    Split 2/5
    Epoch 1/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 1ms/step - accuracy: 0.5840 - loss: 0.7955 - val_accuracy: 0.8148 - val_loss: 0.5553
    Epoch 2/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6473 - loss: 0.7113 - val_accuracy: 0.7169 - val_loss: 0.5467
    Epoch 3/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6475 - loss: 0.7035 - val_accuracy: 0.8782 - val_loss: 0.5400
    Epoch 4/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6471 - loss: 0.7029 - val_accuracy: 0.8303 - val_loss: 0.4818
    Epoch 5/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 1ms/step - accuracy: 0.6463 - loss: 0.7036 - val_accuracy: 0.5607 - val_loss: 0.7278
    Epoch 6/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6474 - loss: 0.7025 - val_accuracy: 0.5345 - val_loss: 0.8634
    Epoch 7/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6489 - loss: 0.7001 - val_accuracy: 0.9315 - val_loss: 0.5939
    Epoch 8/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6467 - loss: 0.7014 - val_accuracy: 0.9171 - val_loss: 0.4861
    Epoch 9/10
    [1m4421/4421[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 1ms/step - accuracy: 0.6475 - loss: 0.7000 - val_accuracy: 0.7172 - val_loss: 0.5496
    Validation Accuracy for Split 2: 0.8303
    Validation Loss for Split 2: 0.4818
    
    Split 3/5
    Epoch 1/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 947us/step - accuracy: 0.6209 - loss: 0.7524 - val_accuracy: 0.8523 - val_loss: 0.5954
    Epoch 2/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 898us/step - accuracy: 0.6680 - loss: 0.6833 - val_accuracy: 0.6590 - val_loss: 0.6131
    Epoch 3/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 926us/step - accuracy: 0.6665 - loss: 0.6803 - val_accuracy: 0.8996 - val_loss: 0.6353
    Epoch 4/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 986us/step - accuracy: 0.6686 - loss: 0.6779 - val_accuracy: 0.5768 - val_loss: 0.6922
    Epoch 5/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 982us/step - accuracy: 0.6679 - loss: 0.6783 - val_accuracy: 0.5376 - val_loss: 0.7306
    Epoch 6/10
    [1m6632/6632[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 911us/step - accuracy: 0.6668 - loss: 0.6791 - val_accuracy: 0.6205 - val_loss: 0.6405
    Validation Accuracy for Split 3: 0.8523
    Validation Loss for Split 3: 0.5954
    
    Split 4/5
    Epoch 1/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m9s[0m 873us/step - accuracy: 0.6147 - loss: 0.7696 - val_accuracy: 0.6042 - val_loss: 0.6652
    Epoch 2/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 899us/step - accuracy: 0.6662 - loss: 0.6847 - val_accuracy: 0.8414 - val_loss: 0.6226
    Epoch 3/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 867us/step - accuracy: 0.6661 - loss: 0.6827 - val_accuracy: 0.6710 - val_loss: 0.6983
    Epoch 4/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m8s[0m 845us/step - accuracy: 0.6648 - loss: 0.6837 - val_accuracy: 0.6096 - val_loss: 0.6822
    Epoch 5/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 833us/step - accuracy: 0.6645 - loss: 0.6842 - val_accuracy: 0.5745 - val_loss: 0.6742
    Epoch 6/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 825us/step - accuracy: 0.6648 - loss: 0.6839 - val_accuracy: 0.8880 - val_loss: 0.7186
    Epoch 7/10
    [1m8842/8842[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 839us/step - accuracy: 0.6647 - loss: 0.6844 - val_accuracy: 0.5325 - val_loss: 0.8737
    Validation Accuracy for Split 4: 0.8414
    Validation Loss for Split 4: 0.6226
    
    Split 5/5
    Epoch 1/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 1ms/step - accuracy: 0.6172 - loss: 0.7584 - val_accuracy: 0.5871 - val_loss: 0.6946
    Epoch 2/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 1ms/step - accuracy: 0.6633 - loss: 0.6883 - val_accuracy: 0.8205 - val_loss: 0.6604
    Epoch 3/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 949us/step - accuracy: 0.6622 - loss: 0.6855 - val_accuracy: 0.5780 - val_loss: 0.7127
    Epoch 4/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m12s[0m 1ms/step - accuracy: 0.6625 - loss: 0.6865 - val_accuracy: 0.5453 - val_loss: 0.7269
    Epoch 5/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 1ms/step - accuracy: 0.6622 - loss: 0.6873 - val_accuracy: 0.5130 - val_loss: 1.0078
    Epoch 6/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 1ms/step - accuracy: 0.6611 - loss: 0.6879 - val_accuracy: 0.5533 - val_loss: 0.7171
    Epoch 7/10
    [1m11053/11053[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m13s[0m 1ms/step - accuracy: 0.6624 - loss: 0.6874 - val_accuracy: 0.8008 - val_loss: 0.6652
    Validation Accuracy for Split 5: 0.8205
    Validation Loss for Split 5: 0.6604
    
    --- Cross-Validation Results ---
    Mean Validation Accuracy: 0.8251
    Mean Validation Loss: 0.6154



```python
# Prepare test dataset and evaluation variables
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(128).prefetch(tf.data.experimental.AUTOTUNE)
```


```python
X_test_evaluate = X_test  # Assign X_test for evaluation
y_test_evaluate = y_test  # Assign y_test for evaluation
```


```python
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")
```

    Test Accuracy: 0.75



```python
# Predict probabilities for the entire test set
predicted_probas = model.predict(X_test).flatten()
```

    [1m9362/9362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 284us/step



```python
# Smooth predicted probabilities
smoothed_probas = uniform_filter1d(predicted_probas, size=5)
```


```python
print("Training set size:", X_train.shape, y_train.shape)
print("Validation set size:", X_val.shape, y_val.shape)
print("Test set size:", X_test.shape, y_test.shape)
```

    Training set size: (1414691, 10) (1414691,)
    Validation set size: (282938, 10) (282938,)
    Test set size: (299581, 10) (299581,)



```python
# Slice prices to match each split
prices_train = prices[:len(X_train)]  # First 70% of prices for train
prices_val = prices[len(X_train):len(X_train) + len(X_val)]  # Next 15% for validation
prices_test = prices[len(X_train) + len(X_val):]  # Remaining 15% for test
```


```python
# Adjusted thresholds
buy_threshold = 0.8
sell_threshold = 0.2

# # Adjust thresholds based on validation data
# buy_threshold = np.percentile(predicted_probas, 90)  # Top 10% probabilities for buys
# sell_threshold = np.percentile(predicted_probas, 10)  # Bottom 10% probabilities for sells
# 
# print(f"Dynamic Buy Threshold: {buy_threshold}")
# print(f"Dynamic Sell Threshold: {sell_threshold}")
```


```python
# Maximum fraction of the portfolio to trade
max_trade_fraction = 0.1  # up to 10% in the most confident case
```


```python
%run "../helpers/evaluate_tf.ipynb"
```

    [1m2341/2341[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 598us/step - accuracy: 0.7755 - loss: 0.7077
    Test Accuracy: 74.81%
    [1m9362/9362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 284us/step
    Confusion Matrix:
    Predicted       0     1
    Actual                 
    0          146603     0
    1          151830  1148



    
![png](mlp_files/mlp_19_1.png)
    



    
![png](mlp_files/mlp_19_2.png)
    



```python
# Starting portfolio values
usd_balance = 10000.0  # Starting USD balance
btc_balance = 0.0      # Starting BTC balance
buy_fee = 0.0025  # 0.25% buy fee
sell_fee = 0.004  # 0.40% sell fee

# Track balances and actions
usd_balances = []
btc_balances = []
actions = []
trade_percentages = []
```


```python
# Loop over prices_test and use precomputed probabilities
for t in range(len(prices_test)):
    # Default action is 'None'
    action = 'None'
    trade_percentage = 0.0

    # Use precomputed probability
    predicted_proba = smoothed_probas[t]

    # Compute confidence: ranges from 0 (proba = 0.5) to 0.5 (proba = 0 or 1)
    confidence = abs(predicted_proba - 0.5)  # 0.0 → not sure, 0.5 → extremely sure

    # Turn confidence into a fraction of max_trade_fraction
    # e.g. if confidence=0.4, fraction_to_trade=0.2*(0.4/0.5)=0.16 (i.e. 16% of USD)
    fraction_to_trade = max_trade_fraction * (confidence / 0.5)

    # Decide action based on thresholds
    if predicted_proba > buy_threshold and usd_balance > 1e-3:
        # Buy if proba > buy_threshold. The fraction_to_trade goes from 0 to max_trade_fraction (0 to 20%)
        usd_spent = fraction_to_trade * usd_balance
        # Convert to BTC, minus the buy fee
        btc_bought = (usd_spent * (1 - buy_fee)) / prices_test[t]
        usd_balance -= usd_spent
        btc_balance += btc_bought
        action = 'Buy'
        trade_percentage = fraction_to_trade  # record how much fraction we traded

    elif predicted_proba < sell_threshold and btc_balance > 1e-6:
        # Sell if proba < sell_threshold. fraction_to_trade is again 0 to 20% based on confidence
        btc_to_sell = fraction_to_trade * btc_balance
        usd_gained = btc_to_sell * prices_test[t] * (1 - sell_fee)
        btc_balance -= btc_to_sell
        usd_balance += usd_gained
        action = 'Sell'
        trade_percentage = fraction_to_trade  # record how much fraction we traded

    # Record balances and actions
    usd_balances.append(usd_balance)
    btc_balances.append(btc_balance)
    actions.append(action)
    trade_percentages.append(trade_percentage)
```


```python
# Final portfolio status
final_btc_price = prices_test[-1]
remaining_btc_value = btc_balance * final_btc_price
total_portfolio_value = usd_balance + remaining_btc_value
profit_loss = ((total_portfolio_value - 10000) / 10000) * 100

print("Final Portfolio Status:")
print(f"  USD Balance: ${usd_balance:.2f}")
print(f"  BTC Balance: {btc_balance:.6f} BTC")
print(f"  BTC Value (in USD at last price): ${remaining_btc_value:.2f}")
print(f"  Total Portfolio Value (USD): ${total_portfolio_value:.2f}")
print(f"  Profit/Loss: {profit_loss:.2f}%")
```

    Final Portfolio Status:
      USD Balance: $9008.81
      BTC Balance: 0.065726 BTC
      BTC Value (in USD at last price): $1971.03
      Total Portfolio Value (USD): $10979.83
      Profit/Loss: 9.80%



```python
builtins.data = pd.DataFrame({
    'prices': prices_test,
    'USD_Balance': usd_balances,
    'BTC_Balance': btc_balances,
    'Action': actions,
    'Trade_Percentage': trade_percentages,
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
      <th>Total_Capital</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>27409.745</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27419.005</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>27430.025</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27410.065</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27394.345</td>
      <td>10000.000000</td>
      <td>0.000000</td>
      <td>None</td>
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
    </tr>
    <tr>
      <th>299576</th>
      <td>29966.285</td>
      <td>8729.248036</td>
      <td>0.075090</td>
      <td>Sell</td>
      <td>0.070282</td>
      <td>10979.423136</td>
    </tr>
    <tr>
      <th>299577</th>
      <td>29970.500</td>
      <td>8879.925510</td>
      <td>0.070043</td>
      <td>Sell</td>
      <td>0.067222</td>
      <td>10979.134511</td>
    </tr>
    <tr>
      <th>299578</th>
      <td>29975.100</td>
      <td>9008.806501</td>
      <td>0.065726</td>
      <td>Sell</td>
      <td>0.061632</td>
      <td>10978.939112</td>
    </tr>
    <tr>
      <th>299579</th>
      <td>29980.890</td>
      <td>9008.806501</td>
      <td>0.065726</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10979.319664</td>
    </tr>
    <tr>
      <th>299580</th>
      <td>29988.730</td>
      <td>9008.806501</td>
      <td>0.065726</td>
      <td>None</td>
      <td>0.000000</td>
      <td>10979.834953</td>
    </tr>
  </tbody>
</table>
<p>299581 rows × 6 columns</p>
</div>


    Processing Rows: 100%|██████████| 299581/299581 [00:05<00:00, 52002.61rows/s]


    Final Portfolio Status:
      USD Balance: $9008.81
      BTC Balance: 0.065726 BTC
      BTC Value (in USD at last price): $1971.03
      Total Portfolio Value (USD): $10979.83
      Profit/Loss: 9.80%
      Total Trades Executed: 408
        Buy Trades: 135
        Sell Trades: 273



    
![png](mlp_files/mlp_24_4.png)
    



    
![png](mlp_files/mlp_24_5.png)
    



    
![png](mlp_files/mlp_24_6.png)
    



    
![png](mlp_files/mlp_24_7.png)
    



```python
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)

# Placeholder for your dataset (X, y)
# Replace these with your actual input_data
# X_train, X_val, X_test, y_train, y_val, y_test should already be defined

# Test set evaluation
def evaluate_on_test_set(model, X_test, y_test):
    """Evaluate the model on the test set and return metrics."""
    y_pred = model.predict(X_test)
    y_test = np.array(y_test).astype(int)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': test_accuracy,
        'report': report,
        'confusion_matrix': cm
    }

# Confidence analysis
def analyze_prediction_confidence(model, X_test, y_test):
    """Analyze prediction confidence and return plots."""
    try:
        predicted_proba = model.predict_proba(X_test)
        confidence_scores = np.max(predicted_proba, axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(confidence_scores, bins=20, color='blue', alpha=0.7, edgecolor='black')
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Confidence")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.6)
        plt.show()

        correct_predictions = (model.predict(X_test) > 0.5).astype(int).flatten() == y_test
        plt.figure(figsize=(10, 6))
        plt.scatter(confidence_scores, correct_predictions, alpha=0.3, color='orange')
        plt.title("Confidence vs Correctness")
        plt.xlabel("Confidence")
        plt.ylabel("Correct Prediction")
        plt.grid(alpha=0.6)
        plt.show()
    except AttributeError:
        print("Model does not have a predict_proba method.")

# Data leakage check
def check_data_leakage(X_train, X_test):
    """Check for input_data leakage between training and test sets."""
    if not isinstance(X_train, (pd.DataFrame, np.ndarray)) or not isinstance(X_test, (pd.DataFrame, np.ndarray)):
        print("X_train and X_test should be DataFrames or NumPy arrays for leakage check.")
        return

    # Convert to DataFrames if they are NumPy arrays
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    common_features = set(X_train.columns) & set(X_test.columns)
    if len(common_features) != X_train.shape[1]:
        print("Warning: Feature mismatch between train and test sets.")
    else:
        print("No apparent input_data leakage detected in features.")

# Evaluate robustness with noisy input_data
def evaluate_with_noise(model, X_test, y_test, noise_level=0.01):
    """Evaluate model's robustness with noisy input_data and return accuracy."""
    if not isinstance(X_test, np.ndarray):
        try:
            X_test = X_test.values  # Convert to numpy array if possible
        except AttributeError:
            print("X_test cannot be converted to a numpy array.")
            return None
    X_noisy = X_test + noise_level * np.random.normal(size=X_test.shape)
    y_pred_noisy = (model.predict(X_noisy) > 0.5).astype(int).flatten()
    noisy_accuracy = accuracy_score(y_test, y_pred_noisy)
    return noisy_accuracy

# Run all checks
def check_model_overfitting(model, X_train, X_val, X_test, y_train, y_val, y_test, noise_level=0.01):
    """Run all evaluations to check for model overfitting."""
    print("\n--- Evaluating on Test Set ---")
    test_results = evaluate_on_test_set(model, X_test, y_test)
    print(f"Test Set Accuracy: {test_results['accuracy']}")
    print("Classification Report (Test Set):\n", test_results['report'])
    print("Confusion Matrix (Test Set):\n", test_results['confusion_matrix'])

    print("\n--- Analyzing Prediction Confidence ---")
    analyze_prediction_confidence(model, X_test, y_test)

    print("\n--- Checking for Data Leakage ---")
    check_data_leakage(X_train, X_test)

    print("\n--- Evaluating with Noise ---")
    noisy_accuracy = evaluate_with_noise(model, X_test, y_test, noise_level)
    print(f"Accuracy with {noise_level * 100:.1f}% Noise:", noisy_accuracy)

# Usage example (replace `model` with your trained model)
noise_level = 0.01  # Define the noise level
check_model_overfitting(model, X_train, X_val, X_test, y_train, y_val, y_test, noise_level)
```

    
    --- Evaluating on Test Set ---
    [1m9362/9362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 280us/step
    Test Set Accuracy: 0.7480948391253117
    Classification Report (Test Set):
                   precision    recall  f1-score   support
    
               0       0.67      0.94      0.78    146603
               1       0.91      0.56      0.70    152978
    
        accuracy                           0.75    299581
       macro avg       0.79      0.75      0.74    299581
    weighted avg       0.79      0.75      0.74    299581
    
    Confusion Matrix (Test Set):
     [[137726   8877]
     [ 66589  86389]]
    
    --- Analyzing Prediction Confidence ---
    Model does not have a predict_proba method.
    
    --- Checking for Data Leakage ---
    No apparent input_data leakage detected in features.
    
    --- Evaluating with Noise ---
    [1m9362/9362[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m3s[0m 278us/step
    Accuracy with 1.0% Noise: 0.5836718617001746



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
    count    299580.000000
    mean          0.003271
    std           1.408453
    min         -81.152782
    25%          -0.332713
    50%           0.000000
    75%           0.337344
    max         152.982029
    dtype: float64
    Sharpe Ratio: -71.00
    
    
    Trading Logic Validation
    
    Profit/Loss Distribution:
    count    1.997209e+06
    mean     9.833487e-03
    std      9.624177e+00
    min     -6.567851e+02
    25%     -2.726032e+00
    50%      0.000000e+00
    75%      2.732253e+00
    max      4.934476e+02
    dtype: float64
    Sharpe Ratio: -10.39


    Processing Rows: 100%|██████████| 299581/299581 [00:05<00:00, 51152.92rows/s]

    
    
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
      <td>9008.806501</td>
      <td>0.065726</td>
      <td>10979.834953</td>
      <td>9.798350</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000.0</td>
      <td>0.0</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>9110.116964</td>
      <td>0.066423</td>
      <td>11102.043387</td>
      <td>11.020434</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10000.0</td>
      <td>0.0</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>8711.450424</td>
      <td>0.063710</td>
      <td>10622.026395</td>
      <td>6.220264</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>18017.613002</td>
      <td>0.131451</td>
      <td>21959.669905</td>
      <td>9.798350</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>18220.233928</td>
      <td>0.132845</td>
      <td>22204.086775</td>
      <td>11.020434</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20000.0</td>
      <td>0.0</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>17422.900848</td>
      <td>0.127420</td>
      <td>21244.052790</td>
      <td>6.220264</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0025</td>
      <td>0.0040</td>
      <td>16880.086501</td>
      <td>0.123152</td>
      <td>20573.265032</td>
      <td>311.465301</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0015</td>
      <td>0.0030</td>
      <td>17076.111418</td>
      <td>0.124503</td>
      <td>20809.801959</td>
      <td>316.196039</td>
      <td>135</td>
      <td>273</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5000.0</td>
      <td>0.5</td>
      <td>0.0050</td>
      <td>0.0075</td>
      <td>16299.115393</td>
      <td>0.119201</td>
      <td>19873.801203</td>
      <td>297.476024</td>
      <td>135</td>
      <td>273</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](mlp_files/mlp_26_5.png)
    

