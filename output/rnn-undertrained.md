```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, GaussianNoise, Bidirectional, Attention, MultiHeadAttention
from tensorflow.keras.optimizers import Adam, Nadam, AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from scipy.ndimage import uniform_filter1d
from functools import partial
import tensorflow as tf
import numpy as np
import os
import builtins
```

    2025-01-22 14:26:26.927731: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
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
output_path = '../export/rnn-undertrained/'
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
    - close
    - ema_15
    - ema_60
    - MACD_Hist
    - open
    - low
    - HMA
    - WMA
    - high
    - ema_30
    - KAMA
    - ema_200
    - ema_100
    - Z-Score
    
    Performing feature selection using RandomForestClassifier...
    
    Cross-validation accuracy scores: [0.99994 1.      0.99994]
    
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
# Enhanced loss function with directional profit incentive
def trading_loss(prices):
    def loss_fn(y_true, y_pred):
        # Base binary crossentropy
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Enhanced profit incentive
        direction_correct = tf.sign((y_pred - 0.5) * (y_true - 0.5))
        profit_multiplier = tf.abs(y_pred - 0.5) * 2  # 0-1 range
        profit_term = direction_correct * profit_multiplier * prices
        profit_reward = -0.3 * tf.reduce_mean(profit_term)  # Increased from 0.2 to 0.3

        # Confidence penalty adjusted
        confidence_penalty = 0.005 * tf.reduce_mean(tf.square(y_pred - 0.5))  # Reduced from 0.01

        return tf.reduce_mean(bce_loss + profit_reward + confidence_penalty)
    return loss_fn
```


```python
# Exclude date feature
X = X[:, 1:]
```


```python
# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```


```python
# Reserve the last 15% as the test set
test_size = int(0.15 * len(X_scaled))
X_test, y_test = X_scaled[-test_size:], y[-test_size:]
X_train_val, y_train_val = X_scaled[:-test_size], y[:-test_size]
```


```python
# Reshape data for LSTM input: (samples, time_steps, features)
time_steps = 30  # Number of past time steps to use for prediction
X_train_val = np.array([X_train_val[i:i + time_steps] for i in range(len(X_train_val) - time_steps)])
y_train_val = y_train_val[time_steps:]
X_test = np.array([X_test[i:i + time_steps] for i in range(len(X_test) - time_steps)])
y_test = y_test[time_steps:]
```


```python
# Time-based split using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```


```python
# Placeholder to store metrics for each split
split_metrics = []

for i, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
    print(f"\nSplit {i + 1}/{tscv.n_splits}")

    # Split the data
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    # Align prices for this split
    prices_train = prices[time_steps:time_steps + len(X_train)]
    prices_val = prices[time_steps + len(X_train):time_steps + len(X_train) + len(X_val)]

    # Balance the classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Build the RNN model
    model = Sequential([
        Input(shape=(time_steps, X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
        GaussianNoise(0.02),  # Added input noise
        Dropout(0.4),  # Increased from 0.3
        BatchNormalization(),
        Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.01))),
        Dropout(0.35),  # Adjusted from 0.3
        BatchNormalization(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Modify optimizer settings
    optimizer = AdamW(learning_rate=0.0001,  # Increased from 0.00005
                      weight_decay=1e-4,  # Increased regularization
                      clipnorm=1.0)  # Added gradient clipping

    # Compile the model
    model.compile(
        optimizer = optimizer,
        loss=trading_loss(prices_train),
        metrics=['accuracy']
    )

    # Early stopping and learning rate scheduler to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


    # Train the model with class weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=256,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, lr_scheduler],
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
```

    
    Split 1/5
    Epoch 1/10
    [1m1106/1106[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m243s[0m 217ms/step - accuracy: 0.6244 - loss: -536.7770 - val_accuracy: 0.6887 - val_loss: -927.9844 - learning_rate: 1.0000e-04
    Epoch 2/10
    [1m 123/1106[0m [32m━━[0m[37m━━━━━━━━━━━━━━━━━━[0m [1m2:38[0m 161ms/step - accuracy: 0.7188 - loss: -1068.3145


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[12], line 51
         47 lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
         50 # Train the model with class weights
    ---> 51 history = model.fit(
         52     X_train, y_train,
         53     validation_data=(X_val, y_val),
         54     epochs=10,
         55     batch_size=256,
         56     class_weight=class_weight_dict,
         57     callbacks=[early_stopping, lr_scheduler],
         58     verbose=1
         59 )
         61 # Evaluate the model on the validation set
         62 val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/keras/src/utils/traceback_utils.py:117, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        115 filtered_tb = None
        116 try:
    --> 117     return fn(*args, **kwargs)
        118 except Exception as e:
        119     filtered_tb = _process_traceback_frames(e.__traceback__)


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/keras/src/backend/tensorflow/trainer.py:368, in TensorFlowTrainer.fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)
        366 for step, iterator in epoch_iterator:
        367     callbacks.on_train_batch_begin(step)
    --> 368     logs = self.train_function(iterator)
        369     callbacks.on_train_batch_end(step, logs)
        370     if self.stop_training:


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/keras/src/backend/tensorflow/trainer.py:216, in TensorFlowTrainer._make_function.<locals>.function(iterator)
        212 def function(iterator):
        213     if isinstance(
        214         iterator, (tf.data.Iterator, tf.distribute.DistributedIterator)
        215     ):
    --> 216         opt_outputs = multi_step_on_iterator(iterator)
        217         if not opt_outputs.has_value():
        218             raise StopIteration


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/util/traceback_utils.py:150, in filter_traceback.<locals>.error_handler(*args, **kwargs)
        148 filtered_tb = None
        149 try:
    --> 150   return fn(*args, **kwargs)
        151 except Exception as e:
        152   filtered_tb = _process_traceback_frames(e.__traceback__)


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:833, in Function.__call__(self, *args, **kwds)
        830 compiler = "xla" if self._jit_compile else "nonXla"
        832 with OptionalXlaContext(self._jit_compile):
    --> 833   result = self._call(*args, **kwds)
        835 new_tracing_count = self.experimental_get_tracing_count()
        836 without_tracing = (tracing_count == new_tracing_count)


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:878, in Function._call(self, *args, **kwds)
        875 self._lock.release()
        876 # In this case we have not created variables on the first call. So we can
        877 # run the first trace but we should fail if variables are created.
    --> 878 results = tracing_compilation.call_function(
        879     args, kwds, self._variable_creation_config
        880 )
        881 if self._created_variables:
        882   raise ValueError("Creating variables on a non-first call to a function"
        883                    " decorated with tf.function.")


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:139, in call_function(args, kwargs, tracing_options)
        137 bound_args = function.function_type.bind(*args, **kwargs)
        138 flat_inputs = function.function_type.unpack_inputs(bound_args)
    --> 139 return function._call_flat(  # pylint: disable=protected-access
        140     flat_inputs, captured_inputs=function.captured_inputs
        141 )


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1322, in ConcreteFunction._call_flat(self, tensor_inputs, captured_inputs)
       1318 possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
       1319 if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
       1320     and executing_eagerly):
       1321   # No tape is watching; skip to running the function.
    -> 1322   return self._inference_function.call_preflattened(args)
       1323 forward_backward = self._select_forward_and_backward_functions(
       1324     args,
       1325     possible_gradient_type,
       1326     executing_eagerly)
       1327 forward_function, args_with_tangents = forward_backward.forward()


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:216, in AtomicFunction.call_preflattened(self, args)
        214 def call_preflattened(self, args: Sequence[core.Tensor]) -> Any:
        215   """Calls with flattened tensor inputs and returns the structured output."""
    --> 216   flat_outputs = self.call_flat(*args)
        217   return self.function_type.pack_output(flat_outputs)


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:251, in AtomicFunction.call_flat(self, *args)
        249 with record.stop_recording():
        250   if self._bound_context.executing_eagerly():
    --> 251     outputs = self._bound_context.call_function(
        252         self.name,
        253         list(args),
        254         len(self.function_type.flat_outputs),
        255     )
        256   else:
        257     outputs = make_call_op_in_graph(
        258         self,
        259         list(args),
        260         self._bound_context.function_call_options.as_attrs(),
        261     )


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/context.py:1500, in Context.call_function(self, name, tensor_inputs, num_outputs)
       1498 cancellation_context = cancellation.context()
       1499 if cancellation_context is None:
    -> 1500   outputs = execute.execute(
       1501       name.decode("utf-8"),
       1502       num_outputs=num_outputs,
       1503       inputs=tensor_inputs,
       1504       attrs=attrs,
       1505       ctx=self,
       1506   )
       1507 else:
       1508   outputs = execute.execute_with_cancellation(
       1509       name.decode("utf-8"),
       1510       num_outputs=num_outputs,
       (...)
       1514       cancellation_manager=cancellation_context,
       1515   )


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/tensorflow/python/eager/execute.py:53, in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
         51 try:
         52   ctx.ensure_initialized()
    ---> 53   tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
         54                                       inputs, attrs, num_outputs)
         55 except core._NotOkStatusException as e:
         56   if name is not None:


    KeyboardInterrupt: 



```python
# Aggregate results across splits
mean_val_accuracy = np.mean([m["val_accuracy"] for m in split_metrics])
mean_val_loss = np.mean([m["val_loss"] for m in split_metrics])

print("\n--- Cross-Validation Results ---")
print(f"Mean Validation Accuracy: {mean_val_accuracy:.4f}")
print(f"Mean Validation Loss: {mean_val_loss:.4f}")
```

    
    --- Cross-Validation Results ---
    Mean Validation Accuracy: nan
    Mean Validation Loss: nan


    /Users/panosru/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/numpy/core/fromnumeric.py:3504: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /Users/panosru/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/numpy/core/_methods.py:129: RuntimeWarning: invalid value encountered in scalar divide
      ret = ret.dtype.type(ret / rcount)



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

    Test Accuracy: 0.65



```python
# Predict probabilities for the entire test set
predicted_probas = model.predict(X_test).flatten()
```

    [1m9361/9361[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m56s[0m 6ms/step



```python
# Smooth predicted probabilities
smoothed_probas = uniform_filter1d(predicted_probas, size=5)
```


```python
print("Training set size:", X_train.shape, y_train.shape)
print("Validation set size:", X_val.shape, y_val.shape)
print("Test set size:", X_test.shape, y_test.shape)
```

    Training set size: (282934, 30, 10) (282934,)
    Validation set size: (282933, 30, 10) (282933,)
    Test set size: (299551, 30, 10) (299551,)



```python
# Slice prices to match each split
prices_train = prices[time_steps:time_steps + len(X_train)]  # Prices corresponding to training data
prices_val = prices[time_steps + len(X_train):time_steps + len(X_train) + len(X_val)]  # Validation prices
prices_test = prices[-len(X_test):]  # Remaining prices for test set
```


```python
# Adjusted thresholds
# buy_threshold = 0.8
# sell_threshold = 0.2

# Adjust thresholds based on validation data
buy_threshold = np.percentile(predicted_probas, 90)  # Try top 10%
sell_threshold = np.percentile(predicted_probas, 10)  # Try bottom 10%

print(f"Dynamic Buy Threshold: {buy_threshold}")
print(f"Dynamic Sell Threshold: {sell_threshold}")
```

    Dynamic Buy Threshold: 0.9999178647994995
    Dynamic Sell Threshold: 0.00046358644613064826



```python
# Maximum fraction of the portfolio to trade
max_trade_fraction = 0.1  # up to 10% in the most confident case
```


```python
# Minimum confidence required to trade
min_confidence = 0.3
```


```python
%run "../helpers/evaluate_tf.ipynb"
```

    [1m2341/2341[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 31ms/step - accuracy: 0.6448 - loss: -717.6374
    Test Accuracy: 64.99%
    [1m9361/9361[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m30s[0m 3ms/step
    Confusion Matrix:
    Predicted       0      1
    Actual                  
    0          141482   5108
    1          128122  24839



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    File /var/folders/11/5vttgmqd6z327wdlspbkxg940000gn/T/ipykernel_40684/2471285546.py:2
          1 # Extract accuracy and validation accuracy from the history object
    ----> 2 train_accuracy = history.history['accuracy']
          3 val_accuracy = history.history['val_accuracy']


    NameError: name 'history' is not defined



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[25], line 1
    ----> 1 get_ipython().run_line_magic('run', '"../helpers/evaluate_tf.ipynb"')


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/IPython/core/interactiveshell.py:2480, in InteractiveShell.run_line_magic(self, magic_name, line, _stack_depth)
       2478     kwargs['local_ns'] = self.get_local_scope(stack_depth)
       2479 with self.builtin_trap:
    -> 2480     result = fn(*args, **kwargs)
       2482 # The code below prevents the output from being displayed
       2483 # when using magics with decorator @output_can_be_silenced
       2484 # when the last Python token in the expression is a ';'.
       2485 if getattr(fn, magic.MAGIC_OUTPUT_CAN_BE_SILENCED, False):


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/IPython/core/magics/execution.py:741, in ExecutionMagics.run(self, parameter_s, runner, file_finder)
        739     with preserve_keys(self.shell.user_ns, '__file__'):
        740         self.shell.user_ns['__file__'] = filename
    --> 741         self.shell.safe_execfile_ipy(filename, raise_exceptions=True)
        742     return
        744 # Control the response to exit() calls made by the script being run


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/IPython/core/interactiveshell.py:3005, in InteractiveShell.safe_execfile_ipy(self, fname, shell_futures, raise_exceptions)
       3003 result = self.run_cell(cell, silent=True, shell_futures=shell_futures)
       3004 if raise_exceptions:
    -> 3005     result.raise_error()
       3006 elif not result.success:
       3007     break


    File ~/miniconda3/envs/NUPMaster/lib/python3.12/site-packages/IPython/core/interactiveshell.py:308, in ExecutionResult.raise_error(self)
        306     raise self.error_before_exec
        307 if self.error_in_exec is not None:
    --> 308     raise self.error_in_exec


        [... skipping hidden 1 frame]


    File /var/folders/11/5vttgmqd6z327wdlspbkxg940000gn/T/ipykernel_40684/2471285546.py:2
          1 # Extract accuracy and validation accuracy from the history object
    ----> 2 train_accuracy = history.history['accuracy']
          3 val_accuracy = history.history['val_accuracy']


    NameError: name 'history' is not defined



```python
# Starting portfolio values
usd_balance = 10000.0  # Starting USD balance
btc_balance = 0.0      # Starting BTC balance
buy_fee = 0.0025  # 0.25% buy fee
sell_fee = 0.004  # 0.40% sell fee

# Add risk management parameters
stop_loss_pct = 0.97  # 3% stop loss
take_profit_pct = 1.05  # 5% take profit
max_position_size = 0.25  # 25% of portfolio

# Add variables to track short positions
btc_shorted = 0.0  # Amount of BTC borrowed for short selling
short_entry_price = 0.0  # Price at which short position was opened

# Track balances and actions
usd_balances = []
btc_balances = []
actions = []
trade_percentages = []

# --- Initialize Trade Counters ---
buy_count = 0
sell_count = 0
short_count = 0
stop_loss_count = 0
take_profit_count = 0

# --- Initialize Trading Data for Plotting ---
trading_data = pd.DataFrame({
    'prices': prices_test,
    'Action': ['None'] * len(prices_test),
    'USD Balance': [0.0] * len(prices_test),
    'BTC Balance': [0.0] * len(prices_test),
    'Total Portfolio Value': [0.0] * len(prices_test)
})

# --- Initialize Balance Histories for Plotting ---
total_capital_history = []
usd_balance_history = []
btc_balance_history = []
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

    # Current price
    current_price = prices_test[t]

    # Check exit conditions first
    if btc_balance > 0:
        # Check stop loss or take profit
        current_value = btc_balance * current_price
        if current_value <= entry_value * stop_loss_pct:
            # Execute stop loss
            usd_balance += current_value * (1 - sell_fee)
            btc_balance = 0
            action = 'StopLoss (Long)'
            stop_loss_count += 1
        elif current_value >= entry_value * take_profit_pct:
            # Execute take profit
            usd_balance += current_value * (1 - sell_fee)
            btc_balance = 0
            action = 'TakeProfit (Long)'
            take_profit_count += 1

    # Check exit conditions for short positions
    if btc_shorted > 0:
        current_short_value = btc_shorted * current_price
        if current_short_value >= short_entry_value * stop_loss_pct:
            # Execute stop loss for short position
            usd_balance -= current_short_value * (1 + buy_fee)  # Buy back BTC to cover short
            btc_shorted = 0
            action = 'StopLoss (Short)'
            stop_loss_count += 1
        elif current_short_value <= short_entry_value * take_profit_pct:
            # Execute take profit for short position
            usd_balance -= current_short_value * (1 + buy_fee)  # Buy back BTC to cover short
            btc_shorted = 0
            action = 'TakeProfit (Short)'
            take_profit_count += 1

    if confidence > min_confidence and btc_balance == 0:
        # Turn confidence into a fraction of max_trade_fraction
        # e.g. if confidence=0.4, fraction_to_trade=0.2*(0.4/0.5)=0.16 (i.e. 16% of USD)
        fraction_to_trade = min(max_trade_fraction * (confidence / 0.5), max_position_size)

        # Long Entry (Buy)
        if predicted_proba > buy_threshold and usd_balance > 1e-3:
            # Buy if proba > buy_threshold. The fraction_to_trade goes from 0 to max_trade_fraction (0 to 20%)
            usd_spent = fraction_to_trade * usd_balance
            # Convert to BTC, minus the buy fee
            btc_bought = (usd_spent * (1 - buy_fee)) / current_price
            usd_balance -= usd_spent
            btc_balance += btc_bought
            entry_value = btc_balance * current_price  # Track entry value
            action = 'Buy'
            trade_percentage = fraction_to_trade  # record how much fraction we traded
            buy_count += 1

        # Short Entry (Sell)
        elif predicted_proba < sell_threshold and usd_balance > 1e-3 and btc_balance == 0:
            # Borrow BTC to sell short
            btc_shorted = fraction_to_trade * (usd_balance / current_price)
            usd_balance += btc_shorted * current_price * (1 - sell_fee)  # Receive USD from short sale
            short_entry_value = btc_shorted * current_price  # Track entry value for short position
            short_entry_price = current_price  # Track entry price for short position
            action = 'Sell (Short)'
            trade_percentage = fraction_to_trade
            short_count += 1

        # Close Long Position (Sell)        
        elif predicted_proba < sell_threshold and btc_balance > 1e-6:
            # Sell if proba < sell_threshold. fraction_to_trade is again 0 to 20% based on confidence
            btc_to_sell = fraction_to_trade * btc_balance
            usd_gained = btc_to_sell * prices_test[t] * (1 - sell_fee)
            btc_balance -= btc_to_sell
            usd_balance += usd_gained
            action = 'Sell (Long)'
            trade_percentage = fraction_to_trade  # record how much fraction we traded
            sell_count += 1

    # --- Record Data for Plotting ---
    trading_data.loc[t, 'Action'] = action
    trading_data.loc[t, 'USD Balance'] = usd_balance
    trading_data.loc[t, 'BTC Balance'] = btc_balance
    trading_data.loc[t, 'Total Portfolio Value'] = usd_balance + btc_balance * current_price
    
    # Record balances and actions
    total_capital_history.append(usd_balance + btc_balance * current_price)
    usd_balance_history.append(usd_balance)
    btc_balance_history.append(btc_balance)
```


```python
# --- Final Portfolio Status ---
final_btc_price = prices_test[-1]
remaining_btc_value = btc_balance * final_btc_price

# Close any remaining long position
if btc_balance > 0:
    final_value = btc_balance * final_btc_price * (1 - sell_fee)
    usd_balance += final_value
    btc_balance = 0

# Close any remaining short position
if btc_shorted > 0:
    final_value = btc_shorted * final_btc_price * (1 + buy_fee)
    usd_balance -= final_value
    btc_shorted = 0

# Calculate final portfolio value
total_portfolio_value = usd_balance + remaining_btc_value
profit_loss = ((total_portfolio_value - 10000) / 10000) * 100
```


```python
# Print final portfolio status
print("Final Portfolio Status:")
print(f"  USD Balance: ${usd_balance:.2f}")
print(f"  BTC Balance: {btc_balance:.6f} BTC")
print(f"  BTC Value (in USD at last price): ${remaining_btc_value:.2f}")
print(f"  Total Portfolio Value (USD): ${total_portfolio_value:.2f}")
print(f"  Profit/Loss: {profit_loss:.2f}%")
print(f"  Total Trades Executed: {buy_count + sell_count + short_count}")
print(f"    Buy Trades: {buy_count}")
print(f"    Sell Trades: {sell_count}")
print(f"    Short Trades: {short_count}")
print(f"    Stop-Loss Triggers: {stop_loss_count}")
print(f"    Take-Profit Triggers: {take_profit_count}")
```

    Final Portfolio Status:
      USD Balance: $9114.65
      BTC Balance: 0.000000 BTC
      BTC Value (in USD at last price): $920.68
      Total Portfolio Value (USD): $10035.33
      Profit/Loss: 0.35%
      Total Trades Executed: 160
        Buy Trades: 41
        Sell Trades: 0
        Short Trades: 119
        Stop-Loss Triggers: 143
        Take-Profit Triggers: 16



```python
# --- Plotting ---
import matplotlib.pyplot as plt
```


```python
# Plot prices and actions
plt.figure(figsize=(12, 6))
plt.plot(trading_data['prices'], label='Prices', color='blue', alpha=0.7, linewidth=1.5)

# Highlight trade actions
buy_indices = trading_data[trading_data['Action'] == 'Buy'].index
sell_long_indices = trading_data[trading_data['Action'].isin(['Sell (Long)', 'StopLoss (Long)', 'TakeProfit (Long)'])].index
sell_short_indices = trading_data[trading_data['Action'].isin(['Sell (Short)', 'StopLoss (Short)', 'TakeProfit (Short)'])].index

plt.scatter(buy_indices, trading_data.loc[buy_indices, 'prices'],
            color='green', label='Buy (Long)', marker='^', s=80, alpha=0.9)
plt.scatter(sell_long_indices, trading_data.loc[sell_long_indices, 'prices'],
            color='red', label='Sell/Close Long', marker='v', s=80, alpha=0.9)
plt.scatter(sell_short_indices, trading_data.loc[sell_short_indices, 'prices'],
            color='purple', label='Short/Close Short', marker='x', s=100, alpha=0.9)

plt.title('Price Action with Trading Signals', fontsize=14)
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
```


    
![png](rnn-undertrained_files/rnn-undertrained_29_0.png)
    



```python
# --- Calculate Cumulative Moving Averages ---
cumulative_average_total = pd.Series(total_capital_history).expanding(min_periods=1).mean()
cumulative_average_usd = pd.Series(usd_balance_history).expanding(min_periods=1).mean()
cumulative_average_btc = pd.Series(btc_balance_history).expanding(min_periods=1).mean()
```


```python
# --- Plot Portfolio Value Progression ---
plt.figure(figsize=(12, 6))
plt.plot(cumulative_average_total, label="Total Portfolio Value (Smoothed)", color='blue')
plt.title("Portfolio Value Over Time (Smoothed)", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Value (USD)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](rnn-undertrained_files/rnn-undertrained_31_0.png)
    



```python
# --- Plot USD Balance Progression ---
plt.figure(figsize=(12, 6))
plt.plot(cumulative_average_usd, label="USD Balance (Smoothed)", color='green', linewidth=2)
plt.axhline(y=10000, color='gray', linestyle='--', label="Initial Balance (10k USD)")
plt.title("USD Balance Progression Over Time (Smoothed)", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("USD Balance (in USD)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](rnn-undertrained_files/rnn-undertrained_32_0.png)
    



```python
# --- Plot BTC Balance Progression ---
plt.figure(figsize=(12, 6))
plt.plot(cumulative_average_btc, label="BTC Balance (Smoothed)", color='orange', linewidth=2)
plt.title("BTC Balance Progression Over Time (Smoothed)", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("BTC Balance (in BTC)", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](rnn-undertrained_files/rnn-undertrained_33_0.png)
    



```python
# Free-up memory
plt.close('all')
```


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


```python

```
