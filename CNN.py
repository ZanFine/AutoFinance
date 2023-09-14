import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


# 窗口划分
def split_windows(data, size):
    X = []
    Y = []
    for i in range(len(data) - size):
        X.append(data[i:i+size, :])
        Y.append(data[i+size, 2])
    return np.array(X), np.array(Y)


df= pd.read_csv('E:\数据建模\量化策略\三一重工.csv',encoding='UTF-8',usecols=['open', 'high', 'close', 'low', 'volume'])
all_data = df.values
train_len = 500
train_data = all_data[:train_len, :]
test_data = all_data[train_len:, :]

# plt.figure(figsize=(12, 8))
# plt.plot(np.arange(train_data.shape[0]), train_data[:, 2], label='train data')
# plt.plot(np.arange(train_data.shape[0], train_data.shape[0] + test_data.shape[0]), test_data[:, 2], label='test data')
# plt.legend()


#数据归一化
# normalizatioin processing
scaler = MinMaxScaler()
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)
# 训练集测试集划分
window_size = 7
train_X, train_Y = split_windows(scaled_train_data, size=window_size)
test_X, test_Y = split_windows(scaled_test_data, size=window_size)
print('train shape', train_X.shape, train_Y.shape)
print('test shape', test_X.shape, test_Y.shape)

#模型搭建
window_size = 7
fea_num = 5

model = keras.models.Sequential([
    keras.layers.Input((window_size, fea_num)),
    keras.layers.Reshape((window_size, fea_num, 1)),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        activation="relu"),
    keras.layers.MaxPooling2D(pool_size=2, strides=1, padding="same"),
    keras.layers.Dropout(0.3),
    keras.layers.Reshape((window_size, -1)),
    keras.layers.LSTM(128, return_sequences=True),
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.summary()

model.fit(x=train_X, y=train_Y, epochs=50)
#模型评估
prediction = model.predict(test_X)
scaled_prediction = prediction * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]
scaled_true = test_Y * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]
plt.plot(range(len(scaled_prediction)), scaled_prediction, label='true')
plt.plot(range(len(scaled_true)), scaled_true, label='prediction', marker='*')
plt.legend()

from sklearn.metrics import mean_squared_error

print('RMSE', np.sqrt(mean_squared_error(scaled_prediction, scaled_true)))