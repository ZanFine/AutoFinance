import pandas as pd
import numpy as np

#按时间步生成输入输出数据集
def Processing_data(array,timeStep):
    df=list()
    for i in range(len(array)-timeStep):
        a=list(array[i:i+timeStep])
        df.append(a)
    return np.array(df)

# 窗口划分
def split_windows(df,size):
    X = []
    Y = []
    for i in range(len(df) - size):
        X.append(df[i:i+size, :])
        Y.append(df[i+size, 2])
    return np.array(X), np.array(Y)

# 标准化数据
def standard(dataset):

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data

class data_process():
    def __init__(self,data,label,data_source,model_name):
        self.raw_data=data
        self.data_source = data_source
        self.model_name = model_name
        self.label=label
        self.len_Train = 0


    def basic_process(self):
        data = self.raw_data.copy()
        if self.data_source == 'efinance':
            data = data.drop(['股票名称','股票代码'], axis=1)
            rename_dict = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'}
            data = data.rename(columns=rename_dict)
            data = data[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            data['date'] = pd.to_datetime(data['date'])

        if self.data_source == 'akshare':
            rename_dict = {'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                           '成交量': 'volume', '成交额': 'amount'}
            data = data.rename(columns=rename_dict)
            data = data[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            data['date'] = pd.to_datetime(data['date'])

        if self.data_source == 'baostock':
            data = data[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            data['date'] = pd.to_datetime(data['date'])

        if self.data_source == 'tushare':
            data = data.sort_values(by='trade_date', ascending=True).reset_index(drop=True)
            rename_dict = {'trade_date': 'date','vol':'volume'}
            data = data.rename(columns=rename_dict)
            data = data[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            data['date'] = pd.to_datetime(data['date'])
        self.raw_data = data
        return self.raw_data


    #划分训练集和测试集
    def model_process(self,Train):




    def LSTM_process(self,Train):
        data = self.basic_process()
        data.set_index('date', inplace=True)

        timeStep = int(input('LSTM输入步长，默认30,默认输入0'))
        outStep = int(input('LSTM输出步长，默认1,默认输入0'))
        if timeStep== 0 :
            timeStep = 30
        if outStep == 0 :
            outStep = 1

        x = Processing_data(data[self.label], timeStep)
        y = data[self.label][timeStep:].values


        # 标准化

        from sklearn.preprocessing import StandardScaler
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        x = x_scaler.fit_transform(x)
        y = y_scaler.fit_transform(y.reshape(-1, 1))


        if type(Train) == str :
            self.len_Train = len(data[data['date'] <= Train])
        elif type(Train) == int:
            self.len_Train = int(len(data) * Train)
        else:
            self.len_Train = int(len(data) * 0.7)

        x_train, y_train, x_test, y_test = x[:self.len_Train], y[:self.len_Train], x[self.len_Train:], y[self.len_Train:]


        # 配置神经网络参数
        num_units = 256
        learning_rate = 0.01
        activation_function = 'relu'
        loss_function = 'mse'
        opt = 'adam'
        batch_size = 65
        epochs = 50
        inputColNum = 1
        from tensorflow.keras.layers import Input, Dense, LSTM, GRU, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam
        # LSTM回归模型
        lstm = self.buildLSTM(timeStep, outStep, inputColNum, opt, loss_function, activation_function, learning_rate,num_units)


        from keras.wrappers.scikit_learn import KerasClassifier
        # 构建分类模型
        lstm = KerasClassifier(build_fn=buildLSTM, verbose=0)

        from tensorflow.keras import activations
        lstm = KerasClassifier(build_fn=buildLSTM, epochs=epochs, batch_size=batch_size, verbose=0)

        activation = ['softmax', 'softplus', 'softsign', 'relu',  'sigmoid', 'hard_sigmoid', 'linear']
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
        grid = GridSearchCV(estimator=lstm, param_grid=parameters, n_jobs=1)
        grid_result = grid.fit(x_train, y_train)
        # 调优结果
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        for params, mean_score, scores in grid_result.grid_scores_:
            print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))



        lstm = self.buildLSTM(timeStep, outStep, inputColNum, opt, loss_function, activation_function, learning_rate,num_units)
        # 训练神经网络
        lstm.fit(x_train, y_train, epochs=epochs, verbose=0, batch_size=batch_size)
        yPredict = lstm.predict(x_test)

        #反归一化
        yPredict = y_scaler.inverse_transform(yPredict)
        yTest = y_scaler.inverse_transform(y_test)

        #生成一个新矩阵
        c1, c2 = [], []
        for i in range(len(yTest)):
            c1.append(yTest[i][0])
            c2.append(yPredict[i][0])
        res = pd.DataFrame({'true': c1, 'pred': c2}, index=df.index[ind + timeStep:])
        res1 = res.shift(1)
        res['true_rate'] = (res['true'] - res1['true']) / res1['true']
        res['pred_rate'] = (res['pred'] - res1['true']) / res1['true']
        res['tag'] = 0
        res['tag'][res['true_rate'] / res['pred_rate'] >= 0] = 1
        print(res.head())

        mae = MAE(res['true'], res['pred'])
        print('MAE', mae)
        accuracy = res['tag'].value_counts()[1] / len(res['tag'])
        print('accuracy', accuracy)

        # 可视化
        plt.figure(figsize=(8, 5))
        plt.plot(res.index, res['pred'], label='pred')
        plt.plot(res.index, res['true'], label='true')
        plt.title('MAE: %2f' % mae)
        plt.legend()
        plt.savefig('fig2.png', dpi=400, bbox_inches='tight')


    # 构建LSTM模型
    def buildLSTM(timeStep, outStep, inputColNum, optimizer, loss_function, activation_function, learning_rate,num_units):
        '''
        搭建LSTM网络，激活函数为tanh
        timeStep：输入时间步
        inputColNum：输入列数
        outStep：输出时间步
        learnRate：学习率
        loss_function:：损失函数
        '''

        model = Sequential()
        #两层LSTM
        model.add(LSTM(num_units, return_sequences=True, activation=activation_function, input_shape=(timeStep, inputColNum)))
        model.add(LSTM(int(num_units/2), return_sequences=False, activation=activation_function))
        #两层全连接层
        model.add(Dense(25, activation=activation_function))
        model.add(Dense(outStep,activation=activation_function))

        ###############################################
        # Add optimizer with learning rate
        if optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError('optimizer {} unrecognized'.format(optimizer))

        ##############################################

        model.compile(optimizer=opt,
                      loss=loss_function,
                      metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
        model.summary()

        return model




    def CNN(self):

    def CNN_LSTM(self):

    def logistic(self):

    def


