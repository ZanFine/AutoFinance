import pandas as pd
import numpy as np
import os
import datetime
import re
import math
def strategy_2(df,label_name):
    df['shift'] = df[label_name].shift(1)
    df['label'] = df['shift'] - df[label_name]
    df['label'] = df['label'].map(lambda x: 'buy' if x > 0 else 'sell')
    df = pd.DataFrame(df['label'])
    return df

def strategy_3(df,label_name):
    df['shift'] = df[label_name].shift(1)
    df['label'] = df['shift'] - df[label_name]
    df['label'] = df['label'].map(lambda x: 'buy' if x > 0 else 'sell')
    df = pd.DataFrame(df['label'])
    return df

class data_obtain():
    def __init__(self,config_row,freq = None):
        # 文件路径
        self.current_dir = os.getcwd()
        self.config=config_row
        self.stock_code = str(self.config['stock_code'])
        pattern = re.compile("[a-zA-Z]")
        if bool(pattern.search(self.stock_code)):
            self.data_source = 'yfinance'
        else:
            # 填充为6位
            self.stock_code=self.stock_code.zfill(6)

        self.beg = str(self.config['start_time'])
        self.beg=self.beg[:10]
        self.end = self.config['end_time']
        self.freq = self.config['freq']
        self.data_source = str(self.config['data_source'])

        #若结束时间为空，默认昨天
        if self.end == 'None':
            self.end = str(datetime.date.today() - datetime.timedelta(days=1))
        self.end = str(self.end)
        self.end = self.end[:10]


        #默认日线
        if self.freq == 'None':
            self.line_type = 'daily'
        else:  #分钟线的处理
            self.freq=str(self.freq)

        #数据保存读取方式
        self.method = str(self.config['save_data_type'])
        if self.method == '1':
            from connector.connect_db_offline import get_data
            self.con = get_data()

        #格式化
        self.label_name = str(self.config['label'])
        self.key_name = str(self.config['key'])

        self.is_oot = None

        #文件路径
        self.data_dir = '{}/temp_file/raw_stock_data'.format(self.current_dir)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        self.clean_data_dir = '{}/temp_file/clean_stock_data'.format(self.current_dir)


    def download(self):
        if self.data_source == 'yfinance':
            #英文code统一转换为小写，采用雅虎数据
            code_str = self.stock_code.lower()
            import yfinance as yf
            yf_class = yf.Ticker(code_str)

            if self.freq != 'None':   #todo 雅虎财经还有一个参数，proxy是代理，后面研究一下
                data = yf_class.history(start=self.beg, end=self.end,interval = self.freq)
            else:
                data = yf_class.history(start=self.beg, end=self.end)
        else:
            import efinance as ef   #todo ef还有频率一个参数，后面研究一下  fqt: 复权方式
            # klt:1 1 分钟
            # klt:5 5 分钟
            # klt:101 日
            # klt:102 周
            beg = self.beg.replace('-', '')
            end = self.end.replace('-', '')
            if self.line_type == 'daily':
                data = ef.stock.get_quote_history(self.stock_code, beg=beg, end=end)
            else:
                self.freq=int(self.freq)
                data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end,klt = self.freq)

            if data.shape[0] != 0:
                print(f'股票数据获取成功！数据来源：efinance,数据大小：{data.shape}')
                self.data_source = 'efinance'
            else:   #todo 高频数据获取和efinance异常处理
                print('efinance获取数据失败,开始从akshare获取数据')
                import akshare as ak
                data =ak.stock_zh_a_hist(symbol=self.stock_code, period='daily', start_date=self.beg, end_date=self.end,adjust='hfq')

                if data.shape[0] != 0:
                    self.data_source = 'akshare'
                    print('股票数据获取成功！数据来源：akshare')

                else:
                    print('akshare获取数据失败,开始从baostock获取数据')
                    import baostock as bs
                    #### 登陆系统 ####
                    lg = bs.login()
                    # 显示登陆返回信息
                    print('login respond error_code:' + lg.error_code)
                    print('login respond  error_msg:' + lg.error_msg)

                    stock_code = 'sh.'+self.stock_code

                    rs = bs.query_history_k_data_plus(stock_code,
                                                          "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                                          start_date=self.beg, end_date=self.end,
                                                          frequency="d", adjustflag="3")
                    stock_code = 'sz.' + self.stock_code  #TODO baostock读取代码还需要修改，所有的输入记得转化为str
                    rs = bs.query_history_k_data_plus(stock_code,
                                                      "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                                                      start_date=self.beg, end_date=self.end,
                                                      frequency="d", adjustflag="3")

                    print('query_history_k_data_plus respond error_code:' + rs.error_code)
                    print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

                    #### 打印结果集 ####
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        # 获取一条记录，将记录合并在一起
                        data_list.append(rs.get_row_data())
                    data = pd.DataFrame(data_list, columns=rs.fields)

                    if data.shape[0] != 0:
                        self.data_source = 'baostock'
                        print('股票数据获取成功！数据来源：baostock')
                    else:
                        print('akshare获取数据失败,开始从tushare获取数据')
                        import tushare as ts
                        tushare_token = input("请输入tushare包的token")
                        ts.set_token('tushare_token')
                        pro = ts.pro_api()
                        data = pro.query('daily', ts_code=self.stock_code, start_date=self.beg, end_date=self.end)

                        if data.shape[0] != 0:
                            self.data_source = 'tushare'
                            print('股票数据获取成功！数据来源：tushare')

                        else:
                            print('股票数据获取失败')
                            data=pd.DataFrame()

        data['data_source']=[self.data_source]*data.shape[0]
        data['line_type'] = [self.line_type] * data.shape[0]
        data['insert_time'] = [datetime.datetime.now()]* data.shape[0]
        if data.shape[0]!=0:
            #判断历史数据
            if self.method == '2':
                if os.path.exists(f'{self.data_dir}/{self.stock_code}.xlsx'):
                    raw_data=pd.read_excel(f'{self.data_dir}/{self.stock_code}.xlsx')
                    date_list = raw_data[self.key_name].unique()
                    date_list=date_list.tolist()
                    new_data = data[~data[self.key_name].isin(date_list)]
                    data=pd.concat([raw_data,new_data],axis=0)
                    data.to_excel(f'{self.data_dir}/{self.stock_code}.xlsx', index=False)
                    print(f'本次插入数据条数：{new_data.shape[0]}')
                else:
                    data.to_excel(f'{self.data_dir}/{self.stock_code}.xlsx',index=False)
                    print(f'本次新增数据条数：{data.shape[0]}')
            else:
                raw_data=self.con.get_data(f'select distinct date from {self.stock_code}')
                date_list = raw_data['date'].tolist()
                new_data = data[~data[self.key_name].isin(date_list)]
                self.con.to_sql(self.stock_code,new_data)
                print(f'本次数据库插入数据条数：{new_data.shape[0]}')
        return data

    def data_y_define(self):
        if self.method == '2':
            data = pd.read_excel(f'{self.data_dir}/{self.stock_code}.xlsx')
        else:
            data = self.con.get_data(f'select * from {self.stock_code}')


        #数据清洗，按时间排序，转化日期格式，日期设置为索引，索引分钟线/日线/某只股票
        data[self.key_name]=pd.to_datetime(data[self.key_name])
        data = data[data['line_type'] == self.line_type]
        beg = pd.to_datetime(self.beg)
        end = pd.to_datetime(self.end)
        data = data[(data[self.key_name] >= beg) & (data[self.key_name] <= end)]
        data = data.sort_values(self.key_name, ascending=True)

        drop_list = ['line_type','data_source','insert_time','股票代码','股票名称']
        for i in drop_list:
            if i in data.columns:
                data=data.drop(i,axis=1)
        '''
        策略选择
        '''
        strategy = str(self.config['strategy'])
        # 传统回归预测价格,预测价格
        if strategy == '1':
            data = data.rename(columns={self.label_name: 'target'})
        # 比前一天高就是好样本日
        elif strategy == '2':
            data['shift'] = data[self.label_name].shift(1)
            data['target'] = data['shift'] - data[self.label_name]
            data['target'] = data['target'].map(lambda x: 'good' if x > 0 else 'bad')
            data = data.drop(['shift',self.label_name], axis=1)
        # 滚动策略
        else:
            data['shift'] = data[self.label_name].shift(1)
            data['target'] = data['shift'] - data[self.label_name]
            data['target'] = data['target'].map(lambda x: 'buy' if x > 0 else 'sell')
            data = data.drop(['shift',self.label_name], axis=1)


        # 删除空值占比大于30%的行
        data.replace({'None': np.nan, None: np.nan}, inplace=True)
        data['null_ratio'] = data.isnull().sum(axis='columns') / data.shape[1]
        data = data.loc[data['null_ratio'] < 0.3, :].copy()
        data.drop(columns=['null_ratio'], inplace=True)
        data.to_excel(f'{self.clean_data_dir}//{self.stock_code}_total_data.xlsx', index=False)
        return data


    def data_split(self):
        '''
        划分样本集,分离x与y
        '''
        #self.is_oot控制是否划分样本外
        data = self.data_y_define()

        index_list = list(data.index)
        if type(self.config['train_len']) == float:
            train_len = self.config['train_len']
            train_point = math.floor(train_len * len(index_list))
            train_data = data.loc[index_list[:train_point], :].copy()
            #分割测试集逻辑
            if self.config['test_len'] == 'None':
                self.is_oot = False
                test_data = data.loc[index_list[train_point:], :].copy()
            else:
                if self.config['test_len']+ self.config['train_len'] == 1:
                    self.is_oot = False
                    test_data = data.loc[index_list[train_point:], :].copy()
                else:
                    self.is_oot = True
                    test_len = self.config['test_len']+ self.config['train_len']
                    test_point = math.floor(test_len * len(index_list))
                    test_data = data.loc[index_list[train_point:test_point], :].copy()
                    oot_data = data.loc[index_list[test_point:], :].copy()
        else:
            train_len=pd.to_datetime(self.config['train_len'])
            train_data = data[data[self.key_name]<=train_len].copy()
            if self.config['test_len'] == 'None':
                test_data = data[data[self.key_name]>train_len].copy()
                self.is_oot = False
            else:
                test_len = pd.to_datetime(self.config['test_len'])
                test_data = data[(data[self.key_name] > train_len)&(data[self.key_name] <= test_len)].copy()
                oot_data = data[data[self.key_name]>test_len].copy()
                if oot_data.shape[0]>0:
                    self.is_oot = True
                else:
                    self.is_oot = False
        #定义好坏,分割x和y
        y_train_df = pd.DataFrame(train_data[self.label_name])
        y_test_df = pd.DataFrame(test_data[self.label_name])
        if self.is_oot:
            x_oot_df = oot_data.drop([self.label_name], axis=1)
            y_oot_df = pd.DataFrame(oot_data[self.label_name])
        x_train_df = train_data.drop([self.label_name], axis=1)
        x_test_df = test_data.drop([self.label_name], axis=1)

        #保存数据清洗结果
        train_data.to_excel(f'{self.clean_data_dir}//{self.stock_code}_train_data.xlsx',index=False)
        test_data.to_excel(f'{self.clean_data_dir}//{self.stock_code}_test_data.xlsx',index=False)
        if self.is_oot:
            oot_data.to_excel(f'{self.clean_data_dir}//{self.stock_code}_oot_data.xlsx',index=False)




    def data_eda(self):
        data = pd.read_excel(f'{self.data_dir}/{self.stock_code}_total_data.xlsx.xlsx')
        import toad
        eda_table=toad.detect(data)[:10]
        print(eda_table)
        # 去掉不做特征的列  算iv要自己写
        data.set_index(self.key_name,inplace=True)
        iv_table = toad.quality(data, 'target')[:15]
        pass

    def data_clean(self):
        data = pd.read_excel(f'{self.clean_data_dir}/{self.stock_code}_total_data.xlsx.xlsx')

        import toad
        eda=toad.detect(data)[:10]

        # 分箱也要自己写
        c = toad.transform.Combiner()
        c.fit(data, y='target', method='chi',min_samples=0.05)  # empty_separate = False

        # 为了演示，仅展示部分分箱
        print('var_d2:', c.export()['开盘'])
        print('var_d5:', c.export()['var_d5'])
        print('var_d6:', c.export()['var_d6'])

        from toad.plot import bin_plot

        # 看'var_d2'在时间内的分箱
        col = '开盘'

        bin_plot(c.transform(data[[col, 'target']], labels=True), x=col, target='target')
        pass

    def daily_update(self):
        stock_code = self.config['stock_code']
        today =str(datetime.datetime.today().date())
        beg = today.replace('-', '')