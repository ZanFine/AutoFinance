import pandas as pd
import numpy as np
import os
import datetime
import re
import math
class data_obtain():
    def __init__(self,freq = None):
        # 文件路径
        self.current_dir = os.getcwd()
        # 配置文件
        self.config = pd.read_excel(f'{self.current_dir}/config.xlsx')
        self.stock_code = self.config['stock_code'].item()
        self.beg = self.config['start_time'].item()
        self.end = self.config['end_time'].item()
        self.freq = self.config['freq'].item()
        self.data_source = self.config['data_source'].item()

        #若结束时间为空，默认昨天
        if math.isnan(self.end):
            self.end = str(datetime.date.today() - datetime.timedelta(days=1))

        pattern = re.compile("[a-zA-Z]")
        if bool(pattern.search(self.stock_code)):
            self.data_source = 'yfinance'

        #默认日线
        if math.isnan(self.freq):
            self.line_type = 'daily'


    def download(self):
        # 文件路径
        self.data_dir = '{}/temp_file/stock_data'.format(self.current_dir)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)

        if self.data_source == 'yfinance':
            #英文code统一转换为小写，采用雅虎数据
            code_str = self.stock_code.lower()
            import yfinance as yf
            yf_class = yf.Ticker(code_str)

            if self.freq is not None:   #todo 雅虎财经还有一个参数，proxy是代理，后面研究一下
                data = yf_class.history(start=self.beg, end=self.end,interval = self.freq)
            else:
                data = yf_class.history(start=self.beg, end=self.end)
        else:
            import efinance as ef   #todo ef还有频率一个参数，后面研究一下

            if self.line_type == 'daily':
                data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)
            else:
                data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end,freq = self.freq)

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
        if data.shape[0]!=0:
            method = self.config['save_data_type'].item()
            method=str(method)
            if method == '2':
                data.to_csv(f'{self.data_dir}/{self.stock_code}.csv',index=False)
            else:
                from connector.connect_db_offline import get_data
                con = get_data()
                con.to_sql(self.stock_code,data)
                con.close()
        print(data)


        return data
