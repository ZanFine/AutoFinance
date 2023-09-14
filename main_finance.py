import pandas as pd
import  numpy as np

class Autofinance:
    def __init__(self):
        self.stock_code = None
        self.beg = None
        self.end = None
        self.freq = None
        self.data_source =None

    def logistic(self):

    def CNN(self):

    def svc(self):

    def arima(self):

    def har(self):

    def cnn(self):

    def lstm(self):

    def data_deal(self):

    def draw(self):

    def help(self,name):
        if name =='':
            print('逻辑斯蒂回归说明')
        elif name == '':
            print('ARIMA说明')

    def data_read(self,stock_name = None):  #还是手动选择输入来源吧？
        if stock_name == None:
            import efinance as ef
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)

            if data.shape[0] != 0:
                print('股票数据获取成功！数据来源：efinance')
                self.data_source = 'efinance'

            else:
                print('efinance获取数据失败,开始从akshare获取数据')
                import akshare as ak
                data =

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

                    rs = bs.query_history_k_data_plus(self.stock_code,"date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
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
                    else :
                        print('akshare获取数据失败,开始从tushare获取数据')
                        import tushare as ts
                        tushare_token=input("请输入tushare包的token")
                        ts.set_token('tushare_token')
                        pro = ts.pro_api()
                        data = pro.query('daily', ts_code=self.stock_code, start_date=self.beg, end_date=self.end)

                        if data.shape[0] != 0:
                            self.data_source = 'tushare'
                            print('股票数据获取成功！数据来源：tushare')

                        else:
                            print('股票数据获取失败')

        elif stock_name == 'efinance':
            import efinance as ef
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)
            if data.shape[0] != 0:
                print('股票数据获取成功!')
                self.data_source = 'efinance'

            else:
                print('股票数据获取失败')

        elif stock_name == 'akshare':
            import akshare as ak
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)
            if data.shape[0] != 0:
                print('股票数据获取成功!')
                self.data_source = 'akshare'

            else:
                print('股票数据获取失败')

        elif stock_name == 'baostock':
            import akshare ak
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)
            if data.shape[0] != 0:
                print('股票数据获取成功!')
                self.data_source = 'baostock'

            else:
                print('股票数据获取失败')


        elif stock_name == 'tushare':
            import akshare
            ak
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)
            if data.shape[0] != 0:
                print('股票数据获取成功!')
                self.data_source = 'baostock'

            else:
                print('股票数据获取失败')

        else:
            print('股票数据获取失败,请输入正确的第三方库名称')
            data = pd.DataFrame()    #要不要都行

        return data


    def process(self):
        self.stock_code = input("请输入分析股票代码")
        self.beg = input("请输入数据开始时间")
        self.end = input('请输入数据结束时间')

        #后期freq要不要加入呢  ToDo
        print('开始获取数据')
        data =




