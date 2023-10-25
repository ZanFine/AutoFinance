import pandas as pd
class data_obtain:
    def __init__(self,stock_code,beg,end):
        self.stock_code = stock_code
        self.beg = beg
        self.end = end
        self.freq = None
        self.data_source =None
    def data_read(self, stock_name=None):  # 还是手动选择输入来源吧？
        if stock_name == None:
            import efinance as ef
            data = ef.stock.get_quote_history(self.stock_code, beg=self.beg, end=self.end)

            if data.shape[0] != 0:
                print('股票数据获取成功！数据来源：efinance')
                self.data_source = 'efinance'

            else:
                print('efinance获取数据失败,开始从akshare获取数据')
                import akshare as ak
                data =ak.stock_zh_a_hist(symbol=self.stock_code, period='daily', start_date=self.beg, end_date=self.end, adjust='hfq')

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
            data = ak.stock_zh_a_hist(symbol=self.stock_code, period='daily', start_date=self.beg, end_date=self.end,adjust='hfq')
            if data.shape[0] != 0:
                print('股票数据获取成功!')
                self.data_source = 'akshare'

            else:
                print('股票数据获取失败')

        elif stock_name == 'baostock':
            import baostock as bs
            #### 登陆系统 ####
            lg = bs.login()
            # 显示登陆返回信息
            print('login respond error_code:' + lg.error_code)
            print('login respond  error_msg:' + lg.error_msg)

            rs = bs.query_history_k_data_plus(self.stock_code,
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
                print('股票数据获取成功!')
                self.data_source = 'baostock'
            else:
                print('股票数据获取失败')

        elif stock_name == 'tushare':
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
        else:
            print('股票数据获取失败,请输入正确的第三方库名称')
            data = pd.DataFrame()  # 要不要都行

        if data.shape[0]!=0:
            path = input('请输入数据保存路径（若不保存请输入0）')
            path=str(path)
            if path != '0':
                filename = str(input('请输入文件名'))
                filetype= str(input('请输入文件后缀,csv输入1,Excel输入2'))

                temp = path + '\\' + filename + '.' + filetype
                if filetype == '1':
                    filetype ==
                    data.to_csv(temp,)
                elif filetype == '2':
                    data.to_excel(temp, )


        return data
