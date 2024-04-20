# coding:utf-8
import math
import random
import warnings
import numpy as np
import traceback

import time
import pandas as pd
import os
from datetime import datetime,timedelta

from functools import partial
import multiprocessing as mp
from machine_learning.lr_train import train
from machine_learning.data_deal import data_deal

warnings.filterwarnings("ignore")


class logistics_model():
    def __init__(self, db_connector,type='bad', bin_type='chi_bin',random_state = None):
        #数据是否随机
        self.random_state= random_state
        #数据库连接类
        self.db_connector=db_connector
        self.type = type
        self.bin_type = bin_type
        # 数据类型
        # 文件路径
        self.current_dir = os.getcwd()
        self.data_dir = '{}/private_file/{}'.format(self.current_dir,self.type)

        self.target_v = 'is_bad'

        self.config = '{}/config.xlsx'.format(self.data_dir)

        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.draw_dir = '{}\\draw'.format(self.data_dir)
        if not os.path.isdir(self.draw_dir):
            os.makedirs(self.draw_dir)
        self.filter_variable_file = '{}\\1_variable_filter_basic.csv'.format(self.data_dir)
        self.tree_rule_file = '{}\\2_feature_in_pair.feather'.format(self.data_dir)

        self.rule_data = '{}\\3_rule_data.csv'.format(self.data_dir)
        self.corr_data = '{}\\4_corr_data.csv'.format(self.data_dir)
        self.filter_variable_update = '{}\\4_filter_variable_update.csv'.format(self.data_dir)

        self.ml_record_file = self.data_dir + '\\5_machine_learnning_record.csv'
        self.predict_file = self.data_dir + '\\6_1_predict_file.csv'
        self.predict_stat_file = self.data_dir + '\\6_2_predict_stat_file.csv'
        self.final_param_file = self.data_dir + '\\6_3_model_param.csv'
        self.score_file = self.data_dir + '\\6_4_score_file.xlsx'
        self.practice_data_file = self.data_dir + '\\6_5_practice_data.csv'
        self.lift_file = self.data_dir + '\\6_5_lift_data.csv'

    def download_model_data(self, feature_sql,trade_sql):
        '''
        从py_feature_monitory抽取特征列表
        :param feature_sql: 抽取特征列表的sql
        :param trade_sql:读取trade表中有贷后表现的sql
        :return:
        '''
        pd.set_option('display.width', None)
        date_today = datetime.now().strftime('%Y-%m-%d')
        # 获取特征值
        # select_sql='select * from py_feature_monitory where operator_create="Dave"  and is_deleted!=1 and importance = 1'

        feature_data = self.db_connector.get_data(feature_sql)
        data = self.db_connector.get_data(trade_sql)

        trade_no = data['trade_no']
        trade_no = trade_no.map(lambda x: '"{}"'.format(x))

        try:
            # 查找对应的数据
            trade_sheet_list = list(feature_data['data_source'].unique())
            for sheet in trade_sheet_list:
                keys = list(feature_data.loc[feature_data['data_source'] == sheet, 'keys'])[0]
                feature_list = list(feature_data.loc[feature_data['data_source'] == sheet, 'feature_name'])
                feature_list.append(keys)
                select_sql = 'select {} from {} where trade_no in ({})'.format(','.join(feature_list), sheet,
                                                                               ','.join(trade_no))
                s_data = self.db_connector.get_data(select_sql)
                data = data.merge(s_data, on=keys, how='left', suffixes=['py_data_trade', sheet])

        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
        # 删除空值占比大于30%的行
        data.replace({'None': np.nan, None: np.nan}, inplace=True)

        data['null_ratio'] = data.isnull().sum(axis='columns') / data.shape[1]

        data = data.loc[data['null_ratio'] < 0.3, :].copy()
        data.drop(columns=['null_ratio'], inplace=True)
        # 生成配置文件
        dtype_data = pd.DataFrame(data.dtypes, columns=['dtype'])
        dtype_data.index.name = 'variable'
        dtype_data.reset_index(drop=False, inplace=True)
        dtype_data = dtype_data.loc[dtype_data['variable'] != 'trade_no', :].copy()
        dtype_data['model_in'] = [1] * dtype_data.shape[0]
        dtype_data['comment'] = ['self_constructed'] * dtype_data.shape[0]
        # 合并source和chi_bins
        raw_feature_config = feature_data.loc[:,
                             ['feature_name', 'data_source', 'bins', 'fillna', 'keys', 'type']].copy()
        raw_feature_config.rename(columns={'feature_name': 'variable'}, inplace=True)
        dtype_data = dtype_data.merge(raw_feature_config, on='variable', how='left')
        # 将没有来源的特征改成不跑模型
        dtype_data.loc[dtype_data['data_source'].isnull(), 'model_in'] = 0
        data.to_excel('{}/model.xlsx'.format(self.data_dir), index=False)
        dtype_data.to_excel('{}/config.xlsx'.format(self.data_dir), index=False)

    def download_all_feature(self, table_list, trade_sql):
        '''
        下载要分析的所有特征表
        :param table_list: 特征表的列表
        :param trade_sql: 目标变量的sql
        :return:
        '''

        pd.set_option('display.width', None)

        target_data = self.db_connector.get_data(trade_sql)

        trade_no = target_data['trade_no']
        trade_no = trade_no.map(lambda x: '"{}"'.format(x))
        dtype_data=pd.DataFrame()
        data = pd.DataFrame()
        source_dict = {}

        for raw_table in table_list:
            select_sql = 'select * from {} where trade_no in ({})'.format(raw_table, ','.join(trade_no))
            temp_data = self.db_connector.get_data(select_sql)
            #列名
            columns_list=list(temp_data.columns)
            if 'trade_no' in columns_list:
                columns_list.remove('trade_no')
            rename_list=['feature{}'.format(x) for x in range(dtype_data.shape[0],dtype_data.shape[0]+len(columns_list))]
            new_dtype=pd.DataFrame({
                'raw_name':columns_list,
                'variable':rename_list,
                'data_source':[raw_table]*len(columns_list)
            })
            if dtype_data.shape[0]==0:
                dtype_data=new_dtype.copy()
            else:
                dtype_data=pd.concat([dtype_data,new_dtype])
            #更名
            temp_data.rename(columns=dict(zip(columns_list,rename_list)),inplace=True)
            if data.shape[0] == 0:
                data = temp_data.copy()
            else:
                data = pd.merge(data, temp_data, on='trade_no', how='outer')
            print()
        # 删除空值占比大于30%的行
        data.replace({'None': np.nan, None: np.nan}, inplace=True)

        data['null_ratio'] = data.isnull().sum(axis='columns') / data.shape[1]
        print(data)

        data = data.loc[data['null_ratio'] < 0.3, :].copy()
        data.drop(columns=['null_ratio'], inplace=True)
        # 生成配置文件
        s_dtype = pd.DataFrame(data.dtypes, columns=['dtype'])
        s_dtype.index.name = 'variable'
        s_dtype.reset_index(drop=False, inplace=True)
        dtype_data=dtype_data.merge(s_dtype,on='variable',how='left')
        print(dtype_data)
        dtype_data = dtype_data.loc[dtype_data['variable'] != 'trade_no', :].copy()
        dtype_data['model_in'] = [0] * dtype_data.shape[0]
        dtype_data['comment'] = ['self_constructed'] * dtype_data.shape[0]
        dtype_data['fillna'] = [-1] * dtype_data.shape[0]
        dtype_data['type'] = [0] * dtype_data.shape[0]
        dtype_data['bins'] = [np.nan] * dtype_data.shape[0]

        print(dtype_data)
        dtype_data['dtype'] = dtype_data['dtype'].astype(str)
        dtype_data.loc[dtype_data['dtype'].isin(['int64', 'float64']), 'model_in'] = 1
        # 合并source和chi_bins
        # 将没有来源的特征改成不跑模型
        data = data.merge(target_data, on='trade_no', how='left')
        data.to_excel('{}/model.xlsx'.format(self.data_dir), index=False)
        dtype_data.to_excel('{}/config.xlsx'.format(self.data_dir), index=False)

    def get_clean_data(self, data_type='total'):
        time_now = datetime.now().strftime('%Y%m%d')
        current_dir = os.getcwd()

        merged_data_dir = '{}/private_file/{}'.format(current_dir, self.type)
        if not os.path.isdir(merged_data_dir):
            os.makedirs(merged_data_dir)
        data_file = '{}/model.xlsx'.format(merged_data_dir)
        config_file = '{}/config.xlsx'.format(merged_data_dir)
        data = pd.read_excel(data_file)
        data.rename(columns={'order_status': 'is_bad', 'trade_no': 'quota_no'}, inplace=True)
        data['is_bad'] = data['is_bad'].map({6: 1, 7: 0})
        data['is_bad'].fillna(value=1, inplace=True)
        # 处理空值
        config = pd.read_excel(config_file)
        config['model_in'] = config['model_in'].astype(np.int64)
        config = config.loc[config['model_in'] == 1, :].copy()
        # 清除其他类型的数据
        data.replace({'None': np.nan}, inplace=True)

        data['repayment_date'] = pd.to_datetime(data['repayment_date'])
        data.sort_values(by='repayment_date', ascending=True, inplace=True)
        for i in list(config.index):
            v = config.loc[i, 'variable']
            fillna = config.loc[i, 'fillna']
            data[v].fillna(value=fillna, inplace=True)

        data.drop_duplicates(subset='quota_no', keep='last', inplace=True)
        data.reset_index(drop=True,inplace=True)
        index_list = list(data.index)
        train_point = math.floor(0.6 * len(index_list))
        test_point = math.floor(0.8 * len(index_list))
        if type(self.random_state) == int:
            data = data.sample(frac=1, random_state=self.random_state)

        if data_type == 'total':
            pass
        elif data_type == 'train':
            data = data.loc[index_list[:train_point], :].copy()
        elif data_type == 'test':
            data = data.loc[index_list[train_point:test_point], :].copy()
        elif data_type == 'out':
            data = data.loc[index_list[test_point:], :].copy()


        # data.to_excel('{}/{}.xlsx'.format(merged_data_dir,data_type))
        return data
    # 筛选psi小于0.2的特征
    def variable_filter_basic(self):
        data = self.get_clean_data( data_type='train')
        print(data)
        dd = data_deal()
        variable_record = pd.read_excel(self.config)
        variable_record = variable_record.loc[variable_record['model_in'] == 1, :].copy()
        if 'max_bins' not in list(variable_record.columns):
            variable_record['max_bins'] = [0] * variable_record.shape[0]
        variable_record.drop_duplicates(subset='variable', inplace=True)
        variable_record.set_index('variable', inplace=True)

        quantity_list = list(variable_record.loc[(variable_record['type'] == 0) \
                                                 & (variable_record['model_in'] == 1), :].index)

        # data[quantity_list] = data[quantity_list].astype(np.float64)
        quality_list = list(variable_record.loc[(variable_record['type'] == 1) \
                                                & (variable_record['model_in'] == 1), :].index)
        for v in quantity_list:
            try:

                if '[' in str(variable_record.loc[v, 'bins']) and self.bin_type == 'chi_bin':
                    bins = eval(variable_record.loc[v, 'bins'])
                else:
                    if variable_record.loc[v, 'max_bins'] > 0:
                        if self.bin_type == 'chi_bin':
                            bins = dd.chi_bins(data[v], data[self.target_v], 0.95, variable_record.loc[v, 'max_bins'])
                        else:
                            bins = dd.same_freq_bins(data[v], data[self.target_v], 0.95,
                                                     variable_record.loc[v, 'max_bins'])
                    else:
                        if self.bin_type == 'chi_bin':
                            bins = dd.chi_bins(data[v], data[self.target_v], 0.95, variable_record.loc[v, 'max_bins'])
                        else:
                            bins = dd.same_freq_bins(data[v], data[self.target_v], 0.95,
                                                     variable_record.loc[v, 'max_bins'])

                cut_x = pd.cut(data[v], bins=bins, labels=bins[1:])
                variable_record.loc[v, 'consistency'] = dd.get_consistency(cut_x, data[self.target_v])
                result = dd.get_woe_iv(cut_x, data[self.target_v])
                propotion = cut_x.value_counts(normalize=True)
                propotion_dict = dict(zip(list(propotion.index), list(propotion.values)))
                variable_record.loc[v, 'iv'] = result['iv']
                variable_record.loc[v, 'woe_dict'] = str(result['woe_dict'])
                variable_record.loc[v, 'min_woe'] = str(min(result['woe_dict'].values()))
                variable_record.loc[v, 'bad_rate_dict'] = str(result['bad_rate_dict'])
                min_bad_rate = min(result['bad_rate_dict'].values())
                variable_record.loc[v, 'min_bad_rate'] = min_bad_rate

                max_bad_rate = max(result['bad_rate_dict'].values())
                variable_record.loc[v, 'max_bad_rate'] = max_bad_rate

                min_bad_rate_index = list(result['bad_rate_dict'].keys())[
                    list(result['bad_rate_dict'].values()).index(min_bad_rate)]
                variable_record.loc[v, 'min_bad_rate_propotion'] = propotion_dict[min_bad_rate_index]

                max_bad_rate_index = list(result['bad_rate_dict'].keys())[
                    list(result['bad_rate_dict'].values()).index(max_bad_rate)]
                variable_record.loc[v, 'max_bad_rate_propotion'] = propotion_dict[max_bad_rate_index]
                variable_record.loc[v, 'bins'] = str(bins)
                psi = dd.compute_psi(cut_x, data[self.target_v])
                monotony = dd.compute_monotony(cut_x, data[self.target_v])
                variable_record.loc[v, 'psi'] = psi
                variable_record.loc[v, 'monotony'] = monotony
                variable_record.loc[v, 'propotion_dict'] = str(propotion_dict)
                # if result['iv'] > iv_threshold:
                #     dd.draw('quantity', self.draw_dir, '{}_{}'.format(round(result['iv'],4),v), cut_x, data[self.target_v])
            except Exception as e:
                msg = traceback.format_exc()
                print(msg)

        for v in quality_list:

            try:
                cut_x = data[v]
                result = dd.get_woe_iv(data[v], data[self.target_v])
                propotion = cut_x.value_counts(normalize=True)
                propotion_dict = dict(zip(list(propotion.index), list(propotion.values)))
                variable_record.loc[v, 'iv'] = result['iv']
                variable_record.loc[v, 'woe_dict'] = str(result['woe_dict'])
                variable_record.loc[v, 'min_woe'] = str(min(result['woe_dict'].values()))
                variable_record.loc[v, 'bad_rate_dict'] = str(result['bad_rate_dict'])
                min_bad_rate = min(result['bad_rate_dict'].values())
                variable_record.loc[v, 'min_bad_rate'] = min_bad_rate

                max_bad_rate = max(result['bad_rate_dict'].values())
                variable_record.loc[v, 'max_bad_rate'] = max_bad_rate

                min_bad_rate_index = list(result['bad_rate_dict'].keys())[
                    list(result['bad_rate_dict'].values()).index(min_bad_rate)]
                variable_record.loc[v, 'min_bad_rate_propotion'] = propotion_dict[min_bad_rate_index]
                max_bad_rate_index = list(result['bad_rate_dict'].keys())[
                    list(result['bad_rate_dict'].values()).index(max_bad_rate)]
                variable_record.loc[v, 'max_bad_rate_propotion'] = propotion_dict[max_bad_rate_index]

                psi = dd.compute_psi(cut_x, data[self.target_v])
                monotony = dd.compute_monotony(cut_x, data[self.target_v])
                variable_record.loc[v, 'psi'] = psi
                variable_record.loc[v, 'monotony'] = monotony
                variable_record.loc[v, 'propotion_dict'] = str(propotion_dict)
            except Exception as e:
                msg = traceback.format_exc()
                print(msg)

            # if result['iv'] > iv_threshold or v=='ApplyLoanStr_m1':
            #     dd.draw('quality', self.draw_dir, '{}_{}'.format(round(result['iv'],4),v), data[v], data[self.target_v])
        # 筛选不合规定的特征
        print(variable_record)
        variable_record.loc[(variable_record['psi'] > 5) | (variable_record['iv'] < 0.01), 'model_in'] = 0
        # 删除高相关性的特征
        variable_record.to_csv(self.filter_variable_file, index=True, encoding='gbk')

    def multiple_period_analysis(self):
        train_data = self.get_clean_data( data_type='train')
        test_data = self.get_clean_data( data_type='test')
        oot_data = self.get_clean_data( data_type='out')
        total_data = self.get_clean_data( data_type='total')

        variable_record = pd.read_csv(self.filter_variable_file, encoding='gbk')

        variable_record[['bins', 'woe_dict']] = variable_record[['bins', 'woe_dict']].astype(str)
        variable_record['bins'] = variable_record['bins'].map(lambda x: x.replace('inf', 'np.inf'))
        variable_record['woe_dict'] = variable_record['woe_dict'].map(lambda x: x.replace('inf', 'np.inf'))

        variable_record.set_index('variable', inplace=True)
        quantity_list = list(variable_record.loc[(variable_record['type'] == 0) \
                                                 & (variable_record['model_in'] == 1), :].index)
        quality_list = list(variable_record.loc[(variable_record['type'] == 1) \
                                                & (variable_record['model_in'] == 1), :].index)
        for data_type in ['训练', '测试', '样本外','全部']:
            # for data_type in ['训练样本外']:

            if data_type == '训练':
                s_data = train_data.copy()
            elif data_type == '测试':
                s_data = test_data.copy()
            elif data_type == '样本外':
                s_data = oot_data.copy()
            elif data_type == '全部':
                s_data = total_data.copy()
            else:
                s_data = train_data.copy()
            dd = data_deal()

            s_data[quantity_list] = s_data[quantity_list].astype(np.float64)

            s_data[quality_list] = s_data[quality_list].astype(str)
            for v in quantity_list:
                bins = eval(variable_record.loc[v, 'bins'])
                iv = variable_record.loc[v, 'iv']
                cut_x = pd.cut(s_data[v], bins=bins, labels=bins[1:])

                result = dd.get_woe_iv(cut_x, s_data[self.target_v])
                variable_record.loc[v, '{}_iv'.format(data_type)] = result['iv']
                variable_record.loc[v, '{}_bad_rate_dict'.format(data_type)] = str(result['bad_rate_dict'])
                variable_record.loc[v, '{}_length'.format(data_type)] = s_data.shape[0]
            for v in quality_list:
                iv = variable_record.loc[v, 'iv']

                result = dd.get_woe_iv(s_data[v], s_data[self.target_v])
                variable_record.loc[v, '{}_iv'.format(data_type)] = result['iv']
                variable_record.loc[v, '{}_bad_rate_dict'.format(data_type)] = str(result['bad_rate_dict'])
                variable_record.loc[v, '{}_length'.format(data_type)] = s_data.shape[0]
            variable_record['{}_bad_rate_dict'.format(data_type)] = variable_record[
                '{}_bad_rate_dict'.format(data_type)].astype(str)
            variable_record['{}_bad_rate_dict'.format(data_type)] = variable_record[
                '{}_bad_rate_dict'.format(data_type)].map(lambda x: x.replace('inf', 'np.inf'))

        # variable_record.to_csv('variable_record.csv')
        # 得到排序情况
        for v in quantity_list:
            try:
                train_bad_rate_dict = eval(variable_record.loc[v, '训练_bad_rate_dict'])
                test_bad_rate_dict = eval(variable_record.loc[v, '测试_bad_rate_dict'])
                oot_bad_rate_dict = eval(variable_record.loc[v, '样本外_bad_rate_dict'])
                total_bad_rate_dict = eval(variable_record.loc[v, '全部_bad_rate_dict'])
                print(pd.DataFrame([train_bad_rate_dict,test_bad_rate_dict,oot_bad_rate_dict,total_bad_rate_dict]))
                if len(train_bad_rate_dict) > 0:
                    #训练集values最大key
                    max_train_key = max(train_bad_rate_dict, key=train_bad_rate_dict.get)
                    min_train_key = min(train_bad_rate_dict, key=train_bad_rate_dict.get)
                    #训练集最大key对应测试集的values
                    max_test_values = test_bad_rate_dict[max_train_key]
                    min_test_values = test_bad_rate_dict[min_train_key]

                    max_train_values =  max(train_bad_rate_dict.values())
                    min_train_values = min(train_bad_rate_dict.values())

                    variable_record.loc[v, '最大箱体_差异度'] =abs(max_train_values-max_test_values)/max_train_values
                    variable_record.loc[v, '最小箱体_差异度'] = abs(min_train_values-min_test_values)/min_train_values


                    train_bad_rate = sorted(train_bad_rate_dict, key=train_bad_rate_dict.get)
                    test_bad_rate = sorted(test_bad_rate_dict, key=test_bad_rate_dict.get)
                    oot_bad_rate = sorted(oot_bad_rate_dict, key=oot_bad_rate_dict.get)
                    # if train_bad_rate_list.index(train_bad_rate_dict[select_bins]) == test_bad_rate_list.index(
                    #         test_bad_rate_dict[select_bins]) == oot_bad_rate_list.index(oot_bad_rate_dict[select_bins]):
                    if train_bad_rate == test_bad_rate:
                        variable_record.loc[v, '训练测试集一致性'] = 1
                    if train_bad_rate == test_bad_rate==oot_bad_rate:
                        variable_record.loc[v, '训练测试样本外一致性'] = 1
            except Exception as e:
                msg = traceback.format_exc()
                print(msg)
        variable_record.to_csv(self.filter_variable_update, index=True, encoding='gbk')

    def select_feature_in(self):
        select_feature_dir = '{}\\模型'.format(self.draw_dir)
        file_list = os.listdir(select_feature_dir)
        feature_list = []
        for file in file_list:
            file = file.replace('_测试.png', '')
            file = file.replace('_训练.png', '')
            file = file.replace('_训练样本外.png', '')
            reverse_file = file[::-1]

            reverse_file = reverse_file[reverse_file.index('_') + 1:]
            feature_list.append(reverse_file[::-1])

        if not os.path.isfile(self.filter_variable_file):
            self.variable_filter_basic()
        iv_record = pd.read_csv(self.filter_variable_file, encoding='gbk')
        iv_record['model_in'] = iv_record['variable'].map(lambda x: 1 if x in feature_list else 0)
        iv_record.to_csv(self.filter_variable_update, index=False, encoding='gbk')

    def corr_filter(self, n=50):
        data = self.get_clean_data( data_type='train')
        iv_record = pd.read_csv(self.filter_variable_update, encoding='gbk')
        iv_record[['bins', 'woe_dict']] = iv_record[['bins', 'woe_dict']].astype(str)
        s_iv_record = iv_record.loc[(iv_record['model_in'] == 1) & (
            iv_record['type'].isin(['quantity', 'quality', 'constructed', 'tree', 'rule'])), :].copy()
        s_iv_record['iv'] = s_iv_record['iv'].astype(np.float64)
        s_iv_record.sort_values(by='iv', ascending=False, inplace=True)
        v_list = list(s_iv_record['variable'])
        # 获取抽取出来的特征
        # quantity_v = s_iv_record.loc[(s_iv_record['type'].isin(['quantity'])), :].copy()
        quantity_v = s_iv_record.loc[(s_iv_record['type'].isin(['quantity', 'constructed'])), :].copy()
        quality_v = s_iv_record.loc[(s_iv_record['type'].isin(['quality', 'tree', 'rule'])), :].copy()
        s_data = data.copy()
        '''
        for v_id in list(quantity_v.index):
            v = quantity_v.loc[v_id, 'variable']
            bins = eval(quantity_v.loc[v_id, 'bins'])
            woe_dict = eval(quantity_v.loc[v_id, 'woe_dict'])
            s_data[v] = s_data[v].astype(np.float64)
            s_data[v] = pd.cut(s_data[v], bins=bins, labels=bins[1:])
            s_data[v] = s_data[v].map(woe_dict)
        '''
        for v_id in list(quality_v.index):
            v = quality_v.loc[v_id, 'variable']
            woe_dict = eval(quality_v.loc[v_id, 'woe_dict'])
            s_data[v] = s_data[v].map(woe_dict)

        s_data[v_list] = s_data[v_list].astype(np.float64)
        corr_data = s_data[v_list].corr()
        high_corr_v = []
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                corr_value = corr_data.loc[v_list[i], v_list[j]]
                if abs(corr_value) > 0.9:
                    high_corr_v.append(v_list[j])
        # 删除高相关性的特征
        iv_record.sort_values(by='iv', ascending=False, inplace=True)
        iv_record.set_index('variable', inplace=True)
        iv_record.loc[high_corr_v, 'model_in'] = 2

        for i in high_corr_v:
            if i in v_list:
                v_list.remove(i)

        v_list = list(
            iv_record.loc[(iv_record['type'].isin(['quantity', 'constructed'])) & (iv_record['model_in'] == 1),
            :].index)
        data[v_list] = data[v_list].astype(np.float64)
        corr_data = data[v_list].corr()

        high_corr_v = []
        for i in range(len(v_list)):
            for j in range(i + 1, len(v_list)):
                corr_value = corr_data.loc[v_list[i], v_list[j]]
                if abs(corr_value) > 0.7:
                    high_corr_v.append(v_list[j])
        # 删除高相关性的特征
        iv_record.sort_values(by='iv', ascending=False, inplace=True)
        iv_record.loc[high_corr_v, 'model_in'] = 2

        for i in high_corr_v:
            if i in v_list:
                v_list.remove(i)

        iv_record.reset_index(drop=False, inplace=True)
        # v_list=list(iv_record.loc[(iv_record['model_in']==1),'variable'])
        # if len(v_list)>n:
        #     v_list=v_list[:n]
        # iv_record.loc[(~iv_record['variable'].isin(v_list))&(iv_record['model_in']==1),'model_in']=3

        iv_record.to_csv(self.filter_variable_update, index=False, encoding='gb18030')

    def feature_visualization(self):
        train_data = self.get_clean_data( data_type='train')
        test_data = self.get_clean_data( data_type='test')
        oot_data = self.get_clean_data( data_type='out')
        total_data = self.get_clean_data( data_type='total')
        variable_record = pd.read_csv(self.filter_variable_update, encoding='gbk')
        # variable_record = pd.read_csv(self.filter_variable_file, encoding='gbk')
        #
        variable_record[['bins', 'woe_dict']] = variable_record[['bins', 'woe_dict']].astype(str)
        variable_record['bins'] = variable_record['bins'].map(
            lambda x: x.replace('inf', 'np.inf') if 'inf' in x and 'np' not in x else x)
        variable_record['woe_dict'] = variable_record['woe_dict'].map(
            lambda x: x.replace('inf', 'np.inf') if 'inf' in x and 'np' not in x else x)

        variable_record.set_index('variable', inplace=True)
        for data_type in ['训练', '测试', '样本外','全部']:

            if data_type == '训练':
                s_data = train_data.copy()
            elif data_type == '测试':
                s_data = test_data.copy()
            elif data_type == '样本外':
                s_data = oot_data.copy()
            else:
                s_data = total_data.copy()
            dd = data_deal()

            quantity_list = list(variable_record.loc[(variable_record['type'] == 0) \
                                                     & (variable_record['model_in'] == 1), :].index)
            s_data[quantity_list] = s_data[quantity_list].astype(np.float64)

            quality_list = list(variable_record.loc[(variable_record['type'] == 1) \
                                                    & (variable_record['model_in'] == 1), :].index)
            s_data[quality_list] = s_data[quality_list].astype(str)
            for v in quantity_list:
                bins = eval(variable_record.loc[v, 'bins'])
                cut_x = pd.cut(s_data[v], bins=bins, labels=bins[1:])
                result = dd.get_woe_iv(cut_x, s_data[self.target_v])
                if '/' in v:
                    v = v.replace('/', '')
                dd.draw('quantity', self.draw_dir, '{}_{}_{}'.format(v, round(result['iv'], 4), data_type), cut_x,
                        s_data[self.target_v])
            for v in quality_list:
                result = dd.get_woe_iv(s_data[v], s_data[self.target_v])
                variable_record.loc[v, '{}_iv'.format(data_type)] = result['iv']
                dd.draw('quality', self.draw_dir, '{}_{}_{}'.format(v, round(result['iv'], 4), data_type), s_data[v],
                        s_data[self.target_v])

    def modify_target_vlist(self):
        try:
            iv_record = pd.read_csv(self.filter_variable_file, encoding='gbk')
        except:
            iv_record = pd.read_csv(self.filter_variable_file, encoding='utf-8')
        else:
            iv_record = pd.read_csv(self.filter_variable_file, encoding='GB18030')
        target_v_list = ['feature1415', 'feature1539', 'feature1695', 'feature883', 'feature2318', 'feature885',
                         'feature2056', 'feature774', 'feature113', 'feature122', 'feature2332', 'feature169',
                         'feature1508', 'feature1037', 'feature1036', 'feature1821', 'feature2300', 'feature745',
                         'feature1702', 'feature1789']
        # target_v_list=list(iv_record.loc[iv_record['model_in'].isin([1]),'variable'])
        # target_v_list=random.sample(target_v_list,50)de
        if len(target_v_list) == 0:
            pass
        else:
            iv_record.loc[(~iv_record['variable'].isin(target_v_list)) & (iv_record['model_in'] == 1), 'model_in'] = 6
            iv_record.loc[iv_record['variable'].isin(target_v_list), 'model_in'] = 1
        iv_record.to_csv(self.filter_variable_update, index=False, encoding='gbk')

    def logistics_regression(self):
        train_data = self.get_clean_data( data_type='train')
        test_data = self.get_clean_data( data_type='test')
        oot_data = self.get_clean_data( data_type='out')
        total_data = self.get_clean_data( data_type='total')
        
        iv_record = pd.read_csv(self.filter_variable_update, encoding='gbk')
        iv_record[['bins', 'woe_dict']] = iv_record[['bins', 'woe_dict']].astype(str)
        iv_record['bins'] = iv_record['bins'].map(
            lambda x: x.replace('inf', 'np.inf') if 'inf' in x and 'np' not in x else x)
        iv_record['woe_dict'] = iv_record['woe_dict'].map(
            lambda x: x.replace('inf', 'np.inf') if 'inf' in x and 'np' not in x else x)
        s_iv_record = iv_record.loc[iv_record['model_in'] == 1, :].copy()
        quantity_v = s_iv_record.loc[(s_iv_record['type'] == 0), :].copy()
        quality_v = s_iv_record.loc[(s_iv_record['type'] == 1), :].copy()
        # 格式化
        v_list = list(quantity_v['variable']) + list(quality_v['variable'])

        for v_id in list(quantity_v.index):
            v = quantity_v.loc[v_id, 'variable']

            bins = eval(quantity_v.loc[v_id, 'bins'])
            woe_dict = eval(quantity_v.loc[v_id, 'woe_dict'])
            # 训练
            train_data[v] = train_data[v].astype(np.float64)
            train_data[v] = pd.cut(train_data[v], bins=bins, labels=bins[1:])
            train_data[v] = train_data[v].map(woe_dict)
            # 测试
            test_data[v] = test_data[v].astype(np.float64)
            test_data[v] = pd.cut(test_data[v], bins=bins, labels=bins[1:])
            test_data[v] = test_data[v].map(woe_dict)
            # 样本外数据
            oot_data[v] = oot_data[v].astype(np.float64)
            oot_data[v] = pd.cut(oot_data[v], bins=bins, labels=bins[1:])
            oot_data[v] = oot_data[v].map(woe_dict)
            # 全部
            total_data[v] = total_data[v].astype(np.float64)
            total_data[v] = pd.cut(total_data[v], bins=bins, labels=bins[1:])
            total_data[v] = total_data[v].map(woe_dict)

        for v_id in list(quality_v.index):
            v = quality_v.loc[v_id, 'variable']
            woe_dict = eval(quality_v.loc[v_id, 'woe_dict'])
            train_data[v] = train_data[v].map(woe_dict)
            test_data[v] = test_data[v].map(woe_dict)
            oot_data[v] = oot_data[v].map(woe_dict)
            total_data[v] = total_data[v].map(woe_dict)

        tr = train()
        train_data[v_list] = train_data[v_list].astype(np.float64)
        train_data.fillna(value=0, inplace=True)
        test_data[v_list] = test_data[v_list].astype(np.float64)
        test_data.fillna(value=0, inplace=True)
        oot_data[v_list] = oot_data[v_list].astype(np.float64)
        oot_data.fillna(value=0, inplace=True)
        lr_result = tr.final_logical_train(
            train_x=train_data[v_list],
            train_y=train_data[self.target_v],
            test_x=test_data[v_list],
            test_y=test_data[self.target_v],
            oot_x=oot_data[v_list],
            oot_y=oot_data[self.target_v])
        final_train_result = lr_result[0]

        final_train_result.update({'target_v': [self.target_v], 'v_num': [len(v_list)], 'v_list': [str(v_list)]})
        #计算各数据集的情况
        for i in ['train','test','oot','total']:
            s_data=eval('{}_data'.format(i))
            s_data['repayment_date']=pd.to_datetime(s_data['repayment_date'])
            start_date=s_data['repayment_date'].min()
            end_date=s_data['repayment_date'].max()
            length=s_data.shape[0]
            final_train_result.update({'{}_start_date'.format(i): [start_date], '{}_end_date'.format(i): [end_date], '{}_length'.format(i): [length]})

        iv_record.set_index('variable', inplace=True)
        final_param_df = []
        for v in v_list:
            if iv_record.loc[v, 'type'] == 0:
                type = 0
                bins = str(iv_record.loc[v, 'bins'])
            else:
                type = 1
                bins = str(list(eval(iv_record.loc[v, 'woe_dict']).keys()))
            if 'inf' in bins and 'np' not in bins:
                bins = bins.replace('inf', 'np.inf')
            labels = str(list(eval(iv_record.loc[v, 'woe_dict']).values()))
            slope = lr_result[1][v]

            data_source = iv_record.loc[v, 'data_source']

            fillna = iv_record.loc[v, 'fillna']
            raw_name = iv_record.loc[v, 'raw_name']

            s_dict = {
                'data_source': data_source,
                'variable': v,
                'raw_name': raw_name,
                'type': type,
                'bins': bins,
                'labels': str(labels),
                'slope': slope,
                'fillna': fillna,
                'status': 1,
            }
            final_param_df.append(s_dict)

        # 模型参数保存为csv
        s_dict = {
            'variable': 'intercept',
            'slope': lr_result[1]['intercept'],
            'status': 1,
        }
        final_param_df.append(s_dict)
        param_data = pd.DataFrame(final_param_df)
        param_data.to_csv(self.final_param_file, index=False)
        # 保存训练数据文件
        if os.path.isfile(self.ml_record_file):
            ml_record = pd.read_csv(self.ml_record_file, encoding='gbk')
        else:
            ml_record = pd.DataFrame()
        if ml_record.shape[0] == 0:
            ml_record = pd.DataFrame(final_train_result).copy()
        else:
            ml_record = pd.concat([ml_record, pd.DataFrame(final_train_result)])

        ml_record.to_csv(self.ml_record_file, index=False, encoding='gbk')
        # 迭代，剔除斜率为负的共线性特征
        n_param_data = param_data.loc[(param_data['slope'] < 0) & (param_data['variable'] != 'intercept'), :].copy()

        if n_param_data.shape[0] > 0:
            print('迭代，剔除斜率为负的共线性特征')
            iv_record = pd.read_csv(self.filter_variable_update, encoding='gbk')
            iv_record.loc[iv_record['variable'].isin(list(n_param_data['variable'])), 'model_in'] = 5
            iv_record.to_csv(self.filter_variable_update, index=False, encoding='gbk')
            self.logistics_regression()

    def get_score(self):
        '''
        计算样本外的贷后表现
        :return: 
        '''
        
        p0 = 800
        pdo = 50
        b = pdo / math.log(2)
        param_data = pd.read_csv(self.final_param_file)
        s_data = self.get_clean_data(data_type='out')

        # s_data['repayment_date']=pd.to_datetime(s_data['repayment_date'])
        # s_data=s_data.loc[s_data['repayment_date']>=datetime.datetime.strptime('2023-05-20', "%Y-%m-%d")]
        # iv_record=pd.read_csv(self.filter_variable_update,encoding='gbk')
        param_data[['bins', 'labels']] = param_data[['bins', 'labels']].astype(str)
        # iv_record['bins']=iv_record['bins'].map(lambda x:x.replace('inf','np.inf'))
        # iv_record['woe_dict']=iv_record['woe_dict'].map(lambda x:x.replace('inf','np.inf'))
        param_data.set_index('variable', inplace=True)
        param_data['type'].fillna(value=0, inplace=True)
        param_data['type'] = param_data['type'].astype(np.float64)

        v_list = list(param_data.index)
        v_list.remove('intercept')

        s_data = s_data.loc[:, ['quota_no', 'is_bad'] + v_list].copy()
        for v in v_list:
            if param_data.loc[v, 'type'] == 0:
                bins = eval(param_data.loc[v, 'bins'])
                woe_dict = eval(param_data.loc[v, 'labels'])
                s_data[v] = s_data[v].astype(np.float64)
                s_data[v] = pd.cut(s_data[v], bins=bins, labels=woe_dict, ordered=False)

            elif param_data.loc[v, 'type'] == 1:
                bins = eval(param_data.loc[v, 'bins'])
                woe_dict = eval(param_data.loc[v, 'labels'])
                s_data[v] = s_data[v].map(dict(zip(bins, woe_dict)))

        s_data[v_list] = s_data[v_list].astype(np.float64)
        for key_value in v_list:

            if key_value == 'intercept':
                pass
            else:
                slope = param_data.loc[key_value, 'slope']

                s_data[key_value] = s_data[key_value] * slope
                s_data[key_value] = -b * s_data[key_value]
                # s_data[key_value].fillna(value=0, inplace=True)
        s_data['initial_score'] = [p0 - b * param_data.loc['intercept', 'slope']] * s_data.shape[0]
        s_data['score_sum'] = s_data[v_list + ['initial_score']].sum(axis='columns')

        cut_bins = pd.qcut(s_data['score_sum'], 10, retbins=True, duplicates='drop')[1]

        s_data['label_name'] = pd.cut(s_data['score_sum'], cut_bins, labels=cut_bins[1:])
        overdue = s_data.pivot_table(index='label_name', columns='is_bad', values='quota_no', aggfunc='count')
        overdue.rename(columns={1: '逾期', 0: '正常'}, inplace=True)

        overdue['总计'] = overdue['逾期'] + overdue['正常']
        overdue['逾期率'] = overdue['逾期'] / overdue['总计']
        # 累计逾期
        overdue['累计逾期'] = overdue['逾期'].cumsum()
        overdue['累计逾期率'] = overdue['累计逾期'] / overdue['逾期'].sum()
        overdue['累计正常'] = overdue['正常'].cumsum()
        overdue['累计正常率'] = overdue['累计正常'] / overdue['正常'].sum()
        overdue['ks'] = overdue['累计逾期率'] - overdue['累计正常率']
        overdue.loc['合计', :] = overdue.sum()

        # 计算ks
        overdue.index.name = '分数段'
        print(overdue)
        overdue.to_csv(self.lift_file, index=True)
        s_data['quota_no'] = s_data['quota_no'].astype(str)
        s_data.to_excel(self.score_file, index=True)
        param_data.reset_index(drop=False, inplace=True)

    def get_model_param(self,name):
        #生成模型配置文件夹
        param_dir = '{}\\param'.format(self.data_dir)
        if not os.path.isdir(param_dir):
            os.makedirs(param_dir)

        param_data = pd.read_csv(self.final_param_file)
        mask = param_data['data_source'].notna()
        param_data.loc[mask, 'variable'] = param_data.loc[mask, 'raw_name']

        today=str(datetime.today().date())
        today = today.replace('-', '')

        param_data['model_name']=[f'{name}_{today}'] * param_data.shape[0]
        # 将最后一列移动到第一列
        new_param_data = pd.concat([param_data['model_name'], param_data.drop('model_name', axis=1)], axis=1)
        new_param_data= new_param_data.drop(['raw_name'],axis=1)
        #模型配置表
        new_param_data.to_csv(f'{param_dir}\\6_3_model_param.csv', index=False)

        #生成表创建文件
        feature_name=['user_quota_details_id','trade_no']
        variable_list = new_param_data['variable'].tolist()
        variable_list.pop()
        feature_name.extend(variable_list)
        feature_name.extend(['initial_score','score_sum','update_time'])

        py_feature_config=pd.DataFrame()
        py_feature_config['feature_name']=feature_name
        py_feature_config['feature_comment'] = [None]*py_feature_config.shape[0]
        py_feature_config['data_format'] = ['float'] * py_feature_config.shape[0]
        py_feature_config['data_length'] = [0] * py_feature_config.shape[0]
        py_feature_config['data_table'] = [f'py_model_{name}_{today}'] * py_feature_config.shape[0]
        py_feature_config['feature_comment'] = [None] * py_feature_config.shape[0]
        py_feature_config['is_key'] = [0] * py_feature_config.shape[0]
        py_feature_config['raw_data_name'] = py_feature_config['feature_name']
        py_feature_config['extract_directly'] = [1] * py_feature_config.shape[0]
        py_feature_config['data_source'] = ['user_quota_details'] * py_feature_config.shape[0]
        py_feature_config['status'] = [1] * py_feature_config.shape[0]
        gmt_create=datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        py_feature_config['gmt_create'] = [gmt_create] * py_feature_config.shape[0]

        py_feature_config.loc[py_feature_config['feature_name'] == 'trade_no', 'data_format'] = 'varchar'
        py_feature_config.loc[py_feature_config['feature_name'] == 'user_quota_details_id', 'data_format'] = 'int'
        py_feature_config.loc[py_feature_config['feature_name'] == 'update_time', 'data_format'] = 'datetime'
        py_feature_config.loc[py_feature_config['data_format'] == 'varchar', 'data_length'] = 255
        py_feature_config.loc[py_feature_config['feature_name'] == 'trade_no', 'is_key'] = 1

        py_feature_config.to_excel(f'{param_dir}\\py_feature_config.xlsx', index=False)
        #生成参数文件
        record_file = pd.read_csv(self.ml_record_file)
        record_file=record_file.iloc[-1]
        py_model_param = pd.DataFrame([{
        'model_name':f'{name}_{today}',
        'train_start_date':record_file['train_start_date'],
        'train_end_date':record_file['train_end_date'],
        'train_length':record_file['train_length'],
        'train_auc':record_file['train_auc'],
        'train_ks':record_file['train_ks'],
        'test_start_date':record_file['test_start_date'],
        'test_end_date':record_file['test_end_date'],
        'test_length':record_file['test_length'],
        'test_auc':record_file['test_auc'],
        'test_ks':record_file['test_ks'],
        'oot_start_date':record_file['oot_start_date'],
        'oot_end_date':record_file['oot_end_date'],
        'oot_length':record_file['oot_length'],
        'oot_auc':record_file['oot_auc'],
        'oot_ks':record_file['oot_ks'],
        'prod_start_date':record_file['train_start_date'],
        'prod_end_date':record_file['train_end_date'],
        'prod_length':record_file['train_length'],
        'online_date':datetime.today().date(),
        'offline_date':None,
        'offline_comment':None
        }])
        py_model_param.to_excel(f'{param_dir}\\py_model_param.xlsx', index=False)


    def model_out_score(self, start, end,bins_num = 15):
        # 获取模型外数据
        pd.set_option('display.width', None)
        date_today = datetime.now().strftime('%Y-%m-%d')
        # 获取特征值
        # select_sql='select * from py_feature_monitory where operator_create="Dave"  and is_deleted!=1 and importance = 1'
        trade_sql = 'select trade_no,order_status,repayment_date from py_data_trade where repayment_date>="{}" and repayment_date<="{}" and is_new_user=1 and period_length=7'.format(
            start, end)
        param_data = pd.read_csv(self.final_param_file)

        feature_data = param_data[['data_source', 'raw_name']]
        feature_data = feature_data.iloc[:-1]
        data = self.db_connector.get_data(trade_sql)

        trade_no = data['trade_no']
        trade_no = trade_no.map(lambda x: '"{}"'.format(x))

        try:
            # 查找对应的数据
            trade_sheet_list = list(feature_data['data_source'].unique())
            keys = 'trade_no'
            for sheet in trade_sheet_list:
                feature_list = list(feature_data.loc[feature_data['data_source'] == sheet, 'raw_name'])
                feature_list.append(keys)
                select_sql = 'select {} from {} where trade_no in ({})'.format(','.join(feature_list), sheet,
                                                                               ','.join(trade_no))
                temp_data = self.db_connector.get_data(select_sql)

                data = data.merge(temp_data, on=keys, how='left')

        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
        # 删除空值占比大于30%的行
        data.replace({'None': np.nan, None: np.nan}, inplace=True)

        data['null_ratio'] = data.isnull().sum(axis='columns') / data.shape[1]

        data = data.loc[data['null_ratio'] < 0.3, :].copy()
        data.drop(columns=['null_ratio'], inplace=True)
        data = data.fillna(-1)

        data.to_excel('{}/model_out.xlsx'.format(self.data_dir), index=False)

        # 清洗数据
        p0 = 800
        pdo = 50
        b = pdo / math.log(2)

        time_now = datetime.now().strftime('%Y%m%d')
        current_dir = os.getcwd()

        merged_data_dir = '{}/private_file/{}'.format(current_dir, self.type)
        if not os.path.isdir(merged_data_dir):
            os.makedirs(merged_data_dir)
        data_file = '{}/model_out.xlsx'.format(merged_data_dir)
        data = pd.read_excel(data_file)
        data.rename(columns={'order_status': 'is_bad', 'trade_no': 'quota_no'}, inplace=True)
        data['is_bad'] = data['is_bad'].map({6: 1, 7: 0})
        data['is_bad'].fillna(value=1, inplace=True)

        # 清除其他类型的数据
        data.replace({'None': np.nan}, inplace=True)

        data['repayment_date'] = pd.to_datetime(data['repayment_date'])
        data.sort_values(by='repayment_date', ascending=True, inplace=True)

        data.drop_duplicates(subset='quota_no', keep='last', inplace=True)
        data.reset_index(drop=True, inplace=True)

        s_data = data.copy()

        # 计算贷后表现

        # s_data['repayment_date']=pd.to_datetime(s_data['repayment_date'])
        # s_data=s_data.loc[s_data['repayment_date']>=datetime.datetime.strptime('2023-05-20', "%Y-%m-%d")]
        # iv_record=pd.read_csv(self.filter_variable_update,encoding='gbk')
        param_data[['bins', 'labels']] = param_data[['bins', 'labels']].astype(str)
        #iv_record['bins']=iv_record['bins'].map(lambda x:x.replace('inf','np.inf'))
        #iv_record['woe_dict']=iv_record['woe_dict'].map(lambda x:x.replace('inf','np.inf'))
        param_data['raw_name'].fillna('intercept', inplace=True)
        param_data.set_index('raw_name', inplace=True)
        param_data['type'].fillna(value=0, inplace=True)

        param_data['type'] = param_data['type'].astype(np.float64)
        v_list = list(param_data.index)
        v_list.pop()
        s_data = s_data.loc[:, ['quota_no', 'is_bad'] + v_list].copy()
        for v in v_list:
            if param_data.loc[v, 'type'] == 0:
                bins = eval(param_data.loc[v, 'bins'])
                woe_dict = eval(param_data.loc[v, 'labels'])
                s_data[v] = s_data[v].astype(np.float64)
                s_data[v] = pd.cut(s_data[v], bins=bins, labels=woe_dict, ordered=False)

            elif param_data.loc[v, 'type'] == 1:
                bins = eval(param_data.loc[v, 'bins'])
                woe_dict = eval(param_data.loc[v, 'labels'])
                s_data[v] = s_data[v].map(dict(zip(bins, woe_dict)))

        s_data[v_list] = s_data[v_list].astype(np.float64)
        for key_value in v_list:
            if key_value == 'intercept':
                pass
            else:
                slope = param_data.loc[key_value, 'slope']
                s_data[key_value] = s_data[key_value] * slope
                s_data[key_value] = -b * s_data[key_value]

                # s_data[key_value].fillna(value=0, inplace=True)

        s_data['initial_score'] = [p0 - b * param_data.loc['intercept', 'slope']] * s_data.shape[0]
        s_data['score_sum'] = s_data[v_list + ['initial_score']].sum(axis='columns')

        cut_bins = pd.qcut(s_data['score_sum'], 10, retbins=True, duplicates='drop')[1]

        s_data['label_name'] = pd.cut(s_data['score_sum'], cut_bins, labels=cut_bins[1:])
        overdue = s_data.pivot_table(index='label_name', columns='is_bad', values='quota_no', aggfunc='count')
        overdue.rename(columns={1: '逾期', 0: '正常'}, inplace=True)

        overdue['总计'] = overdue['逾期'] + overdue['正常']
        overdue['逾期率'] = overdue['逾期'] / overdue['总计']
        # 累计逾期
        overdue['累计逾期'] = overdue['逾期'].cumsum()
        overdue['累计逾期率'] = overdue['累计逾期'] / overdue['逾期'].sum()
        overdue['累计正常'] = overdue['正常'].cumsum()
        overdue['累计正常率'] = overdue['累计正常'] / overdue['正常'].sum()
        overdue['ks'] = overdue['累计逾期率'] - overdue['累计正常率']
        overdue.loc['合计', :] = overdue.sum()

        # 计算ks
        overdue.index.name = '分数段'
        print(overdue)
        # overdue.to_csv(self.lift_file, index=True)
        s_data['quota_no'] = s_data['quota_no'].astype(str)
        s_data.to_excel('{}\\score_file_out.xlsx'.format(self.data_dir), index=True)
        param_data.reset_index(drop=False, inplace=True)

        s_data['is_bad'] = s_data['is_bad'].map({1: 'overdue', 0: 'paid'})
        # 查看还款情况
        data = s_data.copy()


        data['score_num'] = data['score_sum'].astype(np.float64)
        bins = list(pd.qcut(data['score_sum'], bins_num, retbins=True, duplicates='drop')[1])
        bins = np.asarray(bins)
        bins = np.around(bins, 2)
        labels = []
        print(bins)
        for i in range(len(bins)):
            if i == 0:
                pass
            elif i == 1:
                labels.append('below {}'.format(bins[i]))
            elif i == len(bins) - 1:
                labels.append('above {}'.format(bins[i - 1]))
            else:
                labels.append('{}-{}'.format(bins[i - 1], bins[i]))

        data['labels'] = pd.cut(data['score_num'], bins=bins, labels=labels)

        result = data.pivot_table(index='labels', columns='is_bad', values='quota_no', aggfunc='count')
        result['type'] = ['score_num'] * result.shape[0]

        result['sum'] = result['overdue'] + result['paid']
        result['pai_ratio'] = result['paid'] / result['sum']
        print(result)

    def random_forest(self,threshold = 0.15):
        total_data = self.get_clean_data(data_type='total')
        # total_data = pd.read_excel('{}/clean_data_total.xlsx'.format(self.data_dir))
        para_data = pd.read_csv('{}/6_3_model_param.csv'.format(self.data_dir))
        total_data = total_data.fillna(-1)
        x = total_data.iloc[:, 1:-2]
        para_data = para_data.iloc[:-1]
        numeric_cols = list(para_data['variable'])
        x = x[numeric_cols].values
        y = total_data['is_bad'].values

        # n_estimators：森林中树的数量
        # n_jobs  整数 可选（默认=1） 适合和预测并行运行的作业数，如果为-1，则将作业数设置为核心数
        from sklearn.ensemble import RandomForestClassifier
        forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

        forest.fit(x, y)

        # feature_importances_  可以调取关于特征重要程度

        importances = forest.feature_importances_

        indices = np.argsort(importances)[::-1]

        for f in range(x.shape[1]):
            print("%2d) %-*s %f" % (f + 1, 30, numeric_cols[indices[f]], importances[indices[f]]))

        # 筛选变量（选择重要性比较高的变量）
        x_selected = x[:, importances > threshold]
        print(x_selected)

        from xgboost import XGBRegressor
        from xgboost import plot_importance
        xgb = XGBRegressor()
        xgb.fit(x, y)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))
        plot_importance(xgb)
        plt.show()

