import pandas as pd
import os
import numpy  as np
import math
import matplotlib
matplotlib.use('Agg')
from sklearn.tree import DecisionTreeClassifier
import traceback
#            klines[func_name]=klines[func_name].map(lambda x:1/(1+math.exp(-x)))
class data_deal():
    def drop_extreme_value(self,series):
        '''
        :param series:
        :return:
        '''
        series=pd.Series(series)
        if len(series[series==np.inf])==0:
            return series
        else:
            max_value=series[series<np.inf].max()
            series[series == np.inf]=max_value
        return series
    # 等频分箱的函数
    def same_freq_bins(self,x, y, chi_ppf, max_bin_num=6):  # 卡方分箱,返回最优箱体
        '''
        :param x: 需要分箱的变量的值
        :param y: 目标变量的值
        :param chi_ppf: 当相邻箱体>chi_ppf分位点的卡方值
        :param max_bin_num: 最大的箱体数量
        :return: 返回最优分箱的分箱列表
        '''
        try:
            from scipy.stats import chi2
            # 计算
            data = pd.DataFrame({'x': x, 'y': y})
            #值替换
            if 'good'  in data['y'].unique()\
                    and 'bad' in data['y'].unique():
                pass
            if 0  in data['y'].unique()\
                    or 1 in data['y'].unique():
                data['y'].replace({1:'bad',0:'good'},inplace=True)
            if '0'  in data['y'].unique()\
                    or '1' in data['y'].unique():
                data['y'].replace({'1':'bad','0':'good'},inplace=True)
            # 判断
            # 判断数值个数，少于50的单独分箱：

            #等频分箱，获取分位点
            data['x']=np.round(data['x'],6)
            bins_value=list(pd.qcut(data['x'],10,retbins=True,duplicates='drop')[1])
            print(bins_value)
            if -np.inf not in bins_value:
                bins_value.insert(0, -np.inf)
            if np.inf not in bins_value:
                bins_value[-1]=np.inf
            return list(bins_value)
        except:
            return [-np.inf,np.inf]
    # 卡方分箱的函数
    def chi_bins(self,x, y, chi_ppf, max_bin_num=6):  # 卡方分箱,返回最优箱体
        '''
        :param x: 需要分箱的变量的值
        :param y: 目标变量的值
        :param chi_ppf: 当相邻箱体>chi_ppf分位点的卡方值
        :param max_bin_num: 最大的箱体数量
        :return: 返回最优分箱的分箱列表
        '''
        try:
            from scipy.stats import chi2
            # 计算
            data = pd.DataFrame({'x': x, 'y': y})
            #值替换
            if 'good'  in data['y'].unique()\
                    and 'bad' in data['y'].unique():
                pass
            if 0  in data['y'].unique()\
                    or 1 in data['y'].unique():
                data['y'].replace({1:'bad',0:'good'},inplace=True)
            if '0'  in data['y'].unique()\
                    or '1' in data['y'].unique():
                data['y'].replace({'1':'bad','0':'good'},inplace=True)
            # 判断
            # 判断数值个数，少于50的单独分箱：
            unique_value = data['x'].unique()
            if len(unique_value) <= 10:
                data['bins'] = data['x']
            else:
                #等频分箱，获取分位点
                bins_value=list(pd.qcut(data['x'],20,retbins=True,duplicates='drop')[1])
                #按等频分箱分位点再分箱，以右侧分位点标记箱体
                data['bins'] = pd.cut(data['x'], bins=bins_value, labels=bins_value[1:])
            #透视
            pivot_data = pd.pivot_table(data, index='bins', values='x', columns='y', aggfunc='count')

            # 删除值为空的分箱
            pivot_data.fillna(value=0, inplace=True)
            bins_value = list(pivot_data.index.values)
            for i in range(len(bins_value)):
                if pivot_data.loc[bins_value[i], 'good'] + pivot_data.loc[bins_value[i], 'bad'] == 0:
                    pivot_data.drop(bins_value[i], axis=0, inplace=True)

            # 合并箱体
            while True:
                bins_value = list(pivot_data.index.values)
                agg_value = pd.DataFrame()
                if len(bins_value) > 2:
                    #遍历计算两两之间的卡方值
                    for i in range(1, len(bins_value)):
                        #计算合并后的好样本和坏样本数量
                        good_merge = pivot_data.loc[bins_value[i - 1], 'good'] + pivot_data.loc[bins_value[i], 'good']
                        bad_merge = pivot_data.loc[bins_value[i - 1], 'bad'] + pivot_data.loc[bins_value[i], 'bad']
                        #计算合并后的坏样本客户占比
                        b_r = bad_merge / (bad_merge + good_merge)
                        #前一个箱体的理论好样本数量
                        E11 = (pivot_data.loc[bins_value[i - 1], 'good'] + pivot_data.loc[bins_value[i - 1], 'bad']) * (1 - b_r)
                        # 前一个箱体的理论坏样本数量
                        E12 = (pivot_data.loc[bins_value[i - 1], 'good'] + pivot_data.loc[bins_value[i - 1], 'bad']) * b_r
                        # 后一个箱体的理论好样本数量
                        E21 = (pivot_data.loc[bins_value[i], 'good'] + pivot_data.loc[bins_value[i], 'bad']) * (1 - b_r)
                        # 后一个箱体的理论坏样本数量
                        E22 = (pivot_data.loc[bins_value[i], 'good'] + pivot_data.loc[bins_value[i], 'bad']) * b_r
                        #计算卡方值
                        c_value = (pivot_data.loc[bins_value[i - 1], 'good'] - E11) ** 2 / E11 + (
                                pivot_data.loc[bins_value[i - 1], 'bad'] - E12) ** 2 / E12 \
                                  + (pivot_data.loc[bins_value[i], 'good'] - E21) ** 2 / E21 + (
                                              pivot_data.loc[bins_value[i], 'bad'] - E22) ** 2 / E22
                        #将得到的卡方值存到一个新的表中
                        agg_value = pd.concat([agg_value, pd.DataFrame(
                            {'id': [i], 'good': [good_merge], 'bad': [bad_merge], 'c_value': [c_value],
                             'bin': [bins_value[i]]})], axis=0)
                    if agg_value.shape[0] > 0 and len(bins_value) == len(pivot_data.index.values):
                        chi_value = chi2.ppf(chi_ppf, 1)
                        agg_value.fillna(value=0, inplace=True)
                        agg_value.set_index('id', inplace=True)
                        # 获取最小卡方值的索引
                        min_bin_index = agg_value.idxmin(axis=0)['c_value']
                        #满足相邻卡方值大于指定置信水平下的卡方值，并且箱数小于指定箱数
                        if agg_value.loc[min_bin_index, 'c_value'] > chi_value and pivot_data.shape[0] < max_bin_num:
                            break
                        # 更新合并箱体的后一个箱子
                        min_bin = agg_value.loc[min_bin_index, 'bin']
                        pivot_data.loc[min_bin, 'good'] = agg_value.loc[min_bin_index, 'good']
                        pivot_data.loc[min_bin, 'bad'] = agg_value.loc[min_bin_index, 'bad']
                        # 删除合并后卡方值最小的bin的前一个bin
                        pivot_data.drop(bins_value[min_bin_index - 1], axis=0, inplace=True)
                else:
                    break
            bins_value = list(pivot_data.index.values)
            #将范围修改包含负无穷和正无穷
            if -np.inf not in bins_value:
                bins_value.insert(0, -np.inf)
            if np.inf not in bins_value:
                bins_value[-1]=np.inf
            return list(bins_value)
        except:
            return [-np.inf,np.inf]

    def tree_box(self,x, y):
        box_data = pd.DataFrame({'x': x, 'y': y})
        box_data.sort_values(by='x', inplace=True)
        bins_list = list(pd.qcut(box_data['x'], 50, retbins=True, duplicates='drop')[1])
        bins_list.insert(0, -np.inf)
        box_data['bins'] = pd.cut(box_data['x'], bins=bins_list, labels=bins_list[1:])
        box_data.reset_index(drop=True, inplace=True)
        box_data['count'] = [1] * box_data.shape[0]
        pivot_data = pd.pivot_table(box_data, index='bins', values=['y', 'count'], aggfunc='sum')

        pivot_data.reset_index(drop=False, inplace=True)
        sum_count = pivot_data['count'].sum()
        print(pivot_data['count'].sum(), pivot_data['y'].sum())
        # 计算
        while pivot_data.shape[0] > 6:
            pivot_data['p'] = pivot_data['y'] / pivot_data['count']
            pivot_data['entropy'] = -pivot_data['p'] * np.log(pivot_data['p'])
            if 'merged_entropy' in pivot_data.columns.values:
                pivot_data.drop(columns='merged_entropy', inplace=True)

            index_list = pivot_data.index.to_list()
            for index_id in index_list[1:]:
                pivot_data.loc[index_id, 'merged_entropy'] = pivot_data.loc[index_id - 1, 'entropy'] + pivot_data.loc[
                    index_id, 'entropy']

            id_min_p = pivot_data['count'].idxmin()
            if pivot_data.loc[id_min_p, 'count'] / sum_count < 0.05:
                new_count = pivot_data.loc[id_min_p - 1, 'count'] + pivot_data.loc[id_min_p, 'count']
                new_y = pivot_data.loc[id_min_p - 1, 'y'] + pivot_data.loc[id_min_p, 'y']
                pivot_data.drop(index=id_min_p - 1, inplace=True)
                pivot_data.loc[id_min_p, 'count'] = new_count
                pivot_data.loc[id_min_p, 'y'] = new_y
            else:
                id_max_entropy = pivot_data['merged_entropy'].idxmax()
                new_count = pivot_data.loc[id_max_entropy - 1, 'count'] + pivot_data.loc[id_max_entropy, 'count']
                new_y = pivot_data.loc[id_max_entropy - 1, 'y'] + pivot_data.loc[id_max_entropy, 'y']
                pivot_data.drop(index=id_max_entropy - 1, inplace=True)
                pivot_data.loc[id_max_entropy, 'count'] = new_count
                pivot_data.loc[id_max_entropy, 'y'] = new_y
            pivot_data.reset_index(drop=True, inplace=True)

        print(pivot_data)
        print(pivot_data['count'].sum(), pivot_data['y'].sum())
        # print(box_data)

    def optimal_binning_boundary(x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
        '''
            利用决策树获得最优分箱的边界值列表
        '''
        boundary = []  # 待return的分箱边界值列表

        x = x.fillna(nan).values  # 填充缺失值
        y = y.values

        clf = DecisionTreeClassifier(criterion='entropy',  # “信息熵”最小化准则划分
                                     max_leaf_nodes=6,  # 最大叶子节点数
                                     min_samples_leaf=0.05)  # 叶子节点样本数量最小占比

        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])

        boundary.sort()

        min_x = x.min()
        max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
        boundary = [min_x] + boundary + [max_x]

        return boundary

    # 获取woe和iv值
    def get_woe_iv(self, x, y):
        '''
        :param type: qualitive or quantative,定性还是定量
        :param x: 需要分箱的变量的值
        :param y: 目标变量的值
        :return: 返回字典{iv值，分箱边界，分箱对应的woe}
        '''
        try:
            import pandas as pd
            data = pd.DataFrame({'x': x, 'y': y})
            data['count_list']=[1]*data.shape[0]
            #值替换
            if 'good'  in data['y'].unique()\
                    and 'bad' in data['y'].unique():
                pass
            if 0  in data['y'].unique()\
                    or 1 in data['y'].unique():
                data['y'].replace({1:'bad',0:'good'},inplace=True)
            if '0'  in data['y'].unique()\
                    or '1' in data['y'].unique():
                data['y'].replace({'1':'bad','0':'good'},inplace=True)

            single_pivot = data.pivot_table(index='x', columns='y', values='count_list', aggfunc='count')
            single_pivot.fillna(value=0, inplace=True)

            if 'good' not in single_pivot.columns:
                single_pivot['good'] = 0
            if 'bad' not in single_pivot.columns:
                single_pivot['bad'] = 0

            total_bad = single_pivot['bad'].sum()
            total_good = single_pivot['good'].sum()
            single_pivot['woe'] = np.log((single_pivot['bad'] / total_bad + 0.01) / (single_pivot['good'] / total_good + 0.01))
            single_pivot['woe'] = single_pivot['woe'].round(2)
            single_pivot['iv'] = (single_pivot['bad'] / total_bad - single_pivot['good'] / total_good) * single_pivot['woe']
            single_pivot['bad_rate'] = single_pivot['bad'] / (single_pivot['bad'] + single_pivot['good'])
            single_pivot['bad_rate'] = single_pivot['bad_rate'].round(2)
            single_pivot['total'] = single_pivot['bad'] + single_pivot['good']
            single_pivot['percents'] = single_pivot['total'] / single_pivot['total'].sum() * 100
            single_pivot.sort_values(by='bad_rate', inplace=True)
            iv = single_pivot['iv'].sum()
            single_pivot = pd.DataFrame(single_pivot)

            single_pivot.sort_index(inplace=True)
            bin_value_list = list(single_pivot.index.values)
            dict_result = dict(zip(bin_value_list, list(single_pivot['percents'].values)))
            woe_dict=dict(zip(bin_value_list,list(single_pivot['woe'].values)))
            bad_rate_dict=dict(zip(bin_value_list,list(single_pivot['bad_rate'].values)))
            data['woe']=data['x'].map(woe_dict)
            return {'iv': iv, 'woe_dict': woe_dict, 'woe_list': list(data['woe']), 'bad_rate_dict': bad_rate_dict,
                    'bins': bin_value_list, 'percents': dict_result}
        except Exception as e:
            msg = traceback.format_exc()
            return {'iv': 0, 'woe_dict': {}, 'woe_list': [], 'bad_rate_dict': {}, 'bins': {}, 'percents': {}}

    #得到特征按时间段划分的排序一致性
    def get_consistency(self,x,y,n=5):
        data = pd.DataFrame({'x': x, 'y': y})
        label_bins=pd.cut(data.index,n,retbins=True)[1]
        data['bins_label']=pd.cut(data.index,bins=label_bins,labels=np.arange(1,len(label_bins)))
        result=data.pivot_table(index='x',columns='bins_label',values='y',aggfunc='sum')/data.pivot_table(index='x',columns='bins_label',values='y',aggfunc='count')
        for column in list(result.columns):
            result.sort_values(by=column,ascending=True,inplace=True)
            result[column]=np.arange(1,result.shape[0]+1)
        result['std']=result.std(axis='columns')
        return result['std'].mean()

    # 相关性筛选
    def filter_cor(self,main_data, dtype_value):
        '''
        :param main_data: 总数据集，包含dtype_value中的数据，定性变量woe
        :param dtype_value:变量筛选的df，包括了iv列和业务判定in_or_not是否入模
        :return:返回相关系数大于0.7的变量
        '''
        select_dtype_value = dtype_value.loc[(dtype_value['iv'] > 0.02) & (dtype_value['in_or_not'] == 1), :].copy()
        select_dtype_value.sort_values(by='iv', ascending=False, inplace=True)
        variable_list = list(select_dtype_value['variable'].values)
        cor_data = main_data[variable_list].corr().round(4)
        high_cor_list1 = []
        high_cor_list2 = []
        high_cor = []
        for i in range(len(variable_list) - 1):
            for j in range(i + 1, len(variable_list)):
                if cor_data.loc[variable_list[i], variable_list[j]] > 0.7:
                    if variable_list[j] not in high_cor_list2:
                        high_cor_list1.append(variable_list[i])
                        high_cor_list2.append(variable_list[j])
                        high_cor.append(cor_data.loc[variable_list[i], variable_list[j]])
        dtype_value.loc[dtype_value['variable'].isin(high_cor_list2), 'cor_del'] = 0
        high_cor_data = pd.DataFrame({'variable_1': high_cor_list1, 'variable_2': high_cor_list2, 'corr': high_cor})
        high_cor_data.sort_values(by='corr', ascending=False, inplace=True)
        return high_cor_data

    # 画图
    def draw(self,type, dir_name, v_name, x, y):
        '''

        :param type: qualitive或者quantative
        :param dir_name: 保存画图的路径
        :param v_name: 变量名称
        :param x: 类别数据、分箱后的箱体或者定量类别
        :param y: 目标变量
        :return:
        '''
        import matplotlib.pyplot as plt
        import matplotlib.pylab as pl

        current_dir = os.getcwd()
        if os.path.isdir(dir_name):
            pass
        else:
            os.makedirs(dir_name)
        data = pd.DataFrame({'x': x, 'y': y})
        data['count_list']=[1]*data.shape[0]
        if 'good'  in data['y'].unique()\
                and 'bad' in data['y'].unique():
            pass
        if 0  in data['y'].unique()\
                or 1 in data['y'].unique():
            data['y'].replace({1:'bad',0:'good'},inplace=True)
        if '0'  in data['y'].unique()\
                or '1' in data['y'].unique():
            data['y'].replace({'1':'bad','0':'good'},inplace=True)

        pivot_data = pd.pivot_table(data, index='x', columns='y', values='count_list',
                                    aggfunc='count')
        pivot_data.fillna(value=0, inplace=True)
        pivot_data['总数'] = pivot_data['bad'] + pivot_data['good']
        pivot_data['逾期率'] = pivot_data['bad'] / pivot_data['总数']
        if type == 'qualitive':
            pivot_data.sort_values(by='逾期率', inplace=True)
            # 画图
            # sns.set_style('whitegrid')
            pl.xticks(rotation=60)
            pl.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 修改为中文字体

            x = pivot_data.index
            y1 = pivot_data['总数']
            y2 = pivot_data['逾期率']
            plt.rcParams['figure.figsize'] = (8, 5)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.bar(x, y1, alpha=0.7, color='g',label='样本数')
            ax1.set_ylabel('样本数', fontsize='15')
            props = {'xlabel': '类别名称', 'ylabel': '样本数'}
            ax1.set(**props)
            ax2 = ax1.twinx()
            ax2.plot(x, y2, 'r', ms=10,label='逾期率')
            ax2.set_ylabel('逾期率', fontsize='15')
            plt.legend()
            plt.title(v_name, fontsize=20)
            plt.savefig(dir_name + '\\' + v_name + '.png', dpi=200, bbox_inches='tight')
            plt.close('all')
            print('保存成功',dir_name + '\\' + v_name + '.png')
        else:
            pivot_data.sort_index(inplace=True)
            pl.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 修改为中文字体
            pivot_data['位置'] = np.arange(1, pivot_data.shape[0] + 1)
            bins_name = pivot_data.index
            x = pivot_data['位置']
            y1 = pivot_data['总数']
            y2 = pivot_data['逾期率']
            plt.rcParams['figure.figsize'] = (8, 5)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.bar(x, y1, alpha=0.7, color='g',label='样本数')
            ax1.set_ylabel('样本数', fontsize='15')
            props = {'xlabel': '分箱', 'ylabel': '样本数'}
            ax1.set(**props)
            ax2 = ax1.twinx()
            ax2.plot(x, y2, 'r', ms=10,label='逾期率')
            ax2.set_ylabel('逾期率', fontsize='15')
            plt.title(v_name, fontsize=20)
            plt.xticks(x, bins_name)
            plt.legend()
            plt.savefig(dir_name + '\\' + v_name + '.png', dpi=200, bbox_inches='tight')
            plt.close('all')
            print('保存成功',dir_name + '\\' + v_name + '.png')

    def compute_psi(self,x, y):
        import math
        try:
            data = pd.DataFrame({'x': x, 'y': y})
            data.reset_index(drop=True, inplace=True)
            split_point = math.floor(data.shape[0] * 0.5)
            data1 = data.iloc[:split_point,:].copy()
            data2 = data.iloc[split_point:,:].copy()

            result1 = pd.concat([data1.groupby('x')['y'].sum(), data1.groupby('x')['y'].count()],axis='columns')
            result1.columns=['目标客户数1', '客户总数1']
            result2 = pd.concat([data2.groupby('x')['y'].sum(), data2.groupby('x')['y'].count()],axis='columns')
            result2.columns=['目标客户数2', '客户总数2']
            result = pd.concat([result1, result2], axis='columns', join='outer')
            result.fillna(value=0, inplace=True)
            result['占比1'] = result['目标客户数1'] / result['客户总数1']+ 0.01
            result['占比2'] = result['目标客户数2'] / result['客户总数2']+ 0.01

            result['占比差'] = result['占比1'] - result['占比2']
            result.fillna(value=0,inplace=True)
            result['占比对数'] = result.apply(lambda x:math.log(x['占比1'] / x['占比2']),axis='columns')
            result['single_psi'] = result['占比差'] * result['占比对数']
            return result['single_psi'].sum()
        except:
            return 0

    def compute_IC(self,x, y):
        data = pd.DataFrame({'x': x, 'y': y})
        IC=data['x'].corr(data['y'])
        return IC

    def cal_entropy(self,target_series):
        target_series=pd.Series(target_series)
        value_counts_result=target_series.value_counts()
        value_counts_result.reset_index(drop=True,inplace=True)
        enctopy_value=0
        value_counts_result=value_counts_result/value_counts_result.sum()
        for index_id in list(value_counts_result.index):
            p=value_counts_result[index_id]
            enctopy_value-=p*math.log(2,p)
        return enctopy_value

    def compute_information_gain(self,x, y):
        target_series=pd.Series(y)
        value_counts_result=target_series.value_counts()
        enctopy_value=0
        for index_id in list(value_counts_result.index.values):
            p=value_counts_result[index_id]/value_counts_result.sum()
            enctopy_value-=p*math.log(p,2)
        s_data=pd.DataFrame({'x':x,'y':y})
        value_count_x=s_data['x'].value_counts()
        conditional_encropy=0
        print(value_count_x)
        for i in list(value_count_x.index.values):
            print(i)
            p_x=value_count_x[i]/value_count_x.sum()
            x_data_y=s_data.loc[s_data['x']==i,'y']
            conditional_encropy+=p_x*self.cal_entropy(x_data_y)
        return enctopy_value-conditional_encropy

    def compute_information_gain_ratio(self, x, y):

        target_series = pd.Series(y)
        value_counts_result = target_series.value_counts()
        enctopy_value = 0
        for index_id in list(value_counts_result.index.values):
            p = value_counts_result[index_id] / value_counts_result.sum()
            enctopy_value -= p * math.log(p, 2)

        s_data = pd.DataFrame({'x': x, 'y': y})
        value_count_x = s_data['x'].value_counts()
        conditional_encropy = 0
        for i in list(value_count_x.index.values):
            print(i)
            p_x=value_count_x[i]/value_count_x.sum()
            x_data_y=s_data.loc[s_data['x']==i,'y']
            conditional_encropy += p_x * self.cal_entropy(x_data_y)

        feature_series = pd.Series(x)
        value_counts_result = feature_series.value_counts()
        x_enctopy_value = 0
        for index_id in list(value_counts_result.index.values):
            p = value_counts_result[index_id] / value_counts_result.sum()
            x_enctopy_value -= p * math.log(p, 2)

        return (enctopy_value - conditional_encropy)/x_enctopy_value

    def compute_gini(self,x,y):
        s_data = pd.DataFrame({'x': x, 'y': y})
        value_count_x = s_data['x'].value_counts()
        gini=0
        for i in list(value_count_x.index.values):
            p_x = value_count_x[i] / value_count_x.sum()
            x_data_y = s_data.loc[s_data['x'] == i, 'y']
            x_data_y_count=x_data_y.value_counts()
            x_data_y_count=x_data_y_count/x_data_y_count.sum()
            x_data_y_count=x_data_y_count**2
            x_gini=1-x_data_y_count.sum()
            gini+=p_x*x_gini
        return gini

    def compute_monotony(self,x, y):
        import math
        data = pd.DataFrame({'x': x, 'y': y})
        data.reset_index(drop=True, inplace=True)
        result = pd.concat([data.groupby('x')['y'].sum(), data.groupby('x')['y'].count()],axis='columns')
        result.columns=['目标客户数', '客户总数']
        result['逾期率']=result['目标客户数']/result['客户总数']
        bins_list=list(result.index.values)
        monotony=0
        turn_times=0

        for index_id in range(1,len(bins_list)):
            #初始化方向
            if monotony==0:
                if result.loc[bins_list[index_id],'逾期率']>result.loc[bins_list[index_id-1],'逾期率']:
                    monotony=1
                elif result.loc[bins_list[index_id],'逾期率']<result.loc[bins_list[index_id-1],'逾期率']:
                    monotony=-1
            else:
                if (monotony==1)\
                    and (result.loc[bins_list[index_id],'逾期率']<result.loc[bins_list[index_id-1],'逾期率']):
                    turn_times+=1
                    monotony=-1
                elif (monotony==-1)\
                    and (result.loc[bins_list[index_id],'逾期率']>result.loc[bins_list[index_id-1],'逾期率']):
                    turn_times+=1
                    monotony=1
        # print(result,'monotony',monotony,'turn_times',turn_times)
        return turn_times

    def tree(self):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn import metrics
        from sklearn.model_selection import GridSearchCV
        clf = DecisionTreeClassifier(criterion='gini', random_state=1234)
        # 梯度优化
        param_grid = {'max_depth': [3, 4, 5, 6], 'max_leaf_nodes': [4, 6, 8, 10, 12]}
        # cv 表示是创建一个类，还并没有开始训练模型
        cv = GridSearchCV(clf, param_grid=param_grid, scoring='f1')
