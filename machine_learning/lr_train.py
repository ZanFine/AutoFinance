import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import math
import joblib
import matplotlib.pyplot as plt
import shap

def compute_lift(y_prob_1, y, lift_bins=5, ):
    '''
    :param y_prob_1: 预测为1的概率
    :param y: y标签真实值
    :return:
    '''
    # 导入相关库
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')

    # 合并成DataFrame
    data = pd.DataFrame({'prob_1': y_prob_1, 'y': y})
    data.sort_values(by='prob_1', ascending=False, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data['Decile'] = pd.qcut(data.index, lift_bins, np.arange(1, lift_bins + 1))
    data['y'] = data['y'].astype(np.float64)

    # 透视计算lift
    lift_data = data.pivot_table(index='Decile', values='y', aggfunc='count')
    lift_data.rename(columns={'y': 'Obs'}, inplace=True)
    lift_data_bad = data.pivot_table(index='Decile', values='y', aggfunc='sum')
    lift_data_bad.rename(columns={'y': 'Bad'}, inplace=True)
    lift_data = pd.concat([lift_data, lift_data_bad], axis='columns')
    lift_data.reset_index(drop=False, inplace=True)
    lift_data['Bad_Captured_by_model'] = lift_data['Bad'] / lift_data['Obs']
    lift_data['Bad_Captured_by_randomly'] = [lift_data['Bad'].sum() / lift_data['Obs'].sum()] * lift_data.shape[0]
    lift_data['Cumulative_Bad_by_model'] = lift_data['Bad'].cumsum() / lift_data['Bad'].sum()
    # lift_data['Cumulative_Bad_by_randomly'] = (lift_data['Obs'] / lift_data['Obs'].sum()).cumsum()
    lift_data['Cumulative_Bad_by_randomly'] = lift_data['Bad_Captured_by_randomly'].cumsum() / lift_data['Bad_Captured_by_randomly'].sum()
    lift_data['Lift'] = lift_data['Cumulative_Bad_by_model'] / lift_data['Cumulative_Bad_by_randomly']

    #计算单调性
    motonous=1
    model_bins=lift_data['Bad_Captured_by_model']
    for i in range(1,len(list(model_bins))):
        if model_bins[i]>model_bins[i-1]:
            motonous=0
            break


    # 绘制累计lift图
    x = lift_data['Decile']
    y1 = lift_data['Lift']
    plt.rcParams['figure.figsize'] = (10, 5)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y1, alpha=0.7, color='g', label='样本数')
    ax1.set_xlabel('Decile', fontsize='12')
    ax1.set_ylabel('Lift', fontsize='12')
    plt.title('Cumulative Lift Chart', fontsize=15)
    plt.grid()
    # plt.savefig('Cumulative Lift Chart.png', dpi=200, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制对比柱状图
    y1 = list(lift_data['Bad_Captured_by_model'])
    y2 = list(lift_data['Bad_Captured_by_randomly'])
    x = list(lift_data['Decile'])
    totalWidth = 0.8  # 一组柱状体的宽度
    labelNums = 2  # 一组有两种类别
    barWidth = totalWidth / labelNums  # 单个柱体的宽度
    seriesNums = len(y1)
    plt.bar([x for x in range(seriesNums)], height=y1, label="Bad_Captured_by_model", width=barWidth)
    plt.bar([x + barWidth for x in range(seriesNums)], height=y2, label="Bad_Captured_by_randomly", width=barWidth)
    plt.xticks([x + barWidth / 2 * (labelNums - 1) for x in range(seriesNums)], x)
    plt.xlabel("Decile")
    plt.ylabel("Bad Ratio")
    plt.title("Lift Chart")
    plt.legend()
    # plt.savefig('Lift Chart.png', dpi=200, bbox_inches='tight')
    # if motonous==1:
    #     plt.show()
    plt.close()

    # 计算汇总
    lift_data.set_index('Decile', inplace=True)
    lift_data.loc['总计', 'Obs'] = lift_data['Obs'].sum()
    lift_data.loc['总计', 'Bad'] = lift_data['Bad'].sum()
    lift_data.reset_index(drop=False, inplace=True)
    # lift_data.to_csv('lift_data.csv', index=False)

    return motonous

class train:
    def __init__(self):
        self.current_dir = os.getcwd()

    def get_ks_value(self,y,y_prob):
        try:
            data = pd.DataFrame({'y_value': y, 'y_prob': y_prob})
            bins = list(pd.qcut(data['y_prob'],10,retbins=True,duplicates='drop')[1])
            data['bins'] = pd.cut(data['y_prob'], bins=bins, labels=range(1, len(bins)))
            data['y_value'].replace({1: 'one', 0: 'zero'}, inplace=True)
            pivot_table = pd.pivot_table(data, columns='y_value', values='y_prob', index='bins', aggfunc='count')
            if ['one', 'zero'].sort() == list(pivot_table.columns.values).sort():
                pivot_table['total'] = pivot_table['one'] + pivot_table['zero']
                pivot_table['bad_rate'] = pivot_table['one'] / pivot_table['one'].sum()
                pivot_table['good_rate'] = pivot_table['zero'] / pivot_table['zero'].sum()
                pivot_table['cum_bad_rate'] = pivot_table['bad_rate'].cumsum()
                pivot_table['cum_good_rate'] = pivot_table['good_rate'].cumsum()
                pivot_table['single_ks'] = pivot_table['cum_bad_rate'] - pivot_table['cum_good_rate']
            '''
            pivot_table.to_csv('ks.csv',index=True)
            pl.xticks(rotation=60)
            pl.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 修改为中文字体
    
            x = pivot_table.index
            y1 = pivot_table['cum_bad_rate']
            y2 = pivot_table['cum_good_rate']
            plt.rcParams['figure.figsize'] = (8, 5)
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(x, y1, alpha=0.7, color='g',label='cum_bad_rate')
    
            ax1.plot(x, y2,  alpha=0.7, color='r', ms=10,label='cum_good_rate')
            ax1.set_xlabel('bins', fontsize='15')
            ax1.set_ylabel('bad_rate', fontsize='15')
            plt.legend()
            plt.title('KS curve', fontsize=20)
            plt.show()
    
            plt.savefig(dir_name + '\\' + v_name + '.png', dpi=200, bbox_inches='tight')
            plt.close()
            '''
            ks_value = pivot_table['single_ks'].max()
        except:
            ks_value=0
        return ks_value

    def get_ks_prob(self, y, y_prob):
        data = pd.DataFrame({'y_value': y, 'y_prob': y_prob})
        bins = list(pd.qcut(data['y_prob'], 100, retbins=True, duplicates='drop')[1])
        data['bins'] = pd.cut(data['y_prob'], bins=bins, labels=range(1, len(bins)))
        data['y_value'].replace({1: 'one', 0: 'zero'}, inplace=True)
        pivot_table = pd.pivot_table(data, columns='y_value', values='y_prob', index='bins', aggfunc='count')
        if ['one', 'zero'].sort() == list(pivot_table.columns.values).sort():

            pivot_table['total'] = pivot_table['one'] + pivot_table['zero']
            pivot_table['bad_rate'] = pivot_table['one'] / pivot_table['one'].sum()
            pivot_table['good_rate'] = pivot_table['zero'] / pivot_table['zero'].sum()
            pivot_table['cum_bad_rate'] = pivot_table['bad_rate'].cumsum()
            pivot_table['cum_good_rate'] = pivot_table['good_rate'].cumsum()
            pivot_table['single_ks'] = pivot_table['cum_bad_rate'] - pivot_table['cum_good_rate']
        '''
        pivot_table.to_csv('ks.csv',index=True)
        pl.xticks(rotation=60)
        pl.mpl.rcParams['font.sans-serif'] = ['SimHei']  # 修改为中文字体

        x = pivot_table.index
        y1 = pivot_table['cum_bad_rate']
        y2 = pivot_table['cum_good_rate']
        plt.rcParams['figure.figsize'] = (8, 5)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, y1, alpha=0.7, color='g',label='cum_bad_rate')

        ax1.plot(x, y2,  alpha=0.7, color='r', ms=10,label='cum_good_rate')
        ax1.set_xlabel('bins', fontsize='15')
        ax1.set_ylabel('bad_rate', fontsize='15')
        plt.legend()
        plt.title('KS curve', fontsize=20)
        plt.show()

        plt.savefig(dir_name + '\\' + v_name + '.png', dpi=200, bbox_inches='tight')
        plt.close()
        '''
        ks_max_index = pivot_table['single_ks'].idxmax()
        return bins[ks_max_index]
    def get_evaluate_index(self,test_y,test_prob_1,train_y,train_prob_1,oot_y,oot_prob_1):
        import sklearn.metrics as metrics
        # 测试集
        test_fpr,test_tpr,test_threshold=metrics.roc_curve(test_y,test_prob_1)
        test_auc=metrics.auc(test_fpr,test_tpr)
        test_ks=max(test_tpr-test_fpr)
        test_prob_data=pd.DataFrame({'prob_1':test_prob_1,'test_y':test_y})
        cut_bins=pd.qcut(test_prob_data['prob_1'],10,retbins=True,duplicates='drop')[1]
        test_prob_data['period']=pd.cut(test_prob_data['prob_1'],cut_bins,labels=np.arange(len(cut_bins)-1))
        test_lift_rate=(test_prob_data.loc[test_prob_data['period']==len(cut_bins)-2,'test_y'].mean())/(test_prob_data['test_y'].mean())
        test_lower_rate=(test_prob_data.loc[test_prob_data['period']==0,'test_y'].mean())/(test_prob_data['test_y'].mean())

        # 训练集
        train_fpr, train_tpr, train_threshold = metrics.roc_curve(train_y, train_prob_1)
        train_auc = metrics.auc(train_fpr, train_tpr)
        train_ks = max(train_tpr - train_fpr)
        train_prob_data = pd.DataFrame({'prob_1': train_prob_1, 'train_y': train_y})
        cut_bins=pd.qcut(train_prob_data['prob_1'],10,retbins=True,duplicates='drop')[1]
        train_prob_data['period']=pd.cut(train_prob_data['prob_1'],cut_bins,labels=np.arange(len(cut_bins)-1))
        train_lift_rate = (train_prob_data.loc[train_prob_data['period'] == len(cut_bins)-2, 'train_y'].mean()) / (
            train_prob_data['train_y'].mean())
        train_lower_rate = (train_prob_data.loc[train_prob_data['period'] == 1, 'train_y'].mean()) / (
            train_prob_data['train_y'].mean())

        #样本外
        oot_fpr, oot_tpr, oot_threshold = metrics.roc_curve(oot_y, oot_prob_1)
        oot_auc = metrics.auc(oot_fpr, oot_tpr)
        oot_ks = max(oot_tpr - oot_fpr)
        oot_prob_data = pd.DataFrame({'prob_1': oot_prob_1, 'oot_y': oot_y})
        cut_bins=pd.qcut(oot_prob_data['prob_1'],10,retbins=True,duplicates='drop')[1]
        oot_prob_data['period']=pd.cut(oot_prob_data['prob_1'],cut_bins,labels=np.arange(len(cut_bins)-1))
        oot_lift_rate = (oot_prob_data.loc[oot_prob_data['period'] == len(cut_bins)-2, 'oot_y'].mean()) / (
            oot_prob_data['oot_y'].mean())
        oot_lower_rate = (oot_prob_data.loc[oot_prob_data['period'] == 1, 'oot_y'].mean()) / (
            oot_prob_data['oot_y'].mean())
        #综合结果
        result_dict = {
                        'test_ks':[test_ks],
                        'test_auc':[test_auc],
                        'test_lift_rate':[test_lift_rate],
                        'test_lower_rate':[test_lower_rate],

                        'train_ks':[train_ks],
                        'train_auc':[train_auc],
                        'train_lift_rate':[train_lift_rate],
                        'train_lower_rate':[train_lower_rate],

                        'oot_ks': [oot_ks],
                        'oot_auc': [oot_auc],
                        'oot_lift_rate': [oot_lift_rate],
                        'oot_lower_rate':[oot_lower_rate],
                        }
        return result_dict

    def final_logical_train(self,
                            train_x=pd.DataFrame(),
                            train_y=pd.DataFrame(),
                            test_x=pd.DataFrame(),
                            test_y=pd.DataFrame(),
                            oot_x=pd.DataFrame(),
                            oot_y=pd.DataFrame(),
                            param_type='train'
                            ):
        '''

        :param train_x: 训练集x
        :param train_y: 训练集y
        :param test_x: 测试集x
        :param test_y: 测试集y
        :param oot_x: 样本外x
        :param oot_y: 样本外y
        :param param_type: train：用训练集训练模型返回参数，train_test：用训练测试集训练模型返回参数
        :return:
        '''
        import sklearn.metrics as metrics
        from sklearn.model_selection import train_test_split
        train_test_x=pd.concat([train_x,test_x])
        train_test_y=pd.concat([train_y,test_y])
        #训练集
        lr_train = LogisticRegression(random_state=0)
        lr_train.fit(train_x, train_y)

        lr_train_test = LogisticRegression(random_state=0)
        lr_train_test.fit(train_test_x, train_test_y)

        #测试集
        test_prob=lr_train.predict_proba(test_x)
        test_prob_1=test_prob[:,1]

        #训练集集
        train_prob=lr_train.predict_proba(train_x)
        train_prob_1=train_prob[:,1]

        #样本外数据集
        oot_prob=lr_train_test.predict_proba(oot_x)
        oot_prob_1=oot_prob[:,1]

        #系数和截距
        if param_type=='train_test':
            coef = lr_train_test.coef_
            intercept = lr_train_test.intercept_
        elif param_type=='train':
            coef = lr_train.coef_
            intercept = lr_train.intercept_
        else:
            coef = lr_train.coef_
            intercept = lr_train.intercept_
        cef_dict=dict(zip(list(train_x.columns.values),list(coef[0][:])))
        cef_dict.update({'intercept':intercept[0]})

        result_dict=self.get_evaluate_index(test_y,test_prob_1,train_y,train_prob_1,oot_y,oot_prob_1)
        return result_dict,cef_dict

    def adaptive_logical_train(self,x,y,p0=400,pdo=30,
                            predict_stat_file='predict_stat.csv',
                            predict_file='predict_file.csv',
                            oot_x=pd.DataFrame(),
                            oot_y=pd.DataFrame()
                            ):

        import sklearn.metrics as metrics
        from sklearn.model_selection import train_test_split
        from adaptive_logistics import adaboost_logistics_train
        
        #得到所有的x和y
        merged_x=pd.concat([x,oot_x])
        merged_y=pd.concat([y,oot_y])

        if y[1]in [0,1]:
            y=y.ravel()
        #划分训练集测试集
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
        '''
        split_point = int(x.shape[0] * 0.5)
        train_x = x.loc[:split_point - 1, :]
        train_y = y[:split_point]
        test_x = x.loc[split_point:, :]
        test_y = y[split_point:]
        '''
        lr_test = adaboost_logistics_train()
        lr_test.fit(train_x, train_y)

        lr_all = adaboost_logistics_train()
        lr_all.fit(x, y)

        lr_out = adaboost_logistics_train()
        lr_out.fit(merged_x, merged_y)
        
        #测试集
        test_prob=lr_test.predict_proba(test_x)
        test_prob_1=test_prob[:,1]

        #训练集集
        train_prob=lr_test.predict_proba(train_x)
        train_prob_1=train_prob[:,1]

        #样本外数据集
        oot_prob=lr_all.predict_proba(oot_x)
        oot_prob_1=oot_prob[:,1]

        result_dict=self.get_evaluate_index(test_y,test_prob_1,train_y,train_prob_1,oot_y,oot_prob_1)

        return result_dict


    def compute_shap(self,model,data):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
        shap_data = pd.DataFrame(data=shap_values, columns=data.columns)
        shap_data.to_csv('shap_data.csv', index=False)
        shap.summary_plot(shap_values, data, show=False, auto_size_plot=True)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.savefig('shape.png', bbox_inches='tight')



    def xgboost_train(self,x,y,p0=400,pdo=30,
                                predict_stat_file='predict_stat.csv',
                                predict_file='predict_file.csv',
                                oot_x=pd.DataFrame(),
                                oot_y=pd.DataFrame(),
                                n_estimators=100,
                                eta=0.02,
                                max_depth=5,
                                min_child_weight=0.03,
                                colsample_bytree=0.7,
                                compute_shap=False
                                ):
        import shap

        import xgboost as xgb
        from sklearn.model_selection import train_test_split


        #得到所有的x和y
        merged_x=pd.concat([x,oot_x])
        merged_y=pd.concat([y,oot_y])

        if y[1]in [0,1]:
            y=y.ravel()
        #划分训练集测试集
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

        xgb_test = xgb.XGBClassifier(max_depth=max_depth,
                                     eta=eta,
                                     min_child_weight=min_child_weight,
                                     n_estimators=n_estimators,
                                     objective='binary:logistic',
                                     eval_metric='auc',
                                     colsample_bytree=colsample_bytree)
        xgb_test.fit(train_x, train_y)

        xgb_train = xgb.XGBClassifier(max_depth=max_depth,
                                    eta=eta,
                                    min_child_weight=min_child_weight,
                                    n_estimators=n_estimators,
                                    objective='binary:logistic',
                                    eval_metric='auc',
                                    colsample_bytree=colsample_bytree)
        xgb_train.fit(x, y)
        if compute_shap:
            #计算shap
            self.compute_shap(xgb_train,x)


        xgb_oot = xgb.XGBClassifier(max_depth=max_depth,
                                    eta=eta,
                                    min_child_weight=min_child_weight,
                                    n_estimators=n_estimators,
                                    objective='binary:logistic',
                                    eval_metric='auc',
                                    colsample_bytree=colsample_bytree)
        xgb_oot.fit(merged_x, merged_y)

        #测试集
        test_prob=xgb_test.predict_proba(test_x)
        test_prob_1=test_prob[:,1]

        #训练集集
        train_prob=xgb_test.predict_proba(train_x)
        train_prob_1=train_prob[:,1]

        #样本外数据集
        oot_prob=xgb_train.predict_proba(oot_x)
        oot_prob_1=oot_prob[:,1]

        result_dict=self.get_evaluate_index(test_y,test_prob_1,train_y,train_prob_1,oot_y,oot_prob_1)
        return result_dict

    def random_forest(self, x, y, p0=400, pdo=30,
                      predict_stat_file='predict_stat.csv',
                      predict_file='predict_file.csv',
                      model_file='xgb.pkl'):

        from sklearn.ensemble import RandomForestClassifier
        import sklearn.metrics as metrics
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score
        import xgboost as xgb
        from xgboost import plot_importance
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        '''
        split_point = int(x.shape[0] * 0.8)
        x_train = x.loc[:split_point - 1, :]
        y_train = y[:split_point]
        x_test = x.loc[split_point:, :]
        y_test = y[split_point:]
        '''

        # 构造变量
        from sklearn.model_selection import GridSearchCV
        forest = RandomForestClassifier(max_depth=3,
                                        n_estimators=100,
                                        min_samples_leaf=150,
                                        random_state=1,
                                        n_jobs=-1,
                                        max_features='sqrt')

        forest.fit(x_train, y_train)
        joblib.dump(forest, filename=model_file)
        print(model_file)
        y_predict_test = forest.predict(x_test)
        prob_test = forest.predict_proba(x_test)
        prob_test_0 = prob_test[:, 0]

        # 计算ks
        tr = train()
        s_ks = tr.get_ks_value(y_test, prob_test_0)
        ks_threshold = tr.get_ks_prob(y_test, prob_test_0)
        # 计算auc
        prob_test_1 = prob_test[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y_test, prob_test_1)
        auc_test = metrics.auc(fpr, tpr)
        # 概率
        prob_data = pd.DataFrame({'pro_0': prob_test_0, 'prob_1': prob_test_1, 'y_test': y_test})
        prob_data.sort_values(by='prob_1', inplace=True, ascending=False)
        prob_data.reset_index(drop=True, inplace=True)
        prob_data['period'] = pd.qcut(prob_data.index.values, 10, labels=np.arange(1, 11))
        # prob_data.to_csv(predict_file)
        s_lift_rate = (prob_data.loc[prob_data['period'] == 1, 'y_test'].mean()) / (prob_data['y_test'].mean())

        predict_result = pd.DataFrame({'y_test': y_test, 'y_predict': y_predict_test})
        # 计算召回率
        rate1 = predict_result.loc[predict_result['y_test'] == 1, 'y_predict'].sum() / predict_result.loc[
            predict_result['y_test'] == 1, 'y_predict'].count()
        rate2 = predict_result.loc[predict_result['y_predict'] == 1, 'y_test'].sum() / predict_result.loc[
            predict_result['y_predict'] == 1, 'y_test'].count()
        rate3 = predict_result.loc[predict_result['y_test'] == 1, 'y_predict'].sum() / predict_result.loc[:,
                                                                                       'y_predict'].count()
        s_f01 = 2 * rate1 * rate2 / (rate1 + rate2)
        # 总体样本
        y_predict_total = forest.predict(x)
        prob_total = forest.predict_proba(x)
        prob_total_0 = prob_total[:, 0]
        t_ks = tr.get_ks_value(y, prob_total_0)
        prob_total_1 = prob_total[:, 1]
        fpr, tpr, threshold = metrics.roc_curve(y, prob_total_1)
        auc_total = metrics.auc(fpr, tpr)

        predict_result = pd.DataFrame({'y_test': y, 'y_predict': y_predict_total})
        # 计算召回率
        rate4 = predict_result.loc[predict_result['y_test'] == 1, 'y_predict'].sum() / predict_result.loc[
            predict_result['y_test'] == 1, 'y_predict'].count()
        rate5 = predict_result.loc[predict_result['y_predict'] == 1, 'y_test'].sum() / predict_result.loc[
            predict_result['y_predict'] == 1, 'y_test'].count()
        rate6 = predict_result.loc[predict_result['y_test'] == 1, 'y_predict'].sum() / predict_result.loc[:,
                                                                                       'y_predict'].count()
        # 计算排序后的情况
        b = pdo / math.log(2)
        prob_data = pd.DataFrame({'pro_0': prob_total_0, 'prob_1': prob_total_1, 'y_test': y})
        # prob_data.sort_values(by='prob_1',inplace=True,ascending=False)
        prob_data.reset_index(drop=True, inplace=True)

        prob_data['odds'] = prob_data['prob_1'] / prob_data['pro_0']
        prob_data['odds'] = prob_data['odds'].map(lambda x: np.log(x))
        prob_data['score'] = p0 - prob_data['odds'] * b
        print(prob_data.isnull().sum())
        print(prob_data)
        bins_list = pd.qcut(prob_data['prob_1'], 10, duplicates='drop', retbins=True)[1]
        prob_data['period'] = pd.cut(prob_data['score'], bins=bins_list, labels=np.arange(1, len(bins_list)))
        prob_data.to_csv(predict_file)
        prob_data.pivot_table(index='period', columns='y_test', values='score', aggfunc='count').to_csv(
            predict_stat_file, index=True)

        t_lift_rate = (prob_data.loc[prob_data['period'] == 1, 'y_test'].mean()) / (prob_data['y_test'].mean())

        t_f01 = 2 * rate4 * rate5 / (rate4 + rate5)
        t_prob_mean = prob_total_1.mean()
        t_prob_std = prob_total_1.std()
        ks_mean = (s_ks + t_ks) / 2
        auc_mean = (auc_test + auc_total) / 2
        asses = 2 * ks_mean * auc_mean / (ks_mean + auc_mean)

        # asses=(s_f01+t_f01)/2
        result_dict = {'s_recall_rate': [rate1],
                       's_accurate_rate': [rate2],
                       's_response_rate': [rate3],
                       's_f01': [s_f01],
                       's_ks': [s_ks],
                       's_auc': [auc_test],
                       's_lift_rate': [s_lift_rate],
                       't_recall_rate': [rate4],
                       't_accurate_rate': [rate5],
                       't_response_rate': [rate6],
                       't_f01': [t_f01],
                       't_ks': [t_ks],
                       't_auc': [auc_total],
                       't_lift_rate': [t_lift_rate],
                       't_prob_mean': [t_prob_mean],
                       't_prob_std': [t_prob_std],
                       'threshold': [ks_threshold],
                       'ks_mean': [ks_mean],
                       'auc_mean': [auc_mean],
                       'asses': [asses]}
        return result_dict
    def xgboost_grid_search(self,x,y):
        from sklearn.metrics import accuracy_score
        from xgboost import XGBClassifier
        from sklearn.metrics import r2_score
        split_point = int(x.shape[0] * 0.5)
        x_train = x.loc[:split_point - 1, :]
        y_train = y[:split_point]
        x_test = x.loc[split_point:, :]
        y_test = y[split_point:]
        #构造变量
        from sklearn.model_selection import GridSearchCV
        '''
        #factor_record.set_index('feature',inplace=True)
        my_model = XGBClassifier(objective='multi:softmax')  # xgb.XGBClassifier() XGBoost分类模型
        print(x_train.isnull().sum())
        my_model.fit(x_train, y_train, verbose=False)
        predictions = my_model.predict(x_test)
        '''
        import matplotlib.pyplot as plt
        from xgboost import plot_importance

        # 第七步：开始训练
        from sklearn.model_selection import GridSearchCV
        parameters={'n_estimators':[90],
                    'max_depth':[7],
                    'learning_rate': [0.3],
                    'min_child_weight':range(5, 21, 1),
                    #'subsample':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    #'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    #'colsample_bytree':[0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    #'colsample_bylevel':[0.5, 0.6, 0.7, 0.8, 0.9, 1]
                    }

        cv_params={'n_estimators':[1,2,3,4,5,6],
                   'max_depth':[3,4,5,6,7,8,9,10],
                   'min_child_weight':[1,2,3,4,5,6],
                   'gamma':[0.1,0.2,0.3,0.4,0.5,0.6],
                   'subsample':[0.6,0.7,0.8,0.9],
                   'colsample_btree':[0.6,0.7,0.8,0.9],
                   'reg_alpha':[0.05,0.1,1,2,3],
                   'reg_lambda':[0.05,0.1,1,2,3],
                   'learning_rate':[0.01,0.05,0.07,0.1,0.2]}
        model_seed = 100
        '''
        model = XGBClassifier(learning_rate=0.1,
                              n_estimators=500,  # 树的个数--1000棵树建立xgboost
                              max_depth=5,  # 树的深度
                              min_child_weight=1,  # 叶子节点最小权重
                              seed=0,
                              subsample=0.8,  # 随机选择80%样本建立决策树
                              colsample_btree=0.8,  # 随机选择80%特征建立决策树
                              gamma=0,  # 惩罚项中叶子结点个数前的参数
                              reg_alpha=0,
                              reg_lambda=1
                              )
        '''
        model = XGBClassifier()
        gs=GridSearchCV(estimator= model,
                        param_grid=cv_params,
                        scoring='accuracy',
                        cv=5,
                        verbose=1,
                        n_jobs= 4
                        )
        gs.fit(x_train,y_train)
        print ('最优参数: ' + str(gs.best_params_))

        y_pred = gs.predict(x_test)
        ### model evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print("accuarcy: %.2f%%" % (accuracy * 100.0))

        result_data=pd.DataFrame({'y_test':y_test,'y_pred':list(y_pred)})
        print(result_data['y_test'].value_counts())
        tn=result_data.loc[(result_data['y_test']==0)&(result_data['y_pred']==0)].shape[0]
        fn=result_data.loc[(result_data['y_test']==1)&(result_data['y_pred']==0)].shape[0]
        tp=result_data.loc[(result_data['y_test']==1)&(result_data['y_pred']==1)].shape[0]
        fp=result_data.loc[(result_data['y_test']==0)&(result_data['y_pred']==1)].shape[0]
        precise=tp/(tp+fp)
        recall_rate=tp/(tp+fn)
        f01=2*precise*recall_rate/(precise+recall_rate)
        result= {
            'accuracy':[accuracy],
            'precise':[precise],
            'recall_rate':[recall_rate],
            'f01':[f01],
        }
        return result

