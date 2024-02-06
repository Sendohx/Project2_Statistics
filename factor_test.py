"""
统计及可视化之后，依据因子逻辑及统计表现：
判断如何处理因子值（是否进行正态处理，是否处理偏度）；
如何进行因子有效性的检验（选择合适的模型，合适的评价函数）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from tqdm import tqdm

class FactorTest:
    """"""
    def __init__(self, asset, data, factor, start_date, end_date, reg_start_date, reg_end_date):
        self.asset = asset
        self.data = data
        self.factor = factor
        self.start_date = start_date
        self.end_date = end_date
        self.reg_start_date = reg_start_date
        self.reg_end_date = reg_end_date

    # 缺失值处理
    def process_na(self, method):
        if method == 'ffill':
            self.data[self.factor].fillna(method, limit=2, inplace=True) # 最大连续填充期为2
        elif method == 'bfill':
            self.data[self.factor].fillna(method, limit=2, inplace=True)
        elif method == 'zero':
            self.data[self.factor].fillna(0, inplace=True)
        elif method == 'mean':
            self.data[self.factor].fillna(self.data[self.factor].mean(), inplace=True)
        elif method == 'median':
            self.data[self.factor].fillna(self.data[self.factor].median(), inplace=True)
        elif method == 'linear':
            self.data[self.factor].interpolate(method='linear', inplace=True)  # other methods: 'spline'(order), 'polynomial'(order)
        elif method == 'dropna':
            self.data = self.data.dropna(subset=[self.factor])

    # 异常值处理，如果箱型图显示有比较多的异常值，需进行处理
    def MAD(self, threshold=3.0, cut=True):
        df = self.data.copy()
        median = np.median(df[self.factor])
        diff = np.abs(df[self.factor] - median)
        med_abs_deviation = np.median(diff)
        upper = df[self.factor].median() + med_abs_deviation * threshold
        lower = df[self.factor].median() - med_abs_deviation * threshold
        if cut is True:
            df.drop(df[(df[self.factor] < lower) | (df[self.factor] > upper)].index, axis=0, inplace=True)
        else:
            df.loc[df[df[self.factor] <= lower].index, self.factor] = lower
            df.loc[df[df[self.factor] >= upper].index, self.factor] = upper
        return df

    # 标准化
    def Z_Score(self, window):  ## 分位数转换，也是正态化的一种
        temp_mean = self.data[self.factor].rolling(window).mean()
        temp_std = self.data[self.factor].rolling(window).std(ddof=1)
        self.data['stad_' + self.factor] = (self.data[self.factor] - temp_mean) / temp_std

    def min_max_scaling(self, window):
        self.data['stad_' + self.factor] = (self.data[self.factor].rolling(window)
                                            .apply(lambda x: (x - x.min()) / (x.max() - x.min()), raw=True))

    def rank_trans(self, window):
        self.data['ranked_' + self.factor] = (self.data[self.factor].rolling(window)
                                              .apply(lambda x: pd.rank(x)[-1] / (len(x) + 1)))

    # 正态化
    def log_trans(self):
        self.data['log_' + self.factor] = np.log(self.data[self.factor])

    def pow_trans(self, power):
        self.data['pow_' + self.factor] = self.data[self.factor] ** power

    def box_cox_trans(self):
        self.data['box_cox_' + self.factor], lambda_ = stats.boxcox(self.data[self.factor])

    # 去趋势
    ## 历史分位值
    def formula_q(self, numbers, tier_size):
        """ """
        if pd.isna(numbers):
            return None
        else:
            if tier_size == 1:
                values = list(range(0, 100, 1))
                q_values = min(values, key=lambda x: abs(x - numbers))
            else:
                values = list(range(0, 100, tier_size))
                q_values = max(val for val in values if val < numbers)
            return q_values

    def transform_to_q(self, window, tier_size):
        factor_ranks = self.data[self.factor].rolling(window).rank() / window * 100
        self.data[self.factor + f'_Q_{tier_size}'] = (factor_ranks.
                                            apply(lambda x: self.formula_q(x, tier_size) if not pd.isna(x) else None))

    ## Hodrick-Prescott 滤波
    def hp_filter(self, lamb):
        self.data[self.factor + '_cycle'], self.data[self.factor + '_trend'] = hpfilter(self.data[self.factor], lamb)

    def IC(self, windows):
        self.data['forward_return'] = self.data['return'].shift(-1)
        for window in windows:
            self.data[f'IC_{window}'] = self.data['stad_'+self.factor].rolling(window).corr(self.data['forward_return'], method='pearson')
            self.data[f'RankIC_{window}'] = self.data[self.factor].rolling(window).corr(self.data['forward_return'], method='spearman')
            self.data[f'ICIR_{window}'] = self.data[f'IC_{window}'].rolling(window).apply(lambda x: x.mean()/x.std())
            self.data[f'RankICIR_{window}'] = self.data[f'RankIC_{window}'].rolling(window).apply(lambda x: x.mean() / x.std())

    def ols_regress(self, window):
        ols_df = self.data[(self.data['date']>=self.reg_start_date)&(self.data['date']<=self.reg_end_date)].reset_index(drop=True)
        ols_df['pred_return'] = None
        ols_df[f'{self.factor}_coeff'] = None
        ols_df[f'{self.factor}_t_value'] = None
        ols_df['Rsquare'] = None

        total_iterations = len(ols_df)-window
        with tqdm(total=total_iterations) as pbar:
            for i in range(total_iterations):
                # fit
                x = sm.add_constant(ols_df.loc[i:i+window-1, self.factor])
                y = ols_df.loc[i:i+window-1, 'forward_return']
                model = sm.OLS(y, x).fit()
                # predict
                y_pred = model.predict([1, ols_df.loc[i+window, self.factor]])
                # 因子载荷，对应t值，模型R方，预测
                ols_df.loc[i+window-1, f'reg_coeff'] = model.params[1]
                ols_df.loc[i+window-1, f'coeff_t_value'] = model.tvalues[1]
                ols_df.loc[i+window-1, f'reg_Rsquare'] = model.rsquared
                ols_df.loc[i+window-1, f'reg_pred_return'] = y_pred[0]
                pbar.set_description(f"Processing item {i + 1}/{total_iterations}")
                pbar.update(1)
                #pbar.set_postfix(Result=i)
            print("Loop completed.")
        ols_df[f'reg_coeff_P'] = ols_df[f'reg_coeff'].rolling(window).apply(lambda x: adfuller(x)[1])
        ols_df[f'returns_corr'] = ols_df[f'reg_pred_return'].rolling(window).corr(self.data['forward_return'])
        ols_df[f'returns_corr_P'] = ols_df[f'returns_corr'].rolling(window).apply(lambda x: adfuller(x)[1])
        self.data = ols_df

    def resample(self):
        self.data = self.data[(self.data['date']>=self.start_date)&(self.data['date']<=self.end_date)]

    def period_test(self, test_list):  # 整体和分年度
        self.data['year'] = self.data['date'].str[:4]
        x = self.data['date']
        dates = pd.to_datetime(x)
        # 画趋势图
        fig, axes = plt.subplots(len(test_list)+1, 1, figsize=(20,10))
        # 循环画图
        for object, ax in zip(test_list, axes[:-1]):
            ax.plot(x, self.data[object], label=object) 
            # ax.bar(x, data, label=object)
            ax.xaxis.set_major_locator(plt.MaxNLocator(15))
            ax.set_title(f'{object}_Trend Chart')
            ax.legend()

        df = pd.DataFrame(columns=['2013', '2014', '2015', '2016', '2017', '2018',
                                   '2019', '2020', '2021', '2022', '2023', 'all_time'])
        for object in test_list:
            temp_mean = self.data.groupby('year')[object].mean()
            temp_IR = self.data.groupby('year')[object].apply(lambda x: x.mean()/x.std())
            # temp_win_rate = self.data.groupby('year')[object].apply(lambda x: (x>0).mean())
            df = df._append(
                pd.DataFrame(data=np.array([temp_mean.to_list() + [self.data[object].mean()],
                            temp_IR.to_list() + [self.data[object].mean()/self.data[object].std(ddof=1)]]),
                            columns=['2013', '2014', '2015', '2016', '2017', '2018','2019', '2020', '2021',
                                     '2022', '2023', 'all_time'], index=[f'{object}_Mean', f'{object}_IR']),
                            ignore_index=False)
        df = df.round(4)
        df.index.name = 'Test Objects'
        last_ax = axes[-1]
        last_ax.axis('off')  # Hide the axis
        table = last_ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)
        plt.tight_layout()

        return fig

    def factor_binning_test(self, tier_size1, tier_size2):
        # factor_binning
        ## 箱间单调性
        fig2, ax2 =  plt.subplots(figsize=(20, 6))
        self.transform_to_q(window=242, tier_size=tier_size1)
        temp_mean_1 = self.data.groupby(self.factor + f'_Q_{tier_size1}')[f'reg_coeff'].mean()
        temp_std_1 = self.data.groupby(self.factor + f'_Q_{tier_size1}')[f'reg_coeff'].std()
        temp_IR_1 = temp_mean_1 / temp_std_1
        #temp_winrate_1 = self.data.groupby(self.factor + f'_Q_{tier_size}')[f'{self.factor}_coeff'].apply(lambda x: (x>0).mean())
        temp_mean_2 = self.data.groupby(self.factor + f'_Q_{tier_size1}')[f'returns_corr'].mean()
        temp_std_2 = self.data.groupby(self.factor + f'_Q_{tier_size1}')[f'returns_corr'].std()
        temp_IR_2 = temp_mean_2 / temp_std_2
        #temp_winrate_2 = self.data.groupby(self.factor + f'_Q_{tier_size}')[f'return_corr'].apply(lambda x: (x>0).mean())
        summary = pd.concat([temp_mean_1, temp_std_1, temp_IR_1,
                                  temp_mean_2, temp_std_2, temp_IR_2], axis=1)
        summary.columns =  ['reg_coeff_mean', 'reg_coeff_std', 'reg_coeff_IR',
                            'return_corr_mean', 'returns_corr_std', 'returns_corr_IR']
        summary.index.name = 'Tier'
        summary = summary.round(4)
        summary.reset_index(inplace=True)
        ax2.axis('off')  # Hide the axis
        table = ax2.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
        table.set_fontsize(14)
        table.scale(1, 2)

        ## 箱内解释力 根据分位值的档画趋势图
        self.transform_to_q(window=242, tier_size=tier_size2)
        fig1, axes = plt.subplots(5,4, figsize=(20,20))
        for i, ax in enumerate(axes.flatten()):
            temp_df = self.data[self.data[self.factor + f'_Q_5']==5*i].reset_index(drop=True)
            x = temp_df.index
            y1 = temp_df[f'reg_coeff']
            y2 = temp_df[f'returns_corr']
            ax.plot(x, y1, label='coeff')
            ax.plot(x, y2, label='return_corr')
            ax.set_title(f'Q = {5*i}')
            ax.legend()
        plt.tight_layout()
        return fig1, fig2

    def return_binning_test(self):
        # return_binning
        ret_upper_threshold = self.data['forward_return'].mean() + 3 * self.data['forward_return'].std(ddof=1)
        ret_lower_threshold = self.data['forward_return'].mean() - 3 * self.data['forward_return'].std(ddof=1)

        temp1 = self.data[self.data['forward_return'] > 0]
        temp2 = self.data[self.data['forward_return'] < 0]
        temp3 = self.data[(self.data['forward_return'] > ret_upper_threshold)|(self.data['forward_return'] < ret_lower_threshold)]

        x1 = temp1.reset_index(drop=True).index
        x2 = temp2.reset_index(drop=True).index
        x3 = temp3.reset_index(drop=True).index

        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)

        bar_width = 0.5
        x1_indexes = np.arange(len(x1))
        bar11_x = x1_indexes - bar_width / 2
        bar12_x = x1_indexes + bar_width / 2
        ax1.bar(bar11_x, temp1[f'reg_coeff'], label='coeff')
        ax1.bar(bar12_x, temp1[f'returns_corr'], label='return_corr')
        ax1.xaxis.set_visible(False)
        ax1.set_title('positive return')
        ax1.legend()

        x2_indexes = np.arange(len(x2))
        bar21_x = x2_indexes - bar_width / 2
        bar22_x = x2_indexes + bar_width / 2
        ax2.bar(bar21_x, temp2[f'reg_coeff'], label='coeff')
        ax2.bar(bar22_x, temp2[f'returns_corr'], label='return_corr')
        ax2.xaxis.set_visible(False)
        ax2.set_title('negative return')
        ax2.legend()

        x3_indexes = np.arange(len(x3))
        bar31_x = x3_indexes - bar_width / 2
        bar32_x = x3_indexes + bar_width / 2
        ax3.bar(bar31_x, temp3[f'reg_coeff'], label='coeff')
        ax3.bar(bar32_x, temp3[f'returns_corr'], label='return_corr')
        ax3.xaxis.set_visible(False)
        ax3.set_title('anomalous return')
        ax3.legend()

        return fig
