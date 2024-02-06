
import time
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

from Project1_Factors import price_volume as pv
from factor_test import FactorTest

# start
run_start_time = time.time()

# object parameters
assets = ['000985.CSI']
factor_list = ['yang_zhang_sigma', 'ILLIQ']
start_date = '20130101'
end_date = '20231231'
reg_start_date = '20120101' #避免回测期空值
reg_end_date = '20240110'
today = datetime.today().strftime('%Y%m%d')
data_root = '/nas92/xujiahao/data/raw'
save_root = '/nas92/xujiahao/factor_outcome/validity'

# temporary data dict
dict = {}
for asset in assets:
    raw_ind_data = pd.read_parquet(data_root + f'/ind_{asset}_{start_date}_{today}.parquet')
    volatility_data = pv.volatility_factors(raw_ind_data, 10)
    fluidity_data = pv.fluidity_factors(raw_ind_data, None)
    temp_df = pd.merge(fluidity_data, volatility_data, on=['date','symbol'])
    df = pd.merge(raw_ind_data, temp_df, on=['date','symbol'])
    # df = df[(df['date']>=start_date)&(df['date']<=end_date)]
    dict[asset] = df

# testing loop
for (asset, data) in dict.items():
    for factor in factor_list:
        ft = FactorTest(asset, data, factor, start_date, end_date, reg_start_date, reg_end_date)
        ft.process_na('drop')
        ft.Z_Score(60)
        ft.IC([30, 60, 120])
        ft.ols_regress(60)
        ft.resample()
        ft.data.to_parquet(save_root + f'/{asset}_{factor}.parquet')
        with PdfPages(save_root + f'/{asset}_{factor}.pdf') as pdf:
            pdf.savefig(ft.period_test(['IC_60', 'RankIC_60']))
            pdf.savefig(ft.period_test(['reg_coeff', 'returns_corr']))
            factor_binning_result = ft.factor_binning_test(10, 5)
            pdf.savefig(factor_binning_result[0])
            pdf.savefig(factor_binning_result[1])
            pdf.savefig(ft.return_binning_test())
        print(f'{factor} finished')
    print(f'{asset} finished')

# end
run_end_time = time.time()
run_time = run_end_time - run_start_time
print('Total run time: {:.2f} seconds'.format(run_time))