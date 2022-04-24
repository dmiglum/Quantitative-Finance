
#Basic Quantitative Finance
from pandas_datareader import data
from tabulate import tabulate
import pandas as pd
import numpy as np
import scipy as sp
from pathlib import Path
import statistics as stat
import matplotlib.pyplot as plt
import tempfile
import seaborn as sns
import statsmodels.api as sm

tickers = ['gs', 'jpm', 'bac']
risk_free_asset = '^IRX' #13-week t-bill (expressed as a yearly rate)
print('Symbols: ' + " ".join(sym + ' ' for sym in tickers))

data_source = 'yahoo'
# year-month-day
start_date = '2019-01-01'
end_date = '2021-10-05'

#Downloading Stock prices and risk-free asset price
panel_data = data.DataReader(tickers, data_source, start_date, end_date)
close_prices = panel_data['Adj Close']

panel_data = data.DataReader(risk_free_asset, data_source, start_date, end_date)
rf_prices = panel_data['Adj Close']

# Basic time-series Statistics
close_prices.describe()
rf_prices.describe() #values for rf are quoted in yearly percentage return

# Time-series plots
close_prices.plot()

# Normalizing prices of bank stocks to compare
normalized_close_list = []
for sym in tickers:
    normalized = close_prices[sym] / close_prices[sym][0]
    normalized_close_list.append(normalized)
    
normalized_close_list = pd.DataFrame(normalized_close_list).transpose()
normalized_close_list.plot(grid=True, figsize=(10,6))

# Calculating Simple return, Excess Return, and Sharpe Ratio
def simple_return(time_series, period):
    return list((time_series[i] / time_series[i-period])- 1.0 for i in range(period, len(time_series)))

def return_df(time_series_df):
    """
    Given a data frame consisting of price time series, return a data frame
    that consists of the simple returns for the time series.
    """
    r_df = pd.DataFrame()
    col_names = time_series_df.columns
    index = time_series_df.index
    for col in col_names:
        col_vals = time_series_df[col]
        col_ret = simple_return(col_vals, 1)
        r_df[col] = col_ret
    return r_df.set_index(index[1:len(index)])

def excess_return_series(asset_return, risk_free):  #iterating across the Series and calculating excess return
    excess_ret_list = []
    for i, ret in enumerate(asset_return):
        excess_ret_list.append(ret - risk_free[i])
    return pd.DataFrame(excess_ret_list).set_index(asset_return.index)

def excess_return_df(asset_return, risk_free):  #iterating across the entire dataframe (using Series code from above)
    excess_df = pd.DataFrame()
    for i, col in enumerate(asset_return.columns):
        e_df = excess_return_series(asset_return[col], risk_free)
        excess_df.insert(i, col, e_df)
    return excess_df

def calc_sharpe_ratio(asset_return, risk_free):
    excess_return = excess_return_df(asset_return, risk_free)
    return_mean = []
    return_stddev = []
    for col in excess_return.columns:
        mu = stat.mean(excess_return[col])
        std = stat.stdev(excess_return[col])
        return_mean.append(mu)
        return_stddev.append(std)
    # daily Sharpe ratio
    # https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio
    sharpe_ratio = np.asarray(return_mean) / np.asarray(return_stddev)
    result_df = pd.DataFrame(sharpe_ratio).transpose()
    result_df.columns = asset_return.columns
    ix = asset_return.index
    ix_start = ix[0].date()
    ix_end = ix[len(ix)-1].date()
    index_str = f'{ix_start} : {ix_end}'
    result_df.index = [ index_str ]
    return result_df 

#Aligning Time Series data - Stock price dates and risk-free rate dates do not match, so we need to align them
# calculate the simple return for a set of stock close prices
r_df = return_df(close_prices)
"""
Make sure that all of the indexes are datatime objects
"""
r_df_index = pd.to_datetime(r_df.index)
rf_index = pd.to_datetime(rf_prices.index)
# Make sure that the index types match
assert(type(r_df_index) == type(rf_index))
# filter the close prices
matching_dates = r_df_index.isin( rf_index )
r_df_adj = r_df[matching_dates]
# filter the rf_prices
r_df_index = pd.to_datetime(r_df_adj.index)
matching_dates = rf_index.isin(r_df_index)
rf_prices_adj = rf_prices[matching_dates]
# check that all index elements now match
assert( (r_df_adj.index == rf_prices_adj.index).all() )
# Check that the DataFrame shapes are the same
assert(r_df_adj.shape[0] == rf_prices_adj.shape[0])

# Asset Return Time Series
fig, ax = plt.subplots(r_df_adj.shape[1], figsize=(10,8))
for i, col in enumerate(r_df_adj.columns):
    ax[i].set_xlabel(col)
    ax[i].grid(True)
    ax[i].plot(r_df_adj[col])
fig.tight_layout()
plt.show()

# Asset Return Distributions
colors = ['b', 'g', 'r']
numBins = int(np.sqrt(r_df_adj.shape[0])) * 4
fig, ax = plt.subplots(r_df_adj.shape[1], figsize=(10,8))
for i, col in enumerate(r_df_adj.columns):
    ax[i].set_xlabel( col )
    ax[i].set_ylabel('Count')
    ax[i].grid(True)
    ax[i].hist(r_df_adj[col], bins=numBins, facecolor=colors[i])
    ax[i].axvline(x=stat.mean(r_df_adj[col]), color='black')
fig.tight_layout()
plt.show()

# Daily Return Histograms with Density Plots
def hist_plot(values, numBins, color):
    sns.displot(data=values, kde=True, color=color, bins=numBins, stat='density')
    mean = stat.mean(values)
    sd = stat.stdev(values)
    min = np.min(values)
    max = np.max(values)
    x_linspace = np.linspace(min, max, len(values))
    pdf = sp.stats.norm.pdf(x_linspace, mean, sd)
    plt.plot(x_linspace, pdf, color='black')
    plt.axvline(x=mean, color='black')
    plt.show()

def hist_and_stats(close_prices, period):
    numBins = int(np.sqrt(len(close_prices)) * 4)
    r_stat= []
    ll = []
    for col_ix, col in enumerate(close_prices.columns):
        r_one_day = simple_return(close_prices[col], period)
        ll.append(r_one_day)
        hist_plot(r_one_day, numBins, colors[col_ix])
        mean = stat.mean(r_one_day)
        sd = stat.stdev(r_one_day)
        sem = sp.stats.sem(r_one_day)
        r_stat.append({'mean': mean, 'StdErr Mean': sem, 'sd': sd})
    r_mat = np.vstack(ll).transpose()
    ret_df = pd.DataFrame(r_mat)
    ret_df.columns = close_prices.columns
    r_stat_df = pd.DataFrame(r_stat)
    r_stat_df.index = close_prices.columns
    return ret_df, r_stat_df

# 1-day histograms and density plots
one_day_ret_df, r_stat_df = hist_and_stats(close_prices, 1)
print(tabulate(r_stat_df, headers=['stock', 'mean', 'StdErr Mean', 'stddev'], tablefmt="fancy_grid"))

# 4-day histograms and density plots
four_day_ret_df, r_stat_df = hist_and_stats(close_prices, 4)
print(tabulate(r_stat_df, headers=['stock', 'mean', 'StdErr Mean', 'stddev'], tablefmt="fancy_grid"))

# QQ Plots - QQ plots show the difference between a distribution and the data set
# 1-day QQ Plots
for col in one_day_ret_df.columns:
    sp.stats.probplot(one_day_ret_df[col], dist = 'norm', plot = plt)
    plt.title(f'Probability plot for {col}')
    plt.show()

# 4-day QQ plots    
for col in one_day_ret_df.columns:
    sp.stats.probplot(four_day_ret_df[col], dist="norm", plot=plt)
    plt.title(f'Probability plot for {col}')
    plt.show()

# Sharpe Ratio
# Converting Daily mean and std to a yearly Sharpe: Sharpe_year = (mean_day / std_day) * sqrt(252)
'''
There is a discussion on Stack Exchange about annualizing the Sharpe ratio
link: (https://quant.stackexchange.com/questions/2260/how-to-annualize-sharpe-ratio).
A paper by Andrew Lo, The Statistics of Sharpe Ratios is mentioned. Prof. Lo makes the 
point that because there is often a serial correlation in returns annualizing the Sharpe 
ratio in this way may yield a result that differs from the “correct” value by significant amounts.
Proper annualizing of the Sharpe ratio is complex
'''
# the risk free rate is reported as a yearly percentage return (e.g., 2 percent),
# so divide by 100.
rf_daily = (rf_prices_adj/100.0) / 365.00
trading_days = 252
rf_daily = rf_daily[rf_daily.columns[0]]
"""
Calculate the yearly Sharpe ratios for the assets
"""
sharpe_df = pd.DataFrame()
for end in range(r_df_adj.shape[0], 0, -trading_days):
    start = end - trading_days
    if (start > 0):
        ret_year = r_df_adj[start:end]
        rf_year = rf_daily[start:end]
        sharpe_ratio = calc_sharpe_ratio(ret_year, rf_year)
        sharpe_ratio_year = sharpe_ratio * np.sqrt( trading_days )
        sharpe_df = sharpe_df.append(sharpe_ratio_year)
    else:
        break

print(tabulate(sharpe_df, headers=['Date range', *sharpe_df.columns], tablefmt='fancy_grid'))

 