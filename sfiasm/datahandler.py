from statsmodels.tsa.stattools import bds
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import kurtosis
from statistics import mean, stdev
from sklearn import linear_model, metrics
import numpy as np
from sfiasm.agents import TrendRegressor
from statsmodels.tsa.stattools import kpss
from arch.unitroot import ADF, DFGLS
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


ACF_DECORATOR = True
ACF_TYPE = 'ordinary'
NUM_LAGS = 40


# Decorator
def acfplot(func):
    def wrapper(*args):
        if ACF_TYPE == 'ordinary':
            plot_acf(func(*args), lags=np.arange(1, NUM_LAGS), c='r', title=f'Autocorrelation of {func.__name__}')
            plt.grid()
        elif ACF_TYPE == 'partial':
            plot_pacf(func(*args), lags=np.arange(1, NUM_LAGS), c='r', title=f'Partial Autocorrelation of {func.__name__}')
            plt.grid()
    return wrapper


# # Conditional Decorator
# def return_plot(dec, apply):
#     def decorator(func):
#         if not apply:
#             # Return the function unchanged, not decorated.
#             return func
#         return dec(func)
#     return decorator


def sample_autocovariance(sample, window):
    n = len(sample)
    sampleMean = mean(sample)
    sumstd = 0
    for t in range(n-window):
        sumstd += (sample[t+window] - sampleMean) * (sample[t] - sampleMean)
    return sumstd/n


def sample_autocorrelation(sample, window):
    gamma_h = sample_autocovariance(sample, window)
    gamma_0 = sample_autocovariance(sample, 0)
    return gamma_h / gamma_0


def returns_kurtosis(priceseries, divseries):
    returns = np.zeros(len(priceseries)-1)

    for i in range(len(returns)):
        returns[i] = ((priceseries[i+1] + divseries[i+1] - priceseries[i]) / priceseries[i]) * 100

    return kurtosis(returns)


def std(data):
    return stdev(data)


def arch(data):
    return het_arch(data, ddof=1, nlags=12)


def bds_test(data, distance):
    return bds(data, distance=distance)


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')


def adf_gls(series):
    dfgls = DFGLS(series)
    print(dfgls.summary().as_text())


def adf_test(series):
    adf = ADF(series)
    print(adf.summary().as_text())


def find_residuals(y_train, X_train):

    reg = linear_model.LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_train)

    return y_pred - y_train


def excess_return(priceseries, divseries, rf):
    excess = []
    for t in range(len(priceseries) - 1):
        excess.append(((priceseries[t+1] + divseries[t+1] - priceseries[t]) / priceseries[t] - rf) * 100)
    return mean(excess)


def squared_return(priceseries, divseries):
    squared = []
    for t in range(len(priceseries) - 1):
        squared.append(((priceseries[t+1] + divseries[t+1] - priceseries[t]) / priceseries[t]) ** 2)
    return squared


def absolute_return(priceseries, divseries):
    absolute = []
    for t in range(len(priceseries) - 1):
        absolute.append(abs((priceseries[t+1] + divseries[t+1] - priceseries[t]) / priceseries[t]))
    return absolute


def log_return(priceseries, divseries):
    log = []
    for t in range(len(priceseries) - 1):
        log.append(np.log(priceseries[t+1] + divseries[t+1]) - np.log(priceseries[t]))
    return log


def simple_return(priceseries, divseries):
    returns = [((priceseries[i+1] + divseries[i+1] / priceseries[i]) - 1) for i in range(len(priceseries) - 1)]
    for i in range(0, len(returns), 500):
        try:
            returns[i: i+500] -= np.mean(returns[i: i+500])
        except IndexError:
            returns[i:] -= np.mean(returns[i:])
    return returns


@acfplot
def squared_return_acf(priceseries, divseries):
    squared = []
    for t in range(len(priceseries) - 1):
        squared.append(((priceseries[t+1] + divseries[t+1] - priceseries[t]) / priceseries[t]) ** 2)
    return squared


@acfplot
def absolute_return_acf(priceseries, divseries):
    absolute = []
    for t in range(len(priceseries) - 1):
        absolute.append(abs((priceseries[t+1] + divseries[t+1] - priceseries[t]) / priceseries[t]))
    return absolute


@acfplot
def log_return_acf(priceseries, divseries):
    log = []
    for t in range(len(priceseries) - 1):
        log.append(np.log(priceseries[t+1] + divseries[t+1]) - np.log(priceseries[t]))
    return log


@acfplot
def log_absprices_return_acf(priceseries):
    log = []
    for t in range(len(priceseries) - 1):
        log.append(np.log(priceseries[t+1]) - np.log(priceseries[t]))
    return log


def realized_volatility(returns):
    returns_squared = np.square(returns)
    rv = np.sqrt(np.sum(returns_squared))
    return rv


def universal_trend(market, tau):
    reg = linear_model.LinearRegression().fit(np.arange(tau).reshape(-1, 1), market.priceTimeSeries[-tau:])
    y_pred = reg.coef_ * market.priceTimeSeries[-tau:] + reg.intercept_
    variance = metrics.mean_squared_error(market.priceTimeSeries[-tau:], y_pred)

    if variance < TrendRegressor.minvar:
        variance = TrendRegressor.minvar

    return variance, reg.coef_[0], reg.intercept_


def report_stats(y_train, X_train, priceSeries, divSeries, volumeSeries):
    if np.size(priceSeries) == len(priceSeries):

        residuals = find_residuals(y_train, X_train)
        price_vol = np.std(priceSeries)
        res_std = std(residuals)
        kurt = returns_kurtosis(priceSeries, divSeries)
        res_kurt = kurtosis(residuals)
        res_corr = sample_autocorrelation(residuals, 1)
        res_corr_squared = sample_autocorrelation(residuals ** 2, 1)
        res_arch = arch(residuals)
        res_bds = bds_test(residuals, distance=0.5)
        excess_ret = excess_return(priceSeries, divSeries, 0.1)
        trading_volume = mean(volumeSeries)
        print(f"std = {res_std: 4f}; returns kurtosis = {kurt: 4f}; rho = {res_corr: 4f}; rho_squared = {res_corr_squared: 4f}"
              f"\n ARCH = {res_arch} \n BDS = {res_bds} \n excess retrun = {excess_ret: 4f}"
              f" \n trading volume = {trading_volume: 4f}, price volatility = {price_vol: 4f}, "
              f"residuals kurtosis = {res_kurt: 4f}")

    else:
        residuals = [[]] * len(priceSeries)
        for i in range(len(priceSeries)):
            pricecumdiv = np.add(priceSeries[i], divSeries[i])
            price_vol = np.std(priceSeries[i])
            residuals = find_residuals(pricecumdiv)
            res_std = std(residuals)
            kurt = returns_kurtosis(priceSeries[i], divSeries[i])
            res_corr = sample_autocorrelation(residuals, 1)
            res_corr_squared = sample_autocorrelation(residuals ** 2, 1)
            res_arch = arch(residuals)
            res_bds = bds_test(residuals, distance=0.5)
            excess_ret = excess_return(priceSeries[i], divSeries[i], 0.1)
            trading_volume = mean(volumeSeries[i])
            print(
                f"std = {res_std: 4f}; kurtosis = {kurt: 4f}; rho = {res_corr: 4f}; rho_squared = {res_corr_squared: 4f}"
                f"\n ARCH = {res_arch} \n BDS = {res_bds} \n excess retrun = {excess_ret: 4f}"
                f" \n trading volume = {trading_volume: 4f}, price volatility = {price_vol: 4f}")

    return residuals
