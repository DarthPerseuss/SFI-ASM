import pickle
from sfiasm.agents import TrendRegressor, NoiseTrader
from sfiasm.world import WallStreet
from sfiasm.specialist import Specialist
from sfiasm.stocks import Market
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm.notebook import tqdm
from sfiasm.datahandler import *
import pandas as pd
import copy
from scipy.ndimage.filters import uniform_filter1d
import multiprocessing as mp

with open('Data\SFI_ASM Objects.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

init_market = unserialized_data[0]
init_agents = unserialized_data[1]
# Specialist parameters

max_stockprice = 200
min_stockprice = 0.01
min_excess = 0.01
eta = 0.005
specialist_iterations = 10
theta = 75
taup = 50
nbits = len(init_market.marketCondition)

rho = 0.95
noise_sd = 0.27259
d_bar = 10
interest_rate = 0.1

learning_interval = 250
bit_mutation_rate = 0.03
crossover_rate = 0.1
# Trend follower parameters

min_horizon = 5
max_horizon = 500


num_noise = 10

# gamma needs to be played with
gamma = 1

# Start with homogeneous risk first
risk = 0.5

init_holding = 1
init_cash = 20000


def creatdf_tau(steps, reps, dataspan, ntrend=5, strend=True, mtrend=True, ltrend=True,
                trendtype='linreg', maxvar=100, maxdemand=2, noise_traders=0):
    temp = []
    if strend is True:
        temp.append('short tr profit')
    if mtrend is True:
        temp.append('mid tr profit')
    if ltrend is True:
        temp.append('long tr profit')

    df = pd.DataFrame(index=np.arange(reps), columns=['taus', 'price', 'dividend', 'ga profit',
                                                      'volume', 'pricecumdiv'] + temp)

    gooz_specialist = Specialist(max_stockprice, min_stockprice, min_excess, eta, specialist_iterations,
                                 theta, taup)
    markets = {}
    ga_agents_list = []
    params = []

    for i in range(reps):

        market = copy.deepcopy(init_market)
        ga_agents = copy.deepcopy(init_agents)
        trend_agents = []

        if strend is True:
            strend_agents = [TrendRegressor(init_holding, init_cash, risk, trendtype=trendtype,
                                            tau=random.randint(2, 10),
                                            gamma=gamma, maxvar=maxvar, maxdemand=maxdemand)
                             for _ in range(ntrend)]
            trend_agents += strend_agents

        if mtrend is True:
            mtrend_agents = [TrendRegressor(init_holding, init_cash, risk, trendtype=trendtype,
                                            tau=random.randint(20, 50),
                                            gamma=gamma, maxvar=maxvar, maxdemand=maxdemand)
                             for _ in range(ntrend)]
            trend_agents += mtrend_agents

        if ltrend is True:
            ltrend_agents = [TrendRegressor(init_holding, init_cash, risk, trendtype=trendtype,
                                            tau=random.randint(100, 200),
                                            gamma=gamma, maxvar=maxvar, maxdemand=maxdemand)
                             for _ in range(ntrend)]
            trend_agents += ltrend_agents

        taus = [agent.tau for agent in trend_agents]
        agents = trend_agents + ga_agents

        wallstreet = WallStreet(agents, learning_interval, bit_mutation_rate,
                                crossover_rate, rr_ratio=0.5)

        steps = steps

        if noise_traders > 0:
            noise_agents = [NoiseTrader(init_holding, init_cash, risk) for _ in range(noise_traders)]
            params = wallstreet.trade(steps, market, gooz_specialist, save_specificfracs=True)

        else:
            params.append(wallstreet.trade(steps, market, gooz_specialist, save_specificfracs=True, save_profits=True,
                                           save_strength=True, save_trendvars=True, save_trenddemands=True))

        df['ga profit'][i] = [agent.profit for agent in ga_agents]
        if strend is True:
            df['short tr profit'][i] = [agent.profit for agent in strend_agents]
        if mtrend is True:
            df['mid tr profit'][i] = [agent.profit for agent in mtrend_agents]
        if ltrend is True:
            df['long tr profit'][i] = [agent.profit for agent in ltrend_agents]

        df['price'][i] = market.priceTimeSeries[-dataspan:]
        df['dividend'][i] = market.divTimeSeries[-dataspan:]
        df['volume'][i] = params[i]['volume'][:]

        df['pricecumdiv'][i] = np.add(df['price'][i], df['dividend'][i])

        ga_agents_list.append(ga_agents)

    return df, ga_agents_list, trend_agents, params


# Run a sample
if __name__ == '__main__':
    num_trenders = 3
    steps = 200
    reps = 1
    dataspan = 500 + steps
    pools = []

    multiple_tau_df, ga_agents_list, trenders, parameters = creatdf_tau(steps, reps, dataspan, ntrend=5, strend=True,
                                                                        mtrend=False, ltrend=False, trendtype='expodecay',
                                                                        maxvar=100, maxdemand=5, noise_traders=0)