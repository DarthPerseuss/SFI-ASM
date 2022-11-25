from sfiasm.stocks import Market, BitCondition
from sfiasm.agents import TechnicalGA, GAagent
from sfiasm.specialist import Specialist
from sfiasm.world import WallStreet
import multiprocessing as mp
import pickle
import numpy as np
import copy

GA_AGENT_TYPE = 'technical_ga'
# Stock parameters

dividend_startvalue = 5
rho = 0.95
noise_sd = 0.27259
d_bar = 10
interest_rate = 0.1
init_price = d_bar / interest_rate

num_agents = 25
init_holding = 1
initialcash = 20000

risk_coef = 0.5
num_strategies = 100
learning_interval = 250
bit_mutation_rate = 0.03
crossover_rate = 0.1
rr_ratio = 0.5

max_stockprice = 200
min_stockprice = 0.01
min_excess = 0.01
eta = 0.005
specialist_iterations = 10
theta = 75
taup = 50


# ____________REE trading____________
# steps = 10000
# trade_ree(steps, gooz_market)


# ____________GA trading_____________
if __name__ == '__main__':

    n_pool = mp.cpu_count()
    n_runs = int
    if mp.cpu_count() == 4:
        n_runs = 6
    elif mp.cpu_count() == 8:
        n_runs = 3

    ratios = np.arange(0, .9, .1)
    for run in range(1):
        agent_params = []
        output_markets = []
        output_agents = []

        markets = [Market(init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate,
                          bitkeys=BitCondition.predefined_rulesets(query='original SFI-ASM')) for i in range(n_pool)]
        specialists = [Specialist(max_stockprice, min_stockprice, min_excess, eta,
                                  specialist_iterations, theta, taup) for i in range(n_pool)]
        global_means = []

        # if you need to have every market different
        #
        # for market in markets:
        #     for t in range(500):
        #         market.advance_arprocess()
        #         market.priceTimeSeries.append(market.dividend / interest_rate)
        #         market.price = market.priceTimeSeries[-1]

        # if you want the markets to start with the same conditions
        #
        for t in range(500):
            markets[0].advance_arprocess()
            markets[0].priceTimeSeries.append(markets[0].dividend / interest_rate)
            markets[0].price = markets[0].priceTimeSeries[-1]

        for i in range(1, n_pool):
            markets[i].divTimeSeries = copy.deepcopy(markets[0].divTimeSeries)
            markets[i].dividend = markets[0].dividend
            markets[i].priceTimeSeries = copy.deepcopy(markets[0].priceTimeSeries)
            markets[i].price = markets[0].price

        agents = [[GAagent(init_holding, initialcash, num_strategies, theta,
                           BitCondition.predefined_rulesets(query='original SFI-ASM'),
                           risk_coef)
                   for _ in range(num_agents)]
                  for i in range(n_pool)]

        wallstreets = [WallStreet(agents[i], learning_interval, bit_mutation_rate, crossover_rate, ratios[i])
                       for i in range(n_pool)]

        steps = 120000
        keywords = {'save_specificfracs': 1, 'save_profits': 1, 'print_fitness': 0, 'save_price': 1, 'save_div': 1}

        pools = []
        with mp.Pool(processes=n_pool) as pool:
            for i in range(n_pool):
                pools.append(pool.apply_async(wallstreets[i].trade,
                                              args=(steps, markets[i], specialists[i]), kwds=keywords))
            for i in range(n_pool):
                pools[i].get()

        agent_params += [pool._value for pool in pools]
        output_agents += agents
        output_markets += markets
        save_str = f'Data\Original SFI_ASM different rr_ratios.pickle'
        with open(save_str, 'wb') as handle:
            pickle.dump([output_agents, agent_params], handle, protocol=pickle.HIGHEST_PROTOCOL)
