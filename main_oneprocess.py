from sfiasm.stocks import Market, BitCondition
from sfiasm.agents import GAagent, TechnicalGA, UnderinformedAgent
from sfiasm.specialist import Specialist
from sfiasm.world import WallStreet
import matplotlib.pyplot as plt
import numpy as np
from cProfile import Profile
import pstats


GA_AGENT_TYPE = 'ga_12'
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

gene_length = int
if GA_AGENT_TYPE == 'technical_ga':
    gene_length = 32
elif GA_AGENT_TYPE == 'ga_12':
    gene_length = 12

risk_coef = 0.5
num_strategies = 100
learning_interval = 250
bit_mutation_rate = 0.03
crossover_rate = 0.1
rr_ratio = 0.5
rr_ratio_tech = 0.1

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

    agent_params = []
    markets = []
    agents_sets = []

    for i, err in enumerate([.2, .4, .7, 1, 1.2]):

        gooz_market = Market(init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate,
                             bitkeys=BitCondition.predefined_rulesets(query='original SFI-ASM'))
        gooz_specialist = Specialist(max_stockprice, min_stockprice, min_excess, eta, specialist_iterations, theta,
                                     taup)

        for t in range(500):
            gooz_market.advance_arprocess()
            gooz_market.priceTimeSeries.append(gooz_market.dividend / interest_rate)

        agents = [UnderinformedAgent(init_holding, initialcash, num_strategies, theta,
                                     BitCondition.predefined_rulesets(query='original SFI-ASM'),
                                     risk_coef, divdend_error_var=np.random.uniform(0.2, 1.7))
                  for _ in range(num_agents)]

        wallstreet = WallStreet(agents, learning_interval, bit_mutation_rate, crossover_rate, rr_ratio)

        steps = 1100

        agent_params.append(wallstreet.trade(steps, gooz_market, gooz_specialist, save_nactive=True,
                                             save_specificfracs=True, frequency_specific_frac=20,
                                             save_specbit_count=True))

        agents_sets.append(agents)
        markets.append(gooz_market)

    # steps = 4000
    # keywords = {'save_specificfracs': 1, 'save_profits': 1, 'print_fitness': 0}
    #
    # with Profile() as pr:
    #     agent_params = wallstreet.trade(steps, market, specialist, save_specificfracs=False,
    #                                     save_profits=True, save_nactive=True)
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()

    # plt.plot(market.priceTimeSeries)
    # plt.show()
