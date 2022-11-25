from specialist import Specialist
from world import WallStreet
import pickle
import copy

dividend_startvalue = 5
rho = 0.95
noise_sd = 0.27259
d_bar = 10
interest_rate = 0.1
init_price = d_bar / interest_rate

num_agents = 25
init_holding = 1
initialcash = 20000
gene_length = 32
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


with open('Data\SFI_ASM technical traders.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

init_tech = unserialized_data[1]

with open('Data\SFI_ASM Objects.pickle', 'rb') as handle:
    unserialized_data = pickle.load(handle)

init_market = unserialized_data[0]
init_agents = unserialized_data[1]

tech_agents = copy.deepcopy(init_tech)
ga_agents = copy.deepcopy(init_agents)
market = copy.deepcopy(init_market)
market.change_conditionstring(39)
agents = tech_agents + ga_agents

gooz_specialist = Specialist(max_stockprice, min_stockprice, min_excess, eta, specialist_iterations, theta, taup)
wallstreet = WallStreet(agents, learning_interval, bit_mutation_rate, crossover_rate, rr_ratio)
steps = 150000

if __name__ == '__main__':
    agent_params = wallstreet.trade(steps, market, gooz_specialist, save_specificfracs=True,
                                    save_profits=True, save_strength=True)
