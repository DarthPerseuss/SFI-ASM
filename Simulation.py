from Stocks import Stock
import matplotlib.pyplot as plt
from Agents import Agent
from Specialist import Specialist

num_shares = 100
init_price = 80

dividend_startvalue = 10
rho = 1
noise_sd = 0.5
d_bar = 5
interest_rate = 0.02
init_holding = 1
initialcash = 20000
theta = 75
gene_length = 64
risk_coef = 0.5
num_strategies = 80

max_stockprice = 200
min_stockprice = 0.01
min_excess = 0.005
eta = 0.005
specialist_iterations = 10

the_market = Stock(num_shares, init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate)
agents = [Agent(init_holding, initialcash, num_strategies,
                theta, gene_length, risk_coef, num_shares, init_price,
                dividend_startvalue, rho, noise_sd, d_bar, interest_rate) for _ in range(100)]
the_specialist = Specialist(max_stockprice, min_stockprice, min_excess, eta, specialist_iterations, num_shares,
                             init_price, dividend_startvalue, rho, noise_sd, d_bar, interest_rate)

for t in range(1000):
    the_market.advance_arprocess()
    the_specialist.clear_market(agents)

ma = the_market.calculate_ma(query='dividend', period=50)
