from functions import *

class TradingEnv():
    """
    Represents the trading environment used in our model.
    Handles low-level data scraping, retrieval, and calculation
    Adjustable parameters:
        get_reward(params): the reward function of a certain action
        get_state(params): the state that the model is currently in
    """
    def __init__(self, train_data, window_size):
        '''
        Creates a trading environment from data train_data with window size window_size
        :param train_data: data to be trained on, e.g. daily closing prices
        :param window_size: size of the window on which we examine stock trends
        '''
        # List of all daily closing prices
        self.data = train_data
        # List of Simple Moving Averages from the window
        self.sma_data = getSMAFromVec(train_data, window_size)
        # Size of recent closing price list
        self.window_size = window_size
        # Keeps track of buying prices
        self.inventory = []
        # Keeps track of how much spent
        self.max_spent = 0
        self.current_out = 0  # currently held assets
        self.buys = []
        self.sells = []
        self.total_profit = 0

    def get_reward(self, selling_price, time_sold, bought_price, time_bought):
        """
        Gets the reward of the given action
        :param selling_price: price sold
        :param time_sold: time sold
        :param bought_price: buying price
        :param time_bought: time bought
        :return:
        """
        delta_t = time_sold - time_bought
        profit = selling_price - bought_price
        return max(profit, .0001)
        # reward = max(profit, .0001) // (np.log(delta_t) + 1)
        # return reward

    def get_weighted_diff(self, v1, v2):
        return (abs(v2 - v1)) / v1

    def get_state(self, t):
        '''
        Our state representation.
        :param t: time
        :return: n-day state representation ending at time t with sma indicator at end
        '''
        n = self.window_size + 1
        d = t - n + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1] # pad with t0
        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))

        # add sigmoid of price and sma
        # res.append(sigmoid(self.get_weighted_diff(self.data[t], self.sma_data[t])))
        res = np.array([res])
        return res

    def reset_holdings(self):
        """
        Resets the inventory and amount bought
        """
        self.inventory = []
        self.current_out = 0
        self.max_spent = 0
        self.buys = []
        self.sells = []
        self.total_profit = 0

    def buy(self, t):
        """
        Buys stock at time t
        :param t: time to buy
        """
        price = self.data[t]
        self.inventory.append((price, t))
        self.current_out += price
        self.max_spent = max(self.max_spent, self.current_out)
        self.buys.append(t)

    def sell(self, t):
        """
        Sells the oldest stock in portfolio
        :param t: time at which to sell
        :return: reward and profit from selling
        """
        if len(self.inventory) < 1:
            return 0, 0
        bought_price, time_bought = self.inventory.pop(0)
        selling_price = self.data[t]
        reward = self.get_reward(selling_price, t, bought_price, time_bought)
        profit = selling_price - bought_price
        self.total_profit += profit
        self.current_out -= selling_price
        self.sells.append(t)
        return reward, profit

    def value_held(self, t):
        """
        Returns the total value of the portfolio at time t
        :param t: time
        """
        return len(self.inventory) * self.data[t]

    def net_profit(self, t):
        """
        Returns the total profit of the environment, which represents
        the net profit made on each transaction plus the value of all
        current assets at time t
        :param t: current time (so as to determine market price of stock)
        """
        return self.total_profit + self.value_held(t)
