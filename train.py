from agent.agent import Agent
from envs import TradingEnv
from functions import *
import torch
import yaml, os


#############################
# Trains the model from CLI #
#############################

# PYTHONWARNINGS=ignore::yaml.YAMLLoadWarning

def train():
    profits_list = [] # Will hold list of all profits as we go through training

    # Given command line input as below

    # if len(sys.argv) != 4:
    #     print("Usage: python train.py [stock] [window] [episodes]")
    #     exit()

    with open(os.path.join(os.path.dirname(__file__), 'config.yml'), 'r') as stream:
        config = yaml.load(stream)

    # Unpackage data from terminal/config
    # stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    stock_name, window_size, episode_count = config['stock_name'], config['window_size'], config["num_epochs"]

    num_tech_indicators = config['num_tech_indicators']
    agent = Agent(window_size + num_tech_indicators, config)
    data = getStockDataVec(stock_name)
    env = TradingEnv(data, window_size)
    l = len(data) - 1

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = env.get_state(0)

        env.reset_holdings()

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = env.get_state(t + 1)
            reward = 0

            if action == 1: # buy
                #remembers the price bought at t, and the time bought
                env.buy(t)
                # print("Buy: " + formatPrice(data[t]))

            elif action == 2: # sell
                reward, profit = env.sell(t)
                # print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(profit))

            done = True if t == l - 1 else False
            # Push all values to memory
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            total_profit = env.net_profit(t)
            max_staked = env.max_spent

            if done:
                percent_return = total_profit / max_staked * 100
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("Max staked: " + formatPrice(max_staked))
                print("Percent return: " + "{0:.2f}%".format(percent_return))
                print("--------------------------------")
                profits_list.append((total_profit, percent_return))
                # print(profits_list)
            agent.optimize()

        if e % config['save_freq'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            torch.save(agent.policy_net, config['policy_model'])
            torch.save(agent.target_net, config['target_model'])


if __name__ == "__main__":
    train()