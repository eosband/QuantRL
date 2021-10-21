from agent.memory import Transition, ReplayMemory
from agent.model import DQN

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

# Redirect pytorch processing power
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from agent.memory import Transition, ReplayMemory
from agent.model import DQN

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import os

# Redirect pytorch processing power
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
	"""
	Represents the Agent that plays the reinforcement learning game
	"""
	def __init__(self, state_size, config, is_eval=False):
		"""
		Constructs new Agent object
		:param state_size: size of the state being used
		:param config: config file
		:param is_eval: flag for whether or not we are training (i.e. update parameters) or evaluating
		"""
		# Represents the size of each state
		self.state_size = state_size
		# Represents size of the action space
		self.action_size = 3 # 3 options of actions: hold, buy, sell
		self.memory = ReplayMemory(10000) # Constructs a new memory object
		# self.inventory = []
		self.is_eval = is_eval

		# RL parameters: kept the same from original product
		self.gamma = config['gamma'] # gamma coefficient from Belman equation
		# Epsilon will keep being changed at each iteration
		self.epsilon = config['epsilon']
		self.epsilon_min = config['epsilon_min']  # minimum value epsilon can take
		self.epsilon_decay = config['epsilon_decay']  # amount epsilon decays each iteration
		self.batch_size = config['batch_size']

		# Loads previous models, if they exist
		if os.path.exists(config['target_model']):
			self.policy_net = torch.load(config['policy_model'], map_location=device)
			self.target_net = torch.load(config['target_model'], map_location=device)
		else:
			self.policy_net = DQN(state_size, self.action_size)
			self.target_net = DQN(state_size, self.action_size)
		# Optimization function
		self.optimizer = optim.RMSprop(self.policy_net.parameters(),
									   lr=config['learning_rate'], momentum=config['momentum'])

	def act(self, state):
		"""
		Acts on the current state
		"""
		if not self.is_eval and np.random.rand() <= self.epsilon:
			# Then we are not evaulating, and we are in exploratory phase, so we try something new
			return random.randrange(self.action_size)
		# Otherwise, we convert the state to a tensor, and run it through our target network to determine
		# action of highest probability, which we return.
		tensor = torch.FloatTensor(state).to(device)
		options = self.target_net(tensor)
		return np.argmax(options[0].detach().numpy())

	def decay_epsilon(self):
		"""
		Decays epsilon value according to parameters
		"""
		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.epsilon_decay

	def optimize(self):
		"""
		Optimizes the policy and target nets
		"""
		if len(self.memory) < self.batch_size:
			return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(batch.next_state).to(device)
		# Masks help keep array shape, even in case that we run over the boundary of array size
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.cat([s for s in next_state if s is not None])
		state_batch = torch.FloatTensor(batch.state).to(device)
		# Actions from all elements of the batch - each one 0,1,2
		action_batch = torch.LongTensor(batch.action).to(device)
		# Rewards from the actions corresponding to all elements of the batch
		reward_batch = torch.FloatTensor(batch.reward).to(device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		q_pred = self.policy_net(state_batch).reshape((self.batch_size, 3)).gather(1, action_batch.reshape((self.batch_size, 1)))

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		v_actual = torch.zeros(self.batch_size, device=device)
		# Fills in the predicted state values for each timestamp
		v_actual[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		# Compute what should have been the Q values
		q_target = (v_actual * self.gamma) + reward_batch

		# Compute Huber loss
		loss = F.smooth_l1_loss(q_pred, q_target.unsqueeze(1))

		# Decay Epsilon
		self.decay_epsilon()

		# Optimize the model - standard pytorch procedure
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)  # Keep gradient from going crazy
		self.optimizer.step()
class Agent:
	"""
	Represents the Agent that plays the reinforcement learning game
	"""
	def __init__(self, state_size, config, is_eval=False):
		"""
		Constructs new Agent object
		:param state_size: size of the state being used
		:param config: config file
		:param is_eval: flag for whether or not we are training (i.e. update parameters) or evaluating
		"""
		# Represents the size of each state
		self.state_size = state_size
		# Represents size of the action space
		self.action_size = 3 # 3 options of actions: hold, buy, sell
		self.memory = ReplayMemory(10000) # Constructs a new memory object
		# self.inventory = []
		self.is_eval = is_eval

		# RL parameters: kept the same from original product
		self.gamma = config['gamma'] # gamma coefficient from Belman equation
		# Epsilon will keep being changed at each iteration
		self.epsilon = config['epsilon']
		self.epsilon_min = config['epsilon_min']  # minimum value epsilon can take
		self.epsilon_decay = config['epsilon_decay']  # amount epsilon decays each iteration
		self.batch_size = config['batch_size']

		# Loads previous models, if they exist
		if os.path.exists(config['target_model']):
			self.policy_net = torch.load(config['policy_model'], map_location=device)
			self.target_net = torch.load(config['target_model'], map_location=device)
		else:
			self.policy_net = DQN(state_size, self.action_size)
			self.target_net = DQN(state_size, self.action_size)
		# Optimization function
		self.optimizer = optim.RMSprop(self.policy_net.parameters(),
									   lr=config['learning_rate'], momentum=config['momentum'])

	def act(self, state):
		"""
		Acts on the current state
		"""
		if not self.is_eval and np.random.rand() <= self.epsilon:
			# Then we are not evaulating, and we are in exploratory phase, so we try something new
			return random.randrange(self.action_size)
		# Otherwise, we convert the state to a tensor, and run it through our target network to determine
		# action of highest probability, which we return.
		tensor = torch.FloatTensor(state).to(device)
		options = self.target_net(tensor)
		return np.argmax(options[0].detach().numpy())

	def decay_epsilon(self):
		"""
		Decays epsilon value according to parameters
		"""
		if self.epsilon > self.epsilon_min:
			self.epsilon -= self.epsilon_decay

	def optimize(self):
		"""
		Optimizes the policy and target nets
		"""
		if len(self.memory) < self.batch_size:
			return
		transitions = self.memory.sample(self.batch_size)
		# Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
		# detailed explanation). This converts batch-array of Transitions
		# to Transition of batch-arrays.
		batch = Transition(*zip(*transitions))

		# Compute a mask of non-final states and concatenate the batch elements
		# (a final state would've been the one after which simulation ended)
		next_state = torch.FloatTensor(batch.next_state).to(device)
		# Masks help keep array shape, even in case that we run over the boundary of array size
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)))
		non_final_next_states = torch.cat([s for s in next_state if s is not None])
		state_batch = torch.FloatTensor(batch.state).to(device)
		# Actions from all elements of the batch - each one 0,1,2
		action_batch = torch.LongTensor(batch.action).to(device)
		# Rewards from the actions corresponding to all elements of the batch
		reward_batch = torch.FloatTensor(batch.reward).to(device)

		# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
		# columns of actions taken. These are the actions which would've been taken
		# for each batch state according to policy_net
		q_pred = self.policy_net(state_batch).reshape((self.batch_size, 3)).gather(1, action_batch.reshape((self.batch_size, 1)))

		# Compute V(s_{t+1}) for all next states.
		# Expected values of actions for non_final_next_states are computed based
		# on the "older" target_net; selecting their best reward with max(1)[0].
		# This is merged based on the mask, such that we'll have either the expected
		# state value or 0 in case the state was final.
		v_actual = torch.zeros(self.batch_size, device=device)
		# Fills in the predicted state values for each timestamp
		v_actual[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
		# Compute what should have been the Q values
		q_target = (v_actual * self.gamma) + reward_batch

		# Compute Huber loss
		loss = F.smooth_l1_loss(q_pred, q_target.unsqueeze(1))

		# Decay Epsilon
		self.decay_epsilon()

		# Optimize the model - standard pytorch procedure
		self.optimizer.zero_grad()
		loss.backward()
		for param in self.policy_net.parameters():
			param.grad.data.clamp_(-1, 1)  # Keep gradient from going crazy
		self.optimizer.step()