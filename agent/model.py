import torch.nn as nn

class DQN(nn.Module):
	'''
	Represents the DeepQ Neural Network used by our agent for predicting policy and target values.
	'''
	def __init__(self, state_size, action_size):
		"""
		Simple initialization of a PyTorch neural network.
		Constructs a network to take in [state_size] inputs and output [action_size] values
		:param state_size: number of states (input)
		:param action_size: number of actions (output)
		"""
		super(DQN, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(state_size, 64),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(64, 32),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(32, 8),
			nn.LeakyReLU(0.01, inplace=True),
			nn.Linear(8, action_size),
		)
	
	def forward(self, input):
		return self.main(input)
