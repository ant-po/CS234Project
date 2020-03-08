import random, os, datetime, pickle, json, keras, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_ideal(p, just_once):
	if not just_once:
		diff = np.array(p[1:]) - np.array(p[:-1])
		return sum(np.maximum(np.zeros(diff.shape), diff))
	else:
		best = 0.
		for i in range(len(p)-1):
			best = max(best, max(p[i+1:]) - p[i])
		return best


class Market:
	"""
	state 			MA of prices, normalized using values at t
					ndarray of shape (window_state, n_instruments * n_MA), i.e., 2D
					which is self.state_shape

	action 			three action
					0:	no position, don't open or do close.
					1:	open a position
					2: 	hold a position
	"""

	def __init__(self, data_pipe, lookup_window, t_cost, direction=1., risk_averse=0.):
		self.no_position = True
		self.data_pipe = data_pipe
		self.lookup_window = lookup_window
		self.t_cost = t_cost
		self.direction = direction
		self.risk_averse = risk_averse

		self.n_action = 3
		self.state_shape = (lookup_window, self.data_pipe.n_vars)
		self.action_labels = ['no position', 'open', 'hold']
		self.t0 = lookup_window - 1

	def reset(self, rand_price=True):
		self.no_position = True
		if rand_price:
			prices, self.title = self.data_pipe.sample()
			price = np.reshape(prices[:, 0], prices.shape[0])

			self.prices = prices.copy()
			self.price = price/price[0]*100
			self.t_max = len(self.price) - 1

		self.max_profit = find_ideal(self.price[self.t0:], False)
		self.t = self.t0
		return self.get_state(), self.get_valid_actions()

	def get_state(self, t=None):
		if t is None:
			t = self.t
		state = self.prices[t - self.lookup_window + 1: t + 1, :].copy()
		for i in range(self.data_pipe.n_vars):
			# TODO: review here
			norm = np.mean(state[:, i])
			state[:, i] = (state[:, i]/norm - 1.)*100	
		return state

	def get_valid_actions(self):
		if self.no_position:
			return [0, 1]	# wait, open
		else:
			return [0, 2]	# close, keep

	def get_noncash_reward(self, t=None, empty=None):
		if t is None:
			t = self.t
		if empty is None:
			empty = self.no_position
		reward = self.direction * (self.price[t+1] - self.price[t])
		if empty:
			reward -= self.t_cost
		if reward < 0:
			reward *= (1. + self.risk_averse)
		return reward

	def step(self, action):
		done = False
		if action == 0:		# wait/close
			reward = 0.
			self.no_position = True
		elif action == 1:	# open
			reward = self.get_noncash_reward()
			self.no_position = False
		elif action == 2:	# keep
			reward = self.get_noncash_reward()
		else:
			raise ValueError('no such action: '+str(action))

		self.t += 1
		return self.get_state(), reward, self.t == self.t_max, self.get_valid_actions()




if __name__ == '__main__':
	test_env()
