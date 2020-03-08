import random, os, datetime, pickle, json, keras, sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FLD = os.path.join('..', 'results')
data_location = '../data'

def create_folder(path):
	if not os.path.exists(path):
		os.makedirs(path)


class Generator:

	def __init__(self):
		self.db = {}
		return

	def load_db(self, db_path, source_data):
		for source in source_data:
			asset, name = source
			path = os.path.join(db_path, asset)
			print(f'--- Loading DB from {path}')
			self.db = pickle.load(open(os.path.join(path,'db.pickle'), 'rb'))
			# self.n_db = param['n_episodes']
			# self.sample = self.sample_from_db
			# for attr in param:
			# 	if hasattr(self, attr):
			# 		setattr(self, attr, param[attr])
			# self.title = 'DB_'+param['title']

	def save_db(self, data, path):
		# data -> dataframe, path -> location & name
		pickle.dump(data, open(path, 'wb'))

	# def build_db(self, n_episodes, data_type, path_source):
	# 	db = []
	# 	db_source = pd.read_csv(path_source)
	# 	for i in range(n_episodes):
	# 		prices, title = self.sample()
	# 		db.append((prices, '[%i]_'%i+title))
	# 	os.makedirs(data_type)	# don't overwrite existing fld
	# 	pickle.dump(db, open(os.path.join(data_type, 'db.pickle'), 'wb'))
	# 	param = {'n_episodes':n_episodes}
	# 	for k in self.attrs:
	# 		param[k] = getattr(self, k)
	# 	json.dump(param, open(os.path.join(data_type, 'param.json'), 'w'))

	def build_from_csv(self, db_path, source_data, db_name='db', csv_path='../data/raw'):
		# Read the data from the provided CSV files and create a pickle in the experiment folder
		for source in source_data:
			asset, name = source
			print(f'--- Creating a database for {asset} in {db_path} ...')
			output_path = os.path.join(db_path, asset)
			create_folder(output_path)
			input_path = os.path.join(csv_path, name)
			csv_data = pd.read_csv(input_path, index_col=1, header=1)
			db_path = os.path.join(output_path, f"{db_name}.pickle")
			pickle.dump(csv_data, open(db_path, 'wb'))
			self.db = csv_data

	def sample_from_db(self):
		prices, title = self.db[self.i_db]
		self.i_db += 1
		if self.i_db == self.n_db:
			self.i_db = 0
		return prices, title

	@staticmethod
	def _db_exists(path):
		if not os.path.exists(path):
			return False
		else:
			return True


class DataFetcher(Generator):
	def __init__(self, source_db, experiment_type, lookup_window=None, episode_duration=None, rolling=False,
				 data_path=None, n_vars=1, perc_training=0.95, perc_testing=0.05):
		self.n_vars = n_vars	# 1 -> close price only, 2 -> close + ohl + volume
		self.experiment_type = experiment_type
		self.lookup_window = lookup_window
		self.episode_duration = episode_duration
		self.rolling = rolling
		self.phase = None
		self.episode_rand_order = None
		self.curr_episode_id = None
		self.attrs = ['experiment_type', 'lookup_window', 'episode_duration', 'n_vars', 'rolling']
		# save parameters in a json for easy reference in the future
		param_str = str((self.experiment_type, self.lookup_window, self.episode_duration, self.n_vars, self.rolling))
		self.params = {}
		for k in self.attrs:
			self.params[k] = getattr(self, k)
		# store it in a folder dedicated to the current experiment
		self.path = os.path.join(data_path, param_str)
		create_folder(self.path)
		json.dump(self.params, open(os.path.join(self.path, 'param.json'), 'w'))
		# transform the data in /data/asset and store in /data/experiment
		if experiment_type == 'one_asset_one_agent':
			self.split_into_phases(db=source_db.db,
								   perc_training=perc_training,
								   perc_testing=perc_testing)
		elif experiment_type == 'load':
			pass
		else:
			raise NotImplementedError(f'{experiment_type} - experiment type has not been implemented yet.')

	def split_into_phases(self, db, perc_training, perc_testing):
		# split db into 3 phases according to provided percentiles
		# reserve last portion of data for the test set
		n = len(db)
		if self.n_vars==1:
			data_panel = ['Close']
		elif self.n_vars==2:
			data_panel = ['Close', 'Open', 'High', 'Low', 'Volume']
		test_start = round(n * (1 - perc_testing))
		test_data = db.loc[db.index[test_start:], data_panel]
		# split into episodes: episode consist of lookback window + episode over which game will be played
		# testing set
		if not self._db_exists(os.path.join(self.path, 'testing_db.pickle')):
			print(f'--- Creating testing_db in {self.path} ...')
			testing_db = {}
			num_ep_testing = round(len(test_data)/(self.lookup_window+self.episode_duration))
			for i in range(num_ep_testing):
				testing_db[i] = test_data.iloc[i * self.episode_duration:
											   (i + 1) * self.episode_duration + self.lookup_window]
			self.save_db(data=testing_db,
						 path=os.path.join(self.path,'testing_db.pickle'))
		# for training set - split episodes depending on self.rolling feature
		if not self._db_exists(os.path.join(self.path, 'training_db.pickle')):
			print(f'--- Creating training_db in {self.path} ...')
			training_db = {}
			train_end = round(n * perc_training)
			training_data = db.loc[db.index[:train_end], data_panel]
			if self.rolling:
				num_ep_training = len(training_data) - (self.lookup_window + self.episode_duration) + 1
				for i in range(num_ep_training):
					training_db[i] = training_data.iloc[i:i + self.episode_duration + self.lookup_window]
			else:
				num_ep_training = round(len(training_data) / (self.lookup_window + self.episode_duration))
				for i in range(num_ep_training):
					training_db[i] = training_data.iloc[i * self.episode_duration:
														(i+1) * self.episode_duration + self.lookup_window]
			self.save_db(data=training_db,
						 path=os.path.join(self.path,'training_db.pickle'))
		# check if validation set is implied
		if perc_training + perc_testing < 1:
			if not self._db_exists(os.path.join(self.path, 'validation_db.pickle')):
				print(f'--- Creating validation_db in {self.path} ...')
				validation_db = {}
				validation_start = train_end + 1
				validation_end = test_start - 1
				validation_data = db.loc[db.index[validation_start:validation_end], data_panel]
				num_ep_validation = round(len(validation_data)/(self.lookup_window + self.episode_duration))
				for i in range(num_ep_validation):
					validation_db[i] = validation_data.iloc[i * self.episode_duration:
														(i+1) * self.episode_duration + self.lookup_window]
				self.save_db(data=validation_db,
							 path=os.path.join(self.path,'validation_db.pickle'))
		return

	def prep_samples(self, phase='training'):
		# pre-load samples required for current phase
		self.phase = phase
		self.db = pickle.load(open(os.path.join(self.path, phase+'_db.pickle'), 'rb'))
		random.seed = 1
		self.episode_rand_order = np.arange(len(self.db.keys()))
		random.shuffle(self.episode_rand_order)
		self.curr_episode_id = -1
		return

	def sample(self, randomly=True):
		# fetch a sample form the prepared db for the current phase
		self.curr_episode_id += 1
		if self.phase == 'training':
			if randomly:
				data = self.db[self.episode_rand_order[self.curr_episode_id]]
			else:
				data = self.db[self.curr_episode_id]
		else:
			data = self.db[self.curr_episode_id]
		return data



	# def _rand_sin(self, period_range=None, amplitude_range=None, noise_amplitude_ratio=None, full_episode=False):
	# 	if period_range is None:
	# 		period_range = self.period_range
	# 	if amplitude_range is None:
	# 		amplitude_range = self.amplitude_range
	# 	if noise_amplitude_ratio is None:
	# 		noise_amplitude_ratio = self.noise_amplitude_ratio
	#
	# 	period = random.randrange(period_range[0], period_range[1])
	# 	amplitude = random.randrange(amplitude_range[0], amplitude_range[1])
	# 	noise = noise_amplitude_ratio * amplitude
	#
	# 	if full_episode:
	# 		length = self.lookup_window
	# 	else:
	# 		if self.can_half_period:
	# 			length = int(random.randrange(1,4) * 0.5 * period)
	# 		else:
	# 			length = period
	#
	# 	p = 100. + amplitude * np.sin(np.array(range(length)) * 2 * 3.1416 / period)
	# 	p += np.random.random(p.shape) * noise
	#
	# 	return p, '100+%isin((2pi/%i)t)+%ie'%(amplitude, period, noise)

	#
	# def __sample_concat_sin(self):
	# 	prices = []
	# 	p = []
	# 	while True:
	# 		p = np.append(p, self.__rand_sin(full_episode=False)[0])
	# 		if len(p) > self.episode_window:
	# 			break
	# 	prices.append(p[:self.episode_window])
	# 	return np.array(prices).T, 'concat sin'
	#
	# def __sample_concat_sin_w_base(self):
	# 	prices = []
	# 	p = []
	# 	while True:
	# 		p = np.append(p, self.__rand_sin(full_episode=False)[0])
	# 		if len(p) > self.episode_window:
	# 			break
	# 	base, base_title = self.__rand_sin(
	# 		period_range=self.base_period_range,
	# 		amplitude_range=self.base_amplitude_range,
	# 		noise_amplitude_ratio=0.,
	# 		full_episode=True)
	# 	prices.append(p[:self.episode_window] + base)
	# 	return np.array(prices).T, 'concat sin + base: '+base_title
	#
	# def _sample_single_sin(self):
	# 	prices = []
	# 	funcs = []
	# 	p, func = self._rand_sin(full_episode=True)
	# 	prices.append(p)
	# 	funcs.append(func)
	# 	return np.array(prices).T, str(funcs)


if __name__ == '__main__':
	g = Generator()
	path = '../data/test'
	source_data = [('BTCUSD', 'gemini_BTCUSD_1hr.csv')]
	g.build_from_csv(db_path=path,
					 source_data=source_data)