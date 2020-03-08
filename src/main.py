from lib import *
from data_fetcher import *
from agents import *
from emulator import *
from simulators import *
from visualizer import *


def get_model(model_type, env, learning_rate, path):

	print_t = False

	if model_type == 'MLP':
		m = 16
		layers = 5
		hidden_size = [m]*layers
		model = QModelMLP(env.state_shape, env.n_action)
		model.build_model(hidden_size, learning_rate=learning_rate, activation='tanh')

	elif model_type == 'conv':

		m = 16
		layers = 2
		filter_num = [m]*layers
		filter_size = [3] * len(filter_num)
		#use_pool = [False, True, False, True]
		#use_pool = [False, False, True, False, False, True]
		use_pool = None
		#dilation = [1,2,4,8]
		dilation = None
		dense_units = [48,24]
		model = QModelConv(env.state_shape, env.n_action)
		model.build_model(filter_num, filter_size, dense_units, learning_rate,
			dilation=dilation, use_pool=use_pool)

	elif model_type == 'RNN':

		m = 32
		layers = 3
		hidden_size = [m]*layers
		dense_units = [m,m]
		model = QModelGRU(env.state_shape, env.n_action)
		model.build_model(hidden_size, dense_units, learning_rate=learning_rate)
		print_t = True

	elif model_type == 'ConvRNN':
	
		m = 8
		conv_n_hidden = [m,m]
		RNN_n_hidden = [m,m]
		dense_units = [m,m]
		model = QModelConvGRU(env.state_shape, env.n_action)
		model.build_model(conv_n_hidden, RNN_n_hidden, dense_units, learning_rate=learning_rate)
		print_t = True

	elif model_type == 'pretrained':
		model = load_model(path, learning_rate)

	else:
		raise ValueError
		
	return model, print_t


def main():

	# --- Main Parameters ---

	# Data parameters
	source_data = [('BTCUSD', 'gemini_BTCUSD_1hr.csv')]

	n_episode_training = 10
	n_episode_testing = 1
	batch_size = 8
	training_size = 0.95
	# validation_size is implied from training and testing sizes if applicable
	testing_size = 0.

	# Experiment parameters
	experiment_type = 'one_asset_one_agent'
	n_vars = 1  # 1->close prices only, 2-> close+ohl+volume
	lookup_window = 40
	episode_duration = 180
	rolling = False

	# Model parameters
	model_type = 'MLP'
	exploration_init = 1.
	t_cost = 3.3
	learning_rate = 1e-4
	discount_factor = 0.8
	exploration_decay = 0.99
	exploration_min = 0.01

	# --- Data Pipeline ---
	path = os.path.join('..', 'data', experiment_type)
	# check if DB has already been generated, if not create it
	db = Generator()
	db_exists = db._db_exists(path)
	if db_exists:
		db.load_db(db_path=path,
				   source_data=source_data)
	else:
		db.build_from_csv(db_path=path,
						  source_data=source_data)
	data_pipe = DataFetcher(source_db=db,
							   experiment_type=experiment_type,
							   lookup_window=lookup_window,
							   episode_duration=episode_duration,
							   rolling = rolling,
							   data_path=path,
							   n_vars=n_vars,
							   perc_training=training_size,
							   perc_testing=testing_size)
	data_pipe.prep_samples(phase='training')

	# --- Environment ---
	env = Market(data_pipe, lookup_window, t_cost)
	model, print_t = get_model(model_type, env, learning_rate, fld_load)
	model.model.summary()

	# --- Agent ---
	agent = Agent(model, discount_factor=discount_factor, batch_size=batch_size)
	visualizer = Visualizer(env.action_labels)

	fld_save = os.path.join(OUTPUT_FLD, sampler.title, model.model_name,
							str((env.lookup_window, sampler.episode_window, agent.batch_size, learning_rate,
			 agent.discount_factor, exploration_decay, env.t_cost)))
	
	print('='*20)
	print(fld_save)
	print('='*20)

	# --- Experiment - training ---
	simulator = Simulator(agent, env, visualizer=visualizer, fld_save=fld_save)
	simulator.train(n_episode_training, save_per_episode=1, exploration_decay=exploration_decay, 
		exploration_min=exploration_min, print_t=print_t, exploration_init=exploration_init)
	agent.model = load_model(os.path.join(fld_save, 'model'), learning_rate)

	# --- Experiment - validation ---
	print('='*20+'\nin-sample testing\n'+'='*20)
	simulator.test(n_episode_testing, save_per_episode=1, subfld='in-sample testing')

	# --- Experiment - testing ---
	fld = os.path.join('data', asset_type, db+'B')
	sampler = DataFetcher('load', data_path=fld)
	simulator.env.sampler = sampler
	simulator.test(n_episode_testing, save_per_episode=1, subfld='out-of-sample testing')

	

if __name__ == '__main__':
	main()


# TODO: update Environment, Agent and test simple training experiment
# TODO: