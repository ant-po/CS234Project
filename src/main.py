from lib import *
from data_fetcher import *
from agents import *
from emulator import *
from simulators import *
from visualizer import *
from models import *


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

	env = Market(data_pipe=data_pipe,
				 lookup_window=lookup_window,
				 t_cost=t_cost)
	m = Models()
	model, print_t = m.get_model(model_type, env, learning_rate)
	model.model.summary()

	# --- Agent ---

	agent = Agent(model=model,
				  batch_size=batch_size,
				  discount_factor=discount_factor)

	# --- Visualizer ---

	visualizer = Visualizer(env.action_labels)

	# --- Output ---

	results_path = os.path.join(OUTPUT_FLD, data_pipe.experiment_type, model.model_name,
							str((env.lookup_window, data_pipe.episode_duration, agent.batch_size, learning_rate,
			 agent.discount_factor, exploration_decay, env.t_cost)))
	
	print('='*20)
	print(f'Results will be saved in: \n{results_path}')
	print('='*20)

	# --- Experiment ----

	# --> Training

	simulator = Simulator(agent=agent,
						  env=env,
						  visualizer=visualizer,
						  results_path=results_path)
	simulator.train(n_episode=n_episode_training,
					save_per_episode=1,
					exploration_decay=exploration_decay,
					exploration_min=exploration_min,
					print_t=print_t,
					exploration_init=exploration_init)
	agent.model = load_model(path=os.path.join(results_path, 'model'),
							 learning_rate=learning_rate)

	# ---> Validation (in-sample)

	print('='*20+'\nin-sample testing\n'+'='*20)
	simulator.test(n_episode=n_episode_testing,
				   save_per_episode=1,
				   subfld='in-sample testing')

	# --> Testing (out-of-sample)

	fld = os.path.join('data', asset_type, db+'B')
	sampler = DataFetcher('load', data_path=fld)
	simulator.env.sampler = sampler
	simulator.test(n_episode_testing, save_per_episode=1, subfld='out-of-sample testing')

	

if __name__ == '__main__':
	main()


# TODO: investigate simulator/play_one_episode
# TODO: update Environment, Agent and test simple training experiment
# TODO: in emulator/market class review meaning of direction and risk_averse parameters

