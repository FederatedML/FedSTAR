##########################################
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912
##########################################
import argparse 
import pathlib
import uuid
import gc
import time
import flwr
import logging
import tensorflow as tf
from model import Network
from data import DataBuilder

class AudioServer:

	def __init__(self, flwr_evalution_step, flwr_min_sample_size, flwr_min_num_clients, flwr_rounds,
		model_num_classes, model_lr, model_batch_size, model_epochs, model_ds_test, model_verbose):
		# Flower Parameters
		self.evalution_step = flwr_evalution_step
		self.sample_fraction = float(flwr_min_sample_size/flwr_min_num_clients)
		self.min_sample_size = flwr_min_sample_size
		self.min_num_clients = flwr_min_num_clients
		self.rounds = flwr_rounds
		# Model Parameters
		self.num_classes = model_num_classes
		self.lr = model_lr
		self.batch_size = model_batch_size
		self.epochs = model_epochs
		self.verbose = model_verbose
		self.ds_test = model_ds_test
		# Local Variables Counters and Variables
		self.current_round = 0
		self.final_accuracy = 0.0
		self.round_time = time.time()
		self.strategy = flwr.server.strategy.FedAvg(fraction_fit=self.sample_fraction, min_fit_clients=self.min_sample_size, 
			min_available_clients=self.min_num_clients, on_fit_config_fn=self.get_on_fit_config_fn(), fraction_eval=0,
			min_eval_clients=0, on_evaluate_config_fn=None, eval_fn=self.get_eval_fn(ds_test=self.ds_test), accept_failures=True)
		self.client_manager = flwr.server.client_manager.SimpleClientManager()
		tf.keras.backend.clear_session()

	def server_start(self, server_address):
		flwr.server.start_server(
			server_address=server_address, server = flwr.server.Server(client_manager=self.client_manager, strategy=self.strategy),
			config={"num_rounds": self.rounds}, strategy=self.strategy, grpc_max_message_length= GRPC_MAX_MESSAGE_LENGTH)

	def get_on_fit_config_fn(self):
		def fit_config(rounds=self.rounds, epochs=self.epochs, batch_size=self.batch_size, learning_rate=self.lr):
			if self.current_round!=1:
				print('\nTraining round completed in '+str(round(time.time()-self.round_time,2))+ ' seconds.')
			print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'+
				  '               Server started '+str(self.current_round)+'th round of training.\n'+
				  '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
			# Update round start time
			self.round_time = time.time()
			return {"rounds": str(self.rounds), "c_round": str(self.current_round), "epochs": str(epochs), "batch_size": str(batch_size), "learning_rate": str(learning_rate),}
		return fit_config

	def get_eval_fn(self, ds_test):
		import tensorflow as tf
		def evaluate(weights):
			loss, acc = 0,0
			self.current_round += 1
			if ((self.current_round-1) % self.evalution_step == 0):
				model = Network(num_classes=self.num_classes).get_evaluation_network()
				model.compile(optimizer = tf.keras.optimizers.Adam(self.lr), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss'), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
				model.set_weights(weights)
				loss, acc = model.evaluate(ds_test, verbose=self.verbose)
				self.final_accuracy = acc
				# Clear Memory
				clear_memory()
			return float(loss), {"accuracy": float(acc)}
		return evaluate

	def get_accuracy(self):
		return self.final_accuracy

def clear_memory():
	import tensorflow as tf
	gc.collect()
	tf.keras.backend.clear_session()

def set_logger_level():	
	if 'flower' in [logging.getLogger(name).__repr__()[8:].split(' ')[0] for name in logging.root.manager.loggerDict]:
		logger = logging.getLogger('flower')
		logger.setLevel(logging.INFO)

def set_gpu_limits(gpu_id, gpu_memory):
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
	import tensorflow as tf
	gpus = tf.config.list_physical_devices("GPU")
	if not gpus:
		print("No GPU's available. Server will run on CPU.")
	else:
		try:
			tf.config.experimental.set_virtual_device_configuration(gpus[0], 
				[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
		except RuntimeError as e:
			print(e)

def main(): 

	parser = argparse.ArgumentParser(description="Flower Federated Learning - Server")
	parser.add_argument("--server_address", type=str,   default="[::]:10001",           required=False,  help=f"gRPC server address (default: [::]:8080)")
	parser.add_argument("--rounds",         type=int,   default=100,                    required=False,  help="Number of rounds of federated learning (default: 3)")
	parser.add_argument("--batch_size",     type=int,   default=64,                     required=False,  help="Training Batch Size (no default)")
	parser.add_argument("--dataset_dir",    type=str,   default="../dataset",           required=False,  help="Relevant path to dataset (no default)")
	parser.add_argument("--learning_rate",  type=float, default=0.001,                  required=False,  help="Model Learning Rate (no default)")
	parser.add_argument("--num_clients",    type=int,   default=1,                      required=False,  help="Minimum number of available clients required for sampling (default: 1)")
	parser.add_argument("--min_sample_size",type=int,   default=1,                      required=False,  help="Minimum number of clients used for fit/evaluate (default: 1)")
	parser.add_argument("--train_epochs",   type=int,   default=1,                      required=False,  help="Clients train epochs (no default)")
	parser.add_argument("--seed",           type=int,   default=2021,                   required=False,  help="Seed for clients data shuffling (default: 2021)")
	parser.add_argument("--gpu_memory",     type=int,   default=384,                    required=False,  help="GPU Available Memory (no default)")
	parser.add_argument("--eval_step",      type=int,   default=1,                      required=False,  help="Evaluation Perform Step")
	parser.add_argument("--verbose",        type=int,   default=1,                      required=False,  help="Evaluation Detail to be displayed")
	args = parser.parse_args()

	# Set Experiment Parameters
	unique_id =  str(uuid.uuid1())
	dataset_dir = os.path.join(pathlib.Path.home(),args.dataset_dir)
	# Notify Experiment ID to console.
	print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n'+
		  ' Experiment ID : '+unique_id+'\n'+
		  '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
	# GPU Setup
	set_gpu_limits(gpu_id='0', gpu_memory=args.gpu_memory)
	# Load Test Dataset
	ds_test, num_classes = DataBuilder.get_ds_test(data_dir=dataset_dir, batch_size=args.batch_size, buffer=1024, seed=args.seed)
	# Create Server Object
	audio_server = AudioServer(flwr_evalution_step=args.eval_step, flwr_min_sample_size=args.min_sample_size, 
		flwr_min_num_clients=args.num_clients, flwr_rounds=args.rounds, model_num_classes=num_classes, 
		model_lr=args.learning_rate, model_batch_size=args.batch_size, model_epochs=args.train_epochs, 
		model_ds_test=ds_test, model_verbose=args.verbose,)
	# Run server
	set_logger_level()
	audio_server.server_start(args.server_address[1:-1])
	return f"\nFinal Accuracy on experiment  {unique_id}: {audio_server.get_accuracy():.04f}\n"

if __name__ == "__main__":
    main()
    sys.exit(0)
