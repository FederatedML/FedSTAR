import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
import flwr
import gc
import logging
import tensorflow as tf
from data import DataBuilder
from model import Network, PSL_Network

LOG_LEVEL = logging.ERROR

class AudioClient(flwr.client.NumPyClient):

	def __init__(self, client_id, server_address, num_clients, dataset_dir, fedstar,
			variance=0.25, batch_size=64, l_per=0.2, u_per=1.0, class_distribute=False, mean_class_distribution=3, seed=2021, verbose=2):

		# Client Parameters
		self.server_address = server_address
		self.client_id = client_id
		self.batch_size = batch_size
		self.verbose = verbose
		self.fedstar = fedstar
		self.aux_loss_weight = 0.5
		# Load Clients Data
		self.train_L, self.train_U, self.num_classes, self.num_batches = \
			DataBuilder.load_sharded_dataset(data_dir=dataset_dir, num_clients=num_clients, client=client_id, variance=variance, 
									batch_size=batch_size, l_per=l_per, u_per=u_per, fedstar=fedstar, 
									class_distribute=class_distribute, mean_class_distribution=mean_class_distribution, seed=seed)
		self.num_examples_train = self.num_batches*batch_size if self.train_L else 0
		# Local Variables Initialize
		self.local_train_round = 0
		self.local_evaluate_round = 0
		self.weights = Network.get_init_weights(num_classes=self.num_classes) 
		self.history = {'loss': [], 'accuracy': []}
		tf.keras.backend.clear_session()

	def __call__(self, introduce=False):
		try:
			if introduce: print(f"This is client {self.client_id} with train dataset of {self.num_examples_train} elements.")
			set_logger_level()
			flwr.client.start_numpy_client(self.server_address, self)
			print("Client Shutdown.")
		except RuntimeError as e:
			print(e)
		return 0

	def get_parameters(self):  
		return self.weights

	def fit(self, parameters, config):
		self.local_train_round+=1
		self.weights = parameters
        # Run Training Proccess
		if self.fedstar:
			model = PSL_Network(num_classes=self.num_classes, aux_loss_weight=self.aux_loss_weight)
			model.compile(optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])))
			model.set_weights(parameters)
			history = model.fit((self.train_L,self.train_U), num_batches=self.num_batches, epochs=int(config["epochs"]), c_round=int(config["c_round"]), rounds=int(config["rounds"]), verbose=self.verbose)
		else:
			model = Network(num_classes=self.num_classes).get_network()
			model.compile(optimizer=tf.keras.optimizers.Adam(float(config["learning_rate"])), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss'), metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
			model.set_weights(parameters)
			history = model.fit(self.train_L, batch_size=int(config["batch_size"]), epochs=int(config["epochs"]), verbose=self.verbose)
		# Print Results
		print(f"Client {self.client_id} finished {self.local_train_round}th round of training with loss {history.history['loss'][0]:.4f} and accuracy {history.history['accuracy'][0]:.4f}")
		# Clear Memory
		tf.keras.backend.clear_session()
		gc.collect()
		return model.get_weights(), self.num_examples_train, {"local_train_round": self.local_train_round}

	def evaluate(self, parameters, config):
		raise NotImplementedError('No client-side evaluation.')

def set_logger_level():	
	if 'flower' in [logging.getLogger(name).__repr__()[8:].split(' ')[0] for name in logging.root.manager.loggerDict]:
		logger = logging.getLogger('flower')
		logger.setLevel(LOG_LEVEL)