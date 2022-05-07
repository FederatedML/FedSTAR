##########################################
import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
##########################################
import pathlib
import argparse
import multiprocessing
import distutils.util
from multiprocessing import Process
multiprocessing.set_start_method('spawn', force=True)

def distribute_gpus(num_clients, client_memory=1280):
	gpus = ['0','1']
	clients_gpu= [None]*num_clients
	if not gpus:
		return clients_gpu
	else:
		gpu_free_mem = [11000,11000]
		for client_id in range(num_clients):
			gpu_id = gpu_free_mem.index(max(gpu_free_mem))
			if gpu_free_mem[gpu_id] >= client_memory:
				gpu_free_mem[gpu_id] -=client_memory
				clients_gpu[client_id] = gpus[gpu_id]
			else:
				clients_gpu[client_id] = None
	return clients_gpu

def main():

	parser = argparse.ArgumentParser(description="Flower Federated Learning - Clients")
	parser.add_argument("--server_address",     type=str,   default="[::]:8080",    required=False,  help=f"gRPC server address (default: [::]:8080)")
	parser.add_argument("--dataset_dir",        type=str,   default="../dataset",   required=False,  help="Relevant path to dataset")
	parser.add_argument("--gpu_memory",         type=int,   default=1024,           required=False,  help="GPU Available Memory")
	parser.add_argument("--num_clients",        type=int,   default=2,              required=False,  help="Number of clients")
	parser.add_argument("--batch_size",         type=int,   default=64,             required=False,  help="Training Batch Size")
	parser.add_argument("--l_per",              type=float, default=0.1,            required=False,  help="Percentage of dataset to be used as labelled data")
	parser.add_argument("--u_per",              type=float, default=1.0,            required=False,  help="Percentage of dataset to be used as unlabelled data")
	parser.add_argument("--fedstar",            type=str,   default="False",        required=False,  help="Enabler for Semi-Supervised Learning")
	parser.add_argument("--pseudo_label",       type=str,   default="False",        required=False,  help="Enabler for Pseudo-labelling")
	parser.add_argument("--class_distribute",   type=str,   default="False",        required=False,  help="Enabler for distributing samples according to class")
	parser.add_argument("--seed",               type=int,   default=2021,           required=False,  help="Seed for clients data shuffling")
	parser.add_argument("--verbose",            type=int,   default=0,              required=False,  help="Train Detail to be displayed")
	args = parser.parse_args()

	clients_gpu = distribute_gpus(num_clients=args.num_clients, client_memory=args.gpu_memory)
	data_dir = os.path.join(pathlib.Path.home(),args.dataset_dir)

	# Load Configurations of Clients
	clients_data = [{'client_id': i, 'num_clients': args.num_clients, 'server_address': args.server_address[1:-1],
		'dataset_dir': data_dir, 'batch_size': args.batch_size, 'gpu_id': clients_gpu[i], 
		'gpu_memory': args.gpu_memory, 'seed': args.seed, 'l_per': args.l_per, 'u_per': args.u_per, 
		'fedstar': args.fedstar, 'class_distribute': args.class_distribute, 'verbose': args.verbose,
		} for i in range(args.num_clients)]

	# Start Multi-processing Clients and wait for them to finish
	clients_queue = multiprocessing.JoinableQueue()
	clients = [Client(clients_queue) for i in range(args.num_clients)]
	[client.start() for client in clients]
	[clients_queue.put(client) for client in clients_data]
	[clients_queue.put(None) for i in range(args.num_clients)]
	clients_queue.join()

class Client(Process):

	def __init__(self, queue):
		Process.__init__(self)
		self.queue = queue

	def run(self):
		from client import AudioClient
		while True:
			cfg = self.queue.get()
			if cfg is None:
				self.queue.task_done()
				break
			# Configure GPU
			Client.setup_gpu(gpu=cfg["gpu_id"],gpu_memory=cfg["gpu_memory"])
			# Create Client
			client = AudioClient(client_id=cfg["client_id"], server_address=cfg["server_address"],
				num_clients=cfg["num_clients"], dataset_dir=cfg["dataset_dir"], l_per=cfg["l_per"], u_per=cfg["u_per"],
				batch_size=cfg["batch_size"], verbose=cfg["verbose"],  seed=cfg["seed"], 
				fedstar=bool(distutils.util.strtobool(cfg["fedstar"])), 
				class_distribute=bool(distutils.util.strtobool(cfg["class_distribute"])))
			# Start Client
			client(introduce=bool(cfg["verbose"]))
			# Return job done once client is terminated.
			self.queue.task_done()

	@staticmethod
	def setup_gpu(gpu, gpu_memory):
		os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu is not None else ""
		import tensorflow as tf
		gpus = tf.config.list_physical_devices("GPU")
		if not gpus:
			print("No GPU's available. Client will run on CPU.")
		else:
			try:
				tf.config.experimental.set_virtual_device_configuration(gpus[0],
					[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory)])
			except RuntimeError as e:
				print(e)

if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
	sys.exit(0)
