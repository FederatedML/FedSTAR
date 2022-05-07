import pandas as pd
import numpy as np
import tensorflow as tf
from utils import AudioTools, DataTools

class DataBuilder:

	AUTOTUNE = tf.data.experimental.AUTOTUNE
	WORDS = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', '_silence_']

	@staticmethod
	def get_files(data_dir, train=False, raw=False):

		if train:
			train_user_files = pd.read_csv(f"{data_dir}/train_split.txt", header=None).values.flatten()
			train_files_path, train_labels = [], []
			for tr_uf in train_user_files:
				train_files_path.append(f"{data_dir}/Data/Train/{tr_uf}")
				label = tr_uf.split("/")[0]
				if label in __class__.WORDS:
					train_labels.append(tr_uf.split("/")[0])
				else:
					train_labels.append('_unknown_')
			# One-hot labels transformation
			train_labels = np.array(train_labels, dtype=object)
			# Map labels to 0-11
			unique_labels = np.unique(train_labels)
			num_classes = len(unique_labels)
			labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
			train_labels = [labels_dict[train_labels[i]] for i in range(len(train_labels))]
			# Change from object type to int
			train_labels = np.array(train_labels, dtype=np.int64)
			# Convert to dataset (if necessary)
			ds = (train_files_path, train_labels) if raw else tf.data.Dataset.from_tensor_slices((train_files_path, train_labels))\
					.map(AudioTools.read_audio, num_parallel_calls=__class__.AUTOTUNE)
		else: 
			# Read files from txt file.
			test_user_files = pd.read_csv(f"{data_dir}/test_split.txt", header=None).values.flatten()
			test_files_path, test_labels = [], []
			# Create dataset files.
			for ts_uf in test_user_files:
				test_files_path.append(f"{data_dir}/Data/Test/{ts_uf}")
				test_labels.append(ts_uf.split("/")[0])
			# One-hot labels transformation
			test_labels = np.array(test_labels, dtype=object)
			# Map labels to 0-12
			unique_labels = np.unique(test_labels)
			num_classes = len(unique_labels)
			labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
			test_labels = [labels_dict[test_labels[i]] for i in range(len(test_labels))]
			# Change from object type to int
			test_labels = np.array(test_labels, dtype=np.int64)
			# Convert to dataset (if necessary)
			ds = (test_files_path, test_labels) if raw else tf.data.Dataset.from_tensor_slices((test_files_path, test_labels))\
					.map(AudioTools.read_audio, num_parallel_calls=__class__.AUTOTUNE)
		return ds, num_classes

	@staticmethod
	def split_dataset(data_dir, num_clients, client, batch_size=64, variance=0.25, l_per=0.1, u_per=1.0, 
		mean_class_distribution=3, class_distribute=False, fedstar=False, seed=2021):
		# Load data
		ds_train, num_classes = __class__.get_files(data_dir=data_dir, train=True, raw=True)
		(ds_train_l,labelled_size), (ds_train_u,unlabelled_size) = DataTools.get_subset(dataset=ds_train, percentage=l_per, u_per=u_per, num_classes=num_classes, seed=seed)
		ds_train_u = DataTools.convert_to_unlabelled(dataset=ds_train_u, unlabelled_data_identifier=-1) if fedstar else []
		# Split data
		if not class_distribute: # Split according to number of samples
			labelled_sets = list(DataTools.distribute_per_samples(dataset=ds_train_l, num_clients=num_clients, variance=variance, seed=seed)) 
		else: # Split according to classes
			labelled_sets = DataTools.distribute_per_class_with_class_limit(dataset=ds_train_l, num_clients=num_clients, num_classes=num_classes, mean_class_distribution=mean_class_distribution, class_variance=variance, seed=seed)
		unlabelled_sets = list(DataTools.distribute_per_samples(dataset=ds_train_u, num_clients=num_clients, variance=variance, seed=seed)) if fedstar else ([],[])
		# Convert to tf dataset objects
		ds_train_labelled = tf.data.Dataset.from_tensor_slices((labelled_sets[client][0], labelled_sets[client][1])).map(AudioTools.read_audio, num_parallel_calls=__class__.AUTOTUNE)
		ds_train_unlabelled = tf.data.Dataset.from_tensor_slices((unlabelled_sets[client][0], unlabelled_sets[client][1])).map(AudioTools.read_audio, num_parallel_calls=__class__.AUTOTUNE) if fedstar else None
		# Calculate datasets info for training
		labelled_size, unlabelled_size = len(labelled_sets[client][0]), len(unlabelled_sets[client][0]) if fedstar else 0
		num_batches = (unlabelled_size+batch_size-1)//batch_size if fedstar else (labelled_size+batch_size-1)//batch_size
		# Print datasets sizes
		print(f"Client {client}: Train data {labelled_size+unlabelled_size} (Unlabelled: {unlabelled_size} - Labelled: {labelled_size})")
		return ds_train_labelled, ds_train_unlabelled, num_classes, num_batches

	@staticmethod
	def get_ds_test(data_dir, batch_size, buffer=1024, seed=2021):
		ds_test, num_classes = __class__.get_files(data_dir=data_dir)
		_, _, ds_test = __class__.to_Dataset(ds_train_L=None, ds_train_U=None, ds_test=ds_test, buffer=buffer, batch_size=batch_size, seed=seed)
		return ds_test, num_classes

	@staticmethod
	def load_sharded_dataset(data_dir, num_clients, client, variance=0.25, batch_size=64, l_per=1.0, u_per=1.0, 
		mean_class_distribution=5, fedstar=False, class_distribute=False, seed=2021):
		
		batch_size = batch_size//2 if fedstar else batch_size

		ds_train_L, ds_train_U, num_classes, num_batches = __class__.split_dataset(
			data_dir=data_dir, client=client, num_clients=num_clients,
			l_per=l_per, u_per=u_per, fedstar=fedstar, class_distribute=class_distribute,
			mean_class_distribution=mean_class_distribution,
			batch_size=batch_size, variance=variance, seed=seed)
		ds_train_L, ds_train_U, _ = __class__.to_Dataset(ds_train_L=ds_train_L, ds_train_U=ds_train_U, ds_test=None, seed=seed, batch_size=batch_size)
		return ds_train_L, ds_train_U, num_classes, num_batches

	@staticmethod
	def to_Dataset(ds_train_L, ds_train_U, ds_test, batch_size, buffer=1024, seed=2021):
		ds_train_L = ds_train_L.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)\
			.map(AudioTools.prepare_example, num_parallel_calls=__class__.AUTOTUNE)\
				.batch(batch_size=batch_size).prefetch(__class__.AUTOTUNE) if ds_train_L else None
		ds_test = ds_test.map(AudioTools.prepare_test_example, num_parallel_calls=__class__.AUTOTUNE)\
			.batch(1).prefetch(__class__.AUTOTUNE) if ds_test else None
		ds_train_U = ds_train_U.shuffle(buffer_size=buffer, seed=seed+1, reshuffle_each_iteration=True)\
			.map(AudioTools.prepare_example, num_parallel_calls=__class__.AUTOTUNE)\
				.batch(batch_size=batch_size).prefetch(__class__.AUTOTUNE) if ds_train_U else None
		return ds_train_L, ds_train_U, ds_test 