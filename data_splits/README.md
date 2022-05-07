# Datasets Preparation

The train/test splits for all 3 datasets used in our evaluation (e.g., [Speech Commands v2](https://www.tensorflow.org/datasets/catalog/speech_commands), [VoxForge](https://www.tensorflow.org/datasets/catalog/voxforge) and [Ambient Context](https://www.esense.io/datasets/ambientacousticcontext/index.html)) is provided in the corresponding directories. The splits are provided in the form of `.txt` files, providing the filepath to the corresponding `.wav` file.

---

In the [config.yml](../config.yml) file, the path to the desired dataset is expected to be provided. A typical top-level directory layout of [DATA_DIR](../config.yml#L13) is:
```
  .							# Root directory of dataset.
  ├── Data					# Directory contraining all dataset samples.
  	  ├────Train			# Directory contraining all train files.
  	  ├────Test				# Directory contraining all test files.
  ├── train_split.txt		# File contraining the train .wav samples filepaths with their corresponding labels.
  ├── test_split.txt		# File contraining the test .wav samples filepaths with their corresponding labels.
```
---

## Speech Commands Dataset
To prepare the `Speech Commnads` dataset directory, run the following commands:

```console
foo@bar:~$ mkdir speech_commands && cd speech_commands
foo@bar:~$ wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz -O train.tar.gz && mkdir -p ./Data/Train && tar -zxvf ./train.tar.gz  -C ./Data/Train
foo@bar:~$ wget http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz -O test.tar.gz && mkdir -p ./Data/Test && tar -zxvf ./test.tar.gz  -C ./Data/Test
```
The above commands will create the `Data` directory. In addition to this, you need to copy the corresponding train-test .txt files from [data_splits]((data_splits/speech_commands)) to the root `speech_commands` directory.

