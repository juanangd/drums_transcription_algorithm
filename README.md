# Drum Transcription algorithm 

## Requirements
* Python 3.7
* Poetry

## Poetry package management

Poetry was used for this Project which is a tool for dependency management and packaging in Python. You can declare and manage libraries with it and define specific versions for them. For installation, the following link provides the needed information.

[https://python-poetry.org/docs/](https://python-poetry.org/docs/)

 We decided to use poetry, because we wanted to create an environment to use TensorFlow with the DirectML package. That package was needed to fully use the computing power of an AMD GPU (RX 6800 XT) and save time to train the model.

The basic usage of poetry for is also well documented on here.

[https://python-poetry.org/docs/basic-usage/](https://python-poetry.org/docs/basic-usage/)


Also, this is mentioned in the poetry docs, but i must restate, DO NOT install poetry in the same environment that you're using for this project

NOTE, poetry install does not setup your python environment for you, so you must do it yourself. I recommend Anaconda

```
conda create --name drums python=3.7
```

and 

```
conda activate drums
```



To install our dependencies, simply run:

```
poetry install
```
Poetry will create a new environment with the needed dependecies and packages defined in the *pyproject.toml* file.

To make use of the GPU AMD GPU (RX 6800 XT) the additional library `tensorflow-directml` must be installed. This can be done running the following command:
```
poetry install -E dev_tools
```

## Running the project 

The script `cli.py` located on the root folder of the repository contains all the command implemented on this project. If you run the following command you will see the available commands.
```
poetry run python cli.py
```
To run the preprocessing, input- and output-folders paths need to be passed as arguments where `INPUT PATH` is the folder containing the unzipped dataset groove (which can be downloaded from: https://magenta.tensorflow.org/datasets/groove) and `OUTPUT_PATH` is the folder where the data will be transferred. 
```
poetry run python cli.py pre-process-dataset INPUT_PATH OUTPUT_PATH
```
In our case, we extracted eight tracks from the whole GROOVE Dataset before training the model, to use them later for validating our model predictions.

The training could be done executing the following command where `PRE_PROCESS_DATASET_PATH` is the folder containing the pre-processed data. This function doesn't only run the training part of the model but also it shuffles the data, it splits the dataset into training and validation and feed the data to the algorigthm.  
```
poetry run python cli.py train PRE_PROCESS_DATASET_PATH
```
To predict the transcription of a *.wav* file the following command should be run where `INPUT_AUDIO_FILE` is the *.wav* file to be converted into the `OUTPUT_MIDI_FILE` 
```
poetry run python cli.py predict INPUT_AUDIO_FILE OUTPUT_MIDI_FILE
```
