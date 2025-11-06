import click
from drum_transcription.data_pre_processer import DataPreProcesser
from drum_transcription.ml_model import train_model
from drum_transcription.predict import predict_transcription
import logging
import os

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

@click.group()
def cli():
  pass

@click.command()
@click.argument('input_path')
@click.argument('output_path')
def pre_process_dataset(input_path, output_path):
    click.echo(f'The data in {input_path} will be transfered to {output_path}')
    data = DataPreProcesser(input_path, output_path)
    data.create_data_set()

@click.command()
@click.argument('pre_process_dataset_path')
def train(pre_process_dataset_path):
    click.echo('Training the model based on the dataset ' + pre_process_dataset_path)
    train_model(pre_process_dataset_path)

@click.command()
@click.argument('input_audio_file')
@click.argument('output_midi_file')
def predict(input_audio_file, output_midi_file):
    click.echo('The audio ' + input_audio_file + " will be transcribed to " + output_midi_file)
    predict_transcription(input_audio_file, output_midi_file)

cli.add_command(pre_process_dataset)
cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
