import numpy as np
import librosa
import librosa.feature as feature
from tensorflow import keras
import logging
import os
import mido as md
import pathlib as path
from drum_transcription.settings.settings import Settings
from drum_transcription import GROOVE_PITCH_POST_PROCESS

log = logging.getLogger(__name__)

settings = Settings()

def predict_transcription(input_audio_file, output_midi_file):
    """
    It transcribe a input audio file to a output midi file. It carries out the whole pipeline

    :param input_audio_file:
    :param output_midi_file:
    :return:
    """

    y, sr = librosa.load(input_audio_file, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bpm = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    print(bpm)

    y_size = len(y)
    if sr != settings.sampling_frequency:
        log.warning(
            "The sampling frequency of the audio " + input_audio_file + " is not " + str(settings.sampling_frequency)
        )

    onset_idx = librosa.onset.onset_detect(
        y,
        sr=settings.sampling_frequency,
        hop_length=settings.hop_size_onset,
        units="samples"
    )

    samples_window = int(settings.analysis_window_s * settings.sampling_frequency)
    model = keras.models.load_model("model_full_export")

    predictions = []

    for onset in onset_idx:

        onset_t0 = int(onset - samples_window / 4)
        onset_tf = onset_t0 + samples_window
        if onset_tf >= y_size or onset_t0 < 0:
            continue

        window = y[onset_t0: onset_tf]

        mfcc_features = feature.mfcc(
            y=window,
            sr=settings.sampling_frequency,
            hop_length=settings.FFT_hop_size,
            n_mfcc=settings.n_mfcc_features
        )

        # normalize features
        mfcc_features = (mfcc_features - np.min(mfcc_features)) / (np.max(mfcc_features) - np.min(mfcc_features))

        #fixes inputshape needed for model.predict for a single input
        mfcc_features = mfcc_features[np.newaxis,...]
        mfcc_features = mfcc_features[...,np.newaxis]

        y_predicted = model.predict(mfcc_features)
        y_pre_binary = np.where(y_predicted > settings.threshold_prediction, 1, 0)
        y_pre_binary = y_pre_binary[0, :]

        notes = []
        for id, label in enumerate(y_pre_binary):

            if label == 1:
               notes.append(GROOVE_PITCH_POST_PROCESS[id][0])

        predictions.append([onset/settings.sampling_frequency, [*notes]])
    
    print("just before post processing")
    print(predictions)
    convert_midi(output_midi_file, predictions, bpm)


def convert_midi(midi_file, predictions, bpm):
    """
    It converts the predicition into a midi file.

    :param midi_file:
    :param predictions:
    :param bpm
    :return:
    """
    predictions = np.array(predictions)
    print("predictions")
    print(predictions.shape)
    print(predictions)

    times = predictions[:, 0]
    notes = predictions[:, 1]

    midi = md.MidiFile(type=0, ticks_per_beat=1000)
    tempo = md.bpm2tempo(int(bpm))
    tpb = midi.ticks_per_beat
    track = md.MidiTrack()
    midi.tracks.append(track)
    track.append(md.MetaMessage('set_tempo', tempo=tempo, time=0))
    track.append(md.Message('program_change', program=12, time=0))

    default_length_sec = md.tick2second(tpb / 8, tpb, tempo=tempo) # 1/8 note length drum hit

    events_unsorted = np.zeros((len(predictions) + 2, len(times) * 2))
    print("events_unsorted")
    print(events_unsorted.shape)
    for i, t in enumerate(times):
        if t < 0:
            continue
        events_unsorted[0, 2 * i] = t  # timing note_on
        events_unsorted[0, 2 * i + 1] = t + default_length_sec  # timing note_off
        events_unsorted[1, 2 * i:2 * i + 2] = (1 if len(notes[i]) else 0)  # note exists
        events_unsorted[2, 2 * i:2 * i + 2] = i  # idx to find other notes with same onset
        events_unsorted[3, 2 * i] = 1  # 1 when note_on
        events_unsorted[3, 2 * i + 1] = 0  # 0 when note_off

    events_sorted = events_unsorted[:, events_unsorted[0, :].argsort()]

    jumped = 0
    for i, time in enumerate(events_sorted[0]):
        if i > 0:
            time_gone = events_sorted[0, i - 1 - jumped]
            ticks = md.second2tick(time - time_gone, tpb, tempo=tempo)
        else:
            ticks = md.second2tick(time, tpb, tempo=tempo)
        if events_sorted[1, i]:
            for n in notes[int(events_sorted[2, i])]:
                # note_on
                if ticks >= 0:
                    if events_sorted[3, i] == 1:
                        track.append(md.Message('note_on', note=n, time=int(ticks)))
                    # note_off
                    else:
                        track.append(md.Message('note_off', note=n, time=int(ticks)))
                ticks = 0
            jumped = 0
        else:
            jumped += 1

    if not (midi_file.endswith('.mid') or midi_file.endswith('.midi')):
        midi_file += '.mid'
    midi.save(midi_file)
    return midi
