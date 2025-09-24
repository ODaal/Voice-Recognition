import os
#import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from os.path import dirname, join
import ast
import VGGpreprocess as v
import Model as m
from Model import vggvox_model
from Wav_reader import get_fft_spectrum
from Wav_reader import read_and_process_audio
import Constants as c
import json

###############Made for the model not my code :) ##################
def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets
###############Made for the model not my code :) ##################

def get_embeddings_from_list_file(model, list_file, max_sec):
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    result = pd.read_csv(list_file, delimiter=",")
    result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
    result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
    return result[['filename','speaker','embedding']]


def register_new_user(user_audio_path, user_name, max_sec):
    model = m.vggvox_model()
    model.load_weights(join(dirname(__file__), "weights.h5"))
    #print(f"Registering new user: {user_name}")
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    signal = read_and_process_audio(user_audio_path, buckets)  # Use read_and_process_audio to process the audio
    signal = np.expand_dims(signal, axis=-1)  # Add channel dimension
    embedding = np.squeeze(model.predict(signal.reshape(1, *signal.shape)))
    embedding_str = json.dumps(embedding.tolist())  # Convert embedding to JSON string
    new_user_data = pd.DataFrame([[user_audio_path, user_name, embedding_str]], columns=['filename', 'speaker', 'embedding'])

    # Check if the file exists and append accordingly
    if os.path.exists(c.ENROLL_LIST_FILE):
        new_user_data.to_csv(c.ENROLL_LIST_FILE, mode='a', header=False, index=False)
    else:
        new_user_data.to_csv(c.ENROLL_LIST_FILE, mode='w', header=True, index=False)

    print(f"User {user_name} registered successfully.")

def predict_user(test_audio_path, max_sec):
    model = m.vggvox_model()
    model.load_weights(join(dirname(__file__), "weights.h5"))
    #print(f"Predicting...")
    buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
    signal = read_and_process_audio(test_audio_path, buckets)  # Use read_and_process_audio to process the audio
    signal = np.expand_dims(signal, axis=-1)  # Add channel dimension
    test_embedding = np.squeeze(model.predict(signal.reshape(1, *signal.shape)))

    enroll_result = pd.read_csv(c.ENROLL_LIST_FILE, delimiter=",", header=0, names=['filename', 'speaker', 'embedding'])

    # Convert the embedding strings to numpy arrays
    enroll_result['embedding'] = enroll_result['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

    enroll_embs = np.array([emb for emb in enroll_result['embedding']])
    speakers = enroll_result['speaker']

    distances = pd.DataFrame(cdist([test_embedding], enroll_embs, metric=c.COST_METRIC), columns=speakers)
    predicted_speaker = distances.idxmin(axis=1).values[0]
    min_distance = distances.min(axis=1).values[0]
    #print(min_distance)
    #print(predicted_speaker)
    if min_distance > 0.5:  # If the minimum distance is greater than 0.5, the speaker is unknown
        predicted_speaker = "Unknown"
    #print(f"Predicted speaker: {predicted_speaker}")
    return predicted_speaker
