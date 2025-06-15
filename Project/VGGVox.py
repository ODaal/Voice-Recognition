import os
import librosa
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
import ast
import VGGpreprocess as v
import Model as m
from Model import vggvox_model
from Wav_reader import get_fft_spectrum
from Wav_reader import read_and_process_audio
from record import record_audio
import Constants as c
import json

os.environ['CUDA_HOME'] = 'C:/Users/Lucin/anaconda3/envs/tf_env'
os.environ['PATH'] += ';C:/Users/Lucin/anaconda3/envs/tf_env/Library/bin'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_vggvox_model(model_path="weights.h5"):
    model = tf.keras.models.load_model(model_path)
    return model

model = m.vggvox_model()
model.load_weights("weights.h5")


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


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets))
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1))))
	return result[['filename','speaker','embedding']]


def get_id_result():
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model()
	model.load_weights(c.WEIGHTS_FILE)
	model.summary()

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC)
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	print("Processing test samples....")
	test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers)

	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.concat([scores, distances],axis=1)
	scores['result'] = scores[speakers].idxmin(axis=1)
	scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int

	print("Writing outputs to [{}]....".format(c.RESULT_FILE))
	result_dir = os.path.dirname(c.RESULT_FILE)
	if not os.path.exists(result_dir):
	    os.makedirs(result_dir)
	with open(c.RESULT_FILE, 'w') as f:
		scores.to_csv(f, index=False)

def register_new_user(model, user_audio_path, user_name, max_sec):
    print(f"Registering new user: {user_name}")
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

def predict_user(model, test_audio_path, max_sec):
    print(f"Predicting...")
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
    if min_distance > 0.34:  # If the minimum distance is greater than 0.5, the speaker is unknown
        predicted_speaker = "Unknown"
    print(f"Predicted speaker: {predicted_speaker}")
    return predicted_speaker



if __name__ == '__main__':
    print("Hello, Please make sure you're voice is clear to register")
    user_name = input("Enter your name: ")
    record_audio('recordings/'+user_name+'.wav', 3, 16000)
    register_new_user(model, 'recordings/' + user_name + '.wav', user_name, c.MAX_SEC)