from os.path import dirname, join

# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025 # Assuming FRAME_LEN was in seconds
FRAME_STEP = 0.01 # Assuming FRAME_STEP was in seconds
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

MAX_FRAMES = 998

# Model
WEIGHTS_FILE = "weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE = (NUM_FFT,None, 1)

# IO
ENROLL_LIST_FILE = join(dirname(__file__), "Embeddings.csv")
