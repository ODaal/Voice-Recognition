
# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

# Model
WEIGHTS_FILE = "weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)

# IO
ENROLL_LIST_FILE = "cfg/libri-dev-clean_batch_enroll_list.csv"
TEST_LIST_FILE = "cfg/libri-dev-clean_batch_test_list.csv"
RESULT_FILE = "res/results.csv"
