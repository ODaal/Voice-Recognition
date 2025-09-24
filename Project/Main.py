import VGGVox as v
import constants as c
from record import record_audio
from VGGVox import predict_user
import Model as m
import Constants as c

model = m.vggvox_model()
model.load_weights("weights.h5")

if __name__ == '__main__':
    print("Hello, please use you're voice to authentificate")
    record_audio('recordings/authentificate.wav', 3, 16000)
    user = predict_user(model,'recordings/authentificate.wav', c.MAX_SEC)
    if user == 'Unknown':
        print("Sorry, you're not in the system, please register")
