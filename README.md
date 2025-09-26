# 🏠 Voice Authentication & Command System (Intended for Smart Home)

This project implements a **voice authentication and control system** designed with the intention of being integrated into a **smart home environment** for my group Capstone Project. 
It uses the **VGGVox speaker verification model** (imported with pretrained weights) to authenticate users by their voice before allowing access to voice-based commands.

---

## 🚀 Features
- 🎙️ **Voice Authentication** using a pretrained **VGGVox** deep learning model.  
- 🔐 **Secure Access**: unknown speakers are rejected automatically.  
- 💡 **Voice Command Interface** (intended for smart home use):  
  - "Switch on/off the light"  
  - "Activate/deactivate security system"  
- 📂 **User Management**:
  - Register new users with their voice samples.  
  - Store embeddings in `Embeddings.csv`.  
- 📊 **Signal Processing**:
  - Pre-emphasis, framing, FFT spectrum extraction.  
  - Normalization & bucket-based input shaping for the CNN model.  
- 🤖 **Deep Learning Backend**:
  - Imported **VGGVox CNN** with pretrained weights (`weights.h5`).  
  - Cosine similarity for speaker verification.

---

## 📊 Example Workflow
1. **Authentication**  
   - User speaks → system records audio (`recordings/authentificate.wav`).  
   - Model extracts embedding & compares with enrolled users.  
   - If the distance is too large → marked as **Unknown**.  

2. **Voice Command Recognition** (for smart home integration)  
   - User says **"Please smart home"** (wake word).  
   - Then issues commands such as:  
     - "Switch on the light" → 💡 Light ON  
     - "Switch off the light" → 💡 Light OFF  
     - "Activate the security" → 🔐 Security ON  
     - "Desactive the security" → 🔓 Security OFF  

---

## 🧩 Project Structure
- `Constants.py` → signal processing & model constants.  
- `Model.py` → VGGVox CNN architecture (imported, not self-built).  
- `VGGpreprocess.py` → signal preprocessing (framing, preemphasis, FFT).  
- `Wav_reader.py` → audio reading, FFT spectrum computation.  
- `record.py` → record audio clips.  
- `VGGVox.py` → speaker registration & prediction (embedding management).  
- `voice_control.py` → wake-word detection + command execution.  
- `Embeddings.csv` → stored user embeddings.  
- `weights.h5` → **imported pretrained weights** for VGGVox.  

---

## 🔮 Future Improvements
- Add **face recognition** for multi-factor authentication.  
- Expand voice command set (lights, thermostat, appliances).  
- Improve wake-word detection with a custom keyword spotter.  
- Mobile/IoT integration for full smart home deployment.  

---

## 👤 Author
Othmane Daali

⚠️ The **VGGVox model and weights are imported** and not built from scratch.
