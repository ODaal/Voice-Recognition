import speech_recognition as sr

WAKE_WORD = "Please smart home"

def recognize_speech_from_mic(recognizer, microphone):
    """Capture audio from the microphone and tra  nscribe it using Google API."""
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("🎤 Listening...")
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

def process_command(command):
    """Handle recognized voice commands."""
    if "switch on" in command:
        print("💡 Switching on the light")
    elif "switch off" in command:
        print("💡 Switching off the light")
    elif "activate the security" in command:
        print("🔐 Security system activated.")
    elif "desactive the security" in command:
        print("🔓 Security system deactivated.")
    else:
        print(f"❓ Unknown command: {command}")

def main():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Say 'Hey Smart Home' to activate voice control...")

    while True:
        wake_response = recognize_speech_from_mic(recognizer, microphone)

        if wake_response["transcription"]:
            if WAKE_WORD in wake_response["transcription"].lower():
                print("✅ Wake word detected. Listening for command...")
                command_response = recognize_speech_from_mic(recognizer, microphone)

                if command_response["transcription"]:
                    command = command_response["transcription"].lower()
                    process_command(command)

                elif command_response["error"]:
                    print(f"⚠️ Command error: {command_response['error']}")
            else:
                print("🛑 Wake word not detected. Ignoring.")
        elif wake_response["error"]:
            print(f"⚠️ Wake word error: {wake_response['error']}")

if __name__ == "__main__":
    main()
