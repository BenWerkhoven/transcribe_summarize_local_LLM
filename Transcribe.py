# https://github.com/aza-ali/speech-to-text-in-python

import whisper
import os
from pathlib import Path

import whisper_timestamped as whisper_ts

# Function to load the model and handle errors
def load_model(timestamps):
    try:
        if timestamps:
            model = whisper_ts.load_model("small")  # You can change to "small", "medium", or "large"
        else:
            model = whisper.load_model("small")  # You can change to "small", "medium", or "large"
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Function to transcribe audio
def transcribe_audio_ts(model, audio_path):
    if not os.path.exists(audio_path):
        print("Error: Audio file does not exist.")
        return None

    try:
        # Perform transcription
        audio = whisper_ts.load_audio(audio_path)
        result = whisper_ts.transcribe(model,audio,language="nl")
        #result = model.transcribe(audio_path, language="nl", fp16=False, verbose=True, word_timestamps=True, clip_timestamps=[{'start': 0.0, 'end': 5.0},{'start': 5.0, 'end': 10.0}])
        transcript = ""
        for seg in result['segments']:
            uur_s = str(int(seg['start']/3600))
            min_s = str(int(seg['start']/60)- int(uur_s)*60)
            sec_s = str(int(seg['start'] - int(uur_s)*3600 - int(min_s)*60))
            uur_e = str(int(seg['end']/3600))
            min_e = str(int(seg['end']/60) - int(uur_e)*60)
            sec_e = str(int(seg['end'] - int(uur_e)*3600 - int(min_e)*60))
            transcript = transcript + f"{uur_s}:{min_s}:{sec_s}, {uur_e}:{min_e}:{sec_e}, {seg['text']} \n"
        #return result['text']
        return transcript, result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None
    
# Function to transcribe audio
def transcribe_audio(model, audio_path):
    if not os.path.exists(audio_path):
        print("Error: Audio file does not exist.")
        return None

    try:
        # Perform transcription
        result = model.transcribe(audio_path, language="nl", fp16=False, verbose=True, word_timestamps=True)
        #return result['text']
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

# Main code execution
def main():
    # Load the Whisper model
    timestamps=False
    model = load_model(timestamps)
    if model:
        # Path to the audio file        
        path_root = Path.cwd() / "speech-to-text" 
        project_folder = path_root / "Marcel 6 november 2026"
        audio_file = "26-10 gesprek Marcel .m4a"  # Update with your actual audio file path
        audio_file_name = audio_file.split(".")[0]
        text_file_ts = f"transcript_{audio_file_name}_ts.txt"
        text_file = f"transcript_{audio_file_name}.txt"
        audio_path = str(project_folder / audio_file)
        print(audio_path)
        # Transcribe the audio
        if timestamps:
            transcription, full_text = transcribe_audio_ts(model, audio_path)

            if transcription:
                print("Transcription: ", transcription)
                #print(json.dumps(transcription, indent = 2, ensure_ascii = False))
                with open(project_folder / text_file_ts, 'w', encoding="utf-8") as f:
                    f.write(transcription)
                with open(project_folder / text_file, 'w', encoding="utf-8") as f:
                    f.write(full_text)
            else:
                print("Transcription failed.")
        else:
            transcription = transcribe_audio(model, audio_path)

            if transcription:
                print("Transcription: ", transcription)
                text_file = f"transcript_{audio_file_name}_full.txt"
                with open(project_folder / f"{text_file}", 'w', encoding="utf-8") as f:
                    f.write(transcription)
            else:
                print("Transcription failed.")
    else:
        print("Failed to load Whisper model.")

# Run the main function
if __name__ == "__main__":
    main()