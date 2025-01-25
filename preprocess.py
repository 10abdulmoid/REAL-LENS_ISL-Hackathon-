import librosa  # Make sure this is imported
from transformers import Wav2Vec2Processor

# Load Pretrained Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def load_audio(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)  # Resample to 16kHz
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs.input_values

# Example Usage
file_path = "/Users/abdulmoid/Desktop/audio/AUDIO-2025-01-24-23-53-01.wav"  # Change this!
audio_input = load_audio(file_path)
print(audio_input.shape)  # Check tensor shape
