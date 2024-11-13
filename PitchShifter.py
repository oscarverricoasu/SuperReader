import librosa
import soundfile as sf
import os


def pitch_shift_and_preserve_duration(audio_data, sr, factor):
    # Step 1: Calculate the pitch-shifted rate
    new_sr = int(sr * factor)

    # Step 2: Resample to the new sample rate to shift pitch
    shifted_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=new_sr)

    # Step 3: Time-stretch back to the original duration
    original_length = len(audio_data) / sr
    duration_stretch_factor = len(shifted_audio) / (sr * original_length)
    stretched_audio = librosa.effects.time_stretch(shifted_audio, rate=duration_stretch_factor)

    return stretched_audio


input_files = os.listdir("./audio")
counter = 1
for file in input_files:
    if file.endswith('.wav'):
        # Load the audio file
        audio_data, sr = librosa.load(f"./audio/{file}", sr=None)

        shifted_audio = pitch_shift_and_preserve_duration(audio_data, sr, 0.8)

        # Save the modified audio
        sf.write(f"./audio/shifted_audio{counter}.wav", shifted_audio, sr)

        counter += 1
