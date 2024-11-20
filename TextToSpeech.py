import torch
from pydub import AudioSegment
from TTS.api import TTS

class OutputSpeech:
    def getwav(self, words, file):

        #Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #Init TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

        iter = 0
        speech = [] #add audio parts to this
        #Run TTS
        for iter in range(len(words)):
            #Text to speech list of amplitude values as output
            wav = tts.tts(text=words[iter])
            #Text to speech to a file
            tts.tts_to_file(text=words[iter], file_path= file + str(iter) + ".wav")
            speech.append(file + str(iter) + ".wav")

        #combine wav files into one
        start = AudioSegment.from_wav(speech[0])
        gap = start[:10] - 100 #silent gap between parts
        for part in speech:
            file = AudioSegment.from_wav(part)
            gap = file + gap
        gap.export("mcpoemshort.wav", format="wav")

        #add deletion of all .wav parts after program is more throughly tested

