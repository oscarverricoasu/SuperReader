import torch
from TTS.api import TTS

class OutputSpeech:
    def getwav(self, words, file):
        #need to iterate through words

        #Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #Init TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

        iter = 0
        #Run TTS
        for iter in range(len(words)):
            #Text to speech list of amplitude values as output
            wav = tts.tts(text=words[iter])
            #Text to speech to a file
            tts.tts_to_file(text=words[iter], file_path= file + str(iter) + ".wav")

        #combine wav files into one?
