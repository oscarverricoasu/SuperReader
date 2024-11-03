import torch
from TTS.api import TTS

class OutputSpeech:
    def getwav(self, words, file):
        #need to iterate through words

        #Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        #Init TTS
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)

        #Run TTS
        #Text to speech list of amplitude values as output
        wav = tts.tts(text=words[0])
        #Text to speech to a file
        tts.tts_to_file(text=words[0], file_path= file + ".wav")
