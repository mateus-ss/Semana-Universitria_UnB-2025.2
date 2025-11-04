import numpy as np
from scipy.io import loadmat
import sounddevice as sd  # Para ouvir o áudio

# Carregar o arquivo .mat
data = loadmat('exemplo_audio.mat')

# Extrair o sinal e a frequência de amostragem
x = data['x'].squeeze()
fs = int(data['fs'].squeeze())

# Ouvir o áudio
sd.play(x, fs)
sd.wait()
