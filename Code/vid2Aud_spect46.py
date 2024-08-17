import pandas as pd
import librosa as librosa
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import librosa.display
import os
import moviepy.editor as mp
##########################################################################################################################

def vid2Aud(file,Path):
    file_name = file.split('.')[0]
    pos = file_name.rfind('/')
    file_name = file_name[(pos+1) : ]
    video = mp.VideoFileClip(file)
    video.audio.write_audiofile('%s%s.wav'%(Path, file_name ))
    
def mel_spect(wav_file,Path):
    file_name = wav_file.split('.')[0]
    pos = file_name.rfind('/')
    file_name = file_name[(pos+1) : ]
    n_fft=2048
    hop_length=512
    y, sr = librosa.load(wav_file)
    spectrogram_librosa = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    librosa.display.specshow(spectrogram_librosa_db, sr=sr, y_axis='mel', x_axis='time', hop_length=hop_length,fmax=8000)
    plt.gray()
    plt.tight_layout()
    plt.savefig('%s%s.jpg'%(Path, file_name ))
    plt.show()
    
################################################################################################################################  

metadata = pd.read_csv("Dataset2/metadata_46.csv")

#real_df = metadata.iloc[:, 3 ]
#real_df = real_df.dropna()

#real_fake_df = metadata.iloc[:, 0:2 ]

real_df = metadata[metadata['label'] == 'REAL']
real_df = real_df.reset_index()
real_final = real_df.iloc[ : , 1]

fake_df = metadata[metadata['label'] == 'FAKE']
fake_df = fake_df.reset_index()
fake_final = fake_df.iloc[ : , 1 ]

#real_final = real_df.append(real_nan_df,ignore_index ='True')

del metadata,fake_df, real_df

###########################################################################################################################

DATA_FOLDER = 'Dataset2/'
TRAIN_SAMPLE_FOLDER = 'train46_videos/'
OUTPUT_TRAIN_WAV = 'train46_wav/' 
OUTPUT_TRAIN_REAL_IMG = 'train46_img_jpg/real/'
OUTPUT_TRAIN_FAKE_IMG = 'train46_img_jpg/fake/'

##########################################################################################################################

train_list_of_videos = []
for video in os.listdir(os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)):
    filename_train_video = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)+video
    train_list_of_videos.append(filename_train_video)
del video,filename_train_video

train_wav = os.path.join(DATA_FOLDER,OUTPUT_TRAIN_WAV)
count = 0
for file in train_list_of_videos:
    #extension = file.split('.')[1]
    #if extension == 'json':
        #continue3.
        
    vid2Aud(file,train_wav)
    count+=1
    print(count)
del file, train_list_of_videos, count
#########################################################################################################################
    
train_img_real = os.path.join(DATA_FOLDER, OUTPUT_TRAIN_REAL_IMG)

real_list_of_wavs = []
for wav in real_final:
    wav_name = wav.split('.')[0]
    wav = wav_name+'.wav'
    filename_wav_train = os.path.join(DATA_FOLDER,OUTPUT_TRAIN_WAV)+wav
    real_list_of_wavs.append(filename_wav_train) 
del wav,wav_name,filename_wav_train

for file in real_list_of_wavs:
    mel_spect(file,train_img_real)
del file,real_list_of_wavs

##################################################################

train_img_fake = os.path.join(DATA_FOLDER, OUTPUT_TRAIN_FAKE_IMG)

fake_list_of_wavs = []
for wav in fake_final:
    wav_name = wav.split('.')[0]
    wav = wav_name+'.wav'
    filename_wav_train = os.path.join(DATA_FOLDER,OUTPUT_TRAIN_WAV)+wav
    fake_list_of_wavs.append(filename_wav_train) 
del wav,wav_name,filename_wav_train
count = 0
for file in fake_list_of_wavs:
    count+=1
    mel_spect(file,train_img_fake)
del file, fake_list_of_wavs

########################################################################################################################

