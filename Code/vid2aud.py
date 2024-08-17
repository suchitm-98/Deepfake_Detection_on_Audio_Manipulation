import moviepy.editor as mp
import speech_recognition as sr

file = "aassnaulhq.mp4"
file_name = file.split('.')[0]
#target = '/'
#res = file_name.rfind('/') 
#file_name = file_name[(res+1):]print(file_name)
video = mp.VideoFileClip(file)
#p = 'Dataset/train_wav'
video.audio.write_audiofile('%s.wav'%(file_name ))
#a= "%s.wav"% os.path.splitext(file)[0]
a = '%s.wav'%(file_name )



