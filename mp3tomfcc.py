import librosa 
import matplotlib.pyplot as plt
import librosa.display
import os

def mfccmaker(audio_path,name,output_path,pxsize=64): #this function takes a sound file and converts it into an mfcc image
    song,cache=librosa.load(str(audio_path))
    mfcc = librosa.feature.mfcc(song)
    plt.figure(figsize=(pxsize/100,pxsize/100))
    librosa.display.specshow(mfcc)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(output_path +'\\'+ name, bbox_inches = 'tight',pad_inches = 0)
    plt.close()

def folder_to_mfcc(inputfolder_path,outputfolder_path,name): #this function takes a folder filled with mp3 files and converts them into mfccs
    file_list=os.listdir(inputfolder_path)
    for i in range (len(file_list)):
        mfccmaker(inputfolder_path+'\\'+file_list[i],name+str(i+1),outputfolder_path,100)






#path= r'C:\Users\Ziv Peltz\Downloads\DAY6 Shoot Me MV.mp3'
#folder_to_mfcc(path,r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Classical',"Classical")
#mfccmaker(path,'GAY6.png',r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Testing Examples',100)
