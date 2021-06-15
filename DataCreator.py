import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def FlipArray(arr):
    Newarr=np.zeros((arr.shape[1],arr.shape[0]))
    print(Newarr.shape,arr.shape)
    for X in range(arr.shape[0]):
        for Y in range(arr.shape[1]):
            Newarr[Y,X]=arr[X,Y]
    return Newarr


def CreateData(path,Train_num,Image_num,overall_num,save_path,FileName,GenreNum=4): #converts the song images from my folders into train and test np.arrays, very specific to my project
    TrainX= []
    TrainY= np.zeros((Train_num*GenreNum,GenreNum))
    TestX= []
    TestY= np.zeros(((Image_num-Train_num)*GenreNum,GenreNum))
    Genre = {0: 'Classical', 1: 'Jazz',2:'Kpop',3:'Rock'}
    Train_counter=0
    Test_counter=0
    for i in range(GenreNum):
        for a in range(Image_num):
            im = Image.open(path+'\\'+Genre[i]+'\\'+Genre[i]+str(a+1)+'.png')
            im = im.convert('RGB')
            im=np.asarray(im)
            if (a+1)>Train_num:
                TestX.append(im)
                TestY[Test_counter,i]=1
                Test_counter+=1
            else:
                TrainX.append(im)
                TrainY[Train_counter,i]=1
                Train_counter+=1
    TrainX=np.array(TrainX)
    TestX=np.array(TestX)

    TrainX,TrainY=unison_shuffled_copies(TrainX,TrainY) #shuffles the X and Y arrays so that they have the same random seed
    TestX,TestY=unison_shuffled_copies(TestX,TestY)

    #TrainX=TrainX.reshape(560,30000)
    #TestX=TestX.reshape(240,30000)
    TrainX = (TrainX.reshape(TrainX.shape[0],-1)).T  #flattens the arrays to be read by the model
    TestX = (TestX.reshape(TestX.shape[0],-1)).T
    TrainY=FlipArray(TrainY) #flips in order to fit the shape that the network can work with
    TestY=FlipArray(TestY)

    TrainX = TrainX/255 - 0.5 #normalizes the values to be 0.5 - (-0.5)
    TestX = TestX/255 - 0.5

    hf = h5py.File(save_path+'\\'+FileName+'.h5', 'w')
    hf.create_dataset('TrainX', data=TrainX)
    hf.create_dataset('TrainY', data=TrainY) 
    hf.create_dataset('TestX', data=TestX)
    hf.create_dataset('TestY', data=TestY) 
    hf.close()

CreateData(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs',140,200,800,r'C:\Users\Ziv Peltz\Desktop\Files for final project\DataSets','GrayscaleDATA')
