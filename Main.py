from DL4 import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
from PIL import Image
from mp3tomfcc import *

def Load_Data(path,name='data.h5'):
    hf = h5py.File(path+'\\'+name, 'r')
    n1 = hf.get('TrainX')
    n2 = hf.get('TrainY')
    n3 = hf.get('TestX')
    n4 = hf.get('TestY')
    n1 = np.array(n1)
    n2 = np.array(n2)
    n3 = np.array(n3)
    n4 = np.array(n4)
    return n1,n2,n3,n4
def Load_Image(path,name):
    im = Image.open(path+'\\'+name)
    im = im.convert('RGB')
    im=np.asarray(im)
    im = (im.reshape(30000,1))
    im = im/255 - 0.5
    return im
def ImagePrediction(path,name):
    p=model.predict(Load_Image(path,name))
    name = name.split('.')[0]
    name = name.split('.')[0]
    if(p[0,0]==1): return name + " Is a Classical Song"
    if(p[1,0]==1): return name + " Is a Jazz Song"
    if(p[2,0]==1): return name + " Is a Kpop Song" 
    if(p[3,0]==1): return name + " Is a Rock Song" 
def CheckImageGenre(path,name,temppath): #input mp3 path to check what genre it is
    mfccmaker(path+'\\'+name,name+'.png',temppath,100)
    print(ImagePrediction(temppath,name+'.png'))
    os.remove(temppath+'\\'+name+'.png')

train_x,Y_train,test_x,Y_test=Load_Data(r'C:\Users\Ziv Peltz\Desktop\Files for final project\DataSets','UPDATEDDATA.h5')


path=r'C:\Users\Ziv Peltz\Desktop\Files for final project\Weights\93%'
#print(train_x)
#print(train_x.shape,Y_train.shape,test_x.shape,Y_test.shape)

#hidden1 = DLLayer(50, (train_x.shape[0],), W_initialization=(path+'\layer1.h5'),activation="leaky_relu", learning_rate=0.007, optimization="adaptive",regularization="L2")
#hidden2 = DLLayer(15, (50,), W_initialization=(path+'\layer2.h5'),activation="leaky_relu", learning_rate=0.007, optimization="adaptive",regularization="L2")
#opt = DLLayer(4, (15,), W_initialization=(path+'\layer3.h5'),activation="softmax", learning_rate=0.007, optimization="adaptive",regularization="L2")  
hidden1 = DLLayer(50, (train_x.shape[0],), W_initialization="Xaviar",activation="leaky_relu", learning_rate=0.007, optimization="adaptive",regularization="L2")
hidden2 = DLLayer(15, (50,), W_initialization="Xaviar",activation="leaky_relu", learning_rate=0.007, optimization="adaptive",regularization="L2")
opt = DLLayer(4, (15,), W_initialization="Xaviar",activation="softmax", learning_rate=0.007, optimization="adaptive",regularization="L2")  
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(opt)
model.compile("categorical_cross_entropy")

costs = model.train(train_x, Y_train, 2500)
model.save_weights(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Weights')
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_x) == Y_train))
print("test accuracy:", np.mean(model.predict(test_x) == Y_test))
#print(ImagePrediction(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Testing Examples','ClassicalImage.png'))
#print(ImagePrediction(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Testing Examples','JazzImage.png'))
#print(ImagePrediction(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Testing Examples','KpopImage.png'))
#print(ImagePrediction(r'C:\Users\Ziv Peltz\Desktop\Files for final project\Mfccs\Testing Examples','RockImage.png'))


prediction_train = model.predict(train_x).transpose()
prediction_test = model.predict(test_x).transpose()
#print(Y_train.shape,prediction_train.shape)
new_Y_train=Y_train.transpose()
new_Y_test=Y_test.transpose()
train_correct = sum([1 for i in range(new_Y_train.shape[0]) if np.array_equal(prediction_train[i],new_Y_train[i])])/new_Y_train.shape[0]
test_correct = sum([1 for i in range(new_Y_test.shape[0]) if np.array_equal(prediction_test[i],new_Y_test[i])])/new_Y_test.shape[0]
print("train acc:" + str(train_correct))
print("test acc:" + str(test_correct))



while False:
    name = input("Enter Song File: ")
    CheckImageGenre(r'C:\Users\Ziv Peltz\Downloads',name,r'C:\Users\Ziv Peltz\Downloads')


