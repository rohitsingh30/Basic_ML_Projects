import os 
from pathlib import Path 											# for path iteration
from keras.preprocessing import image 								# for image processing
import matplotlib.pyplot as plt 
import numpy as np

p = Path("./Dataset/")   											# openging the Dataset folder

dirs = p.glob("*") 													# iterating over the whole folder in Datasets 


image_data = []
labels = []

#Optional
image_paths = []

label_dict = {"Pikachu":0,"Bulbasaur":1,'Meowth':2,'Abra':3}  							 # labeling the  pokemon str to integers
label2pokemon = {0:"Pikachu",1:"Bulbasaur",2:'Meowth',3:'Abra'}


for folder_dir in dirs:												# folder dir in Datasets
    label = str(folder_dir).split('/')[-1]                        # extracting the name of pokemon from folder name
    print(label)
    
    cnt = 0
    print(folder_dir)
    
    #Iterate over folder_dir and pick all images of the pokemen
    for img_path in folder_dir.glob("*.jpg"):						# now iterating over each image in the folder of folder_dir
        img = image.load_img(img_path,target_size=(40,40))			# loading the image and deciding the sie of each image fixed
        img_array = image.img_to_array(img)							# converting each iage into the array
        image_data.append(img_array)								# append the image array in image_data array 
        labels.append(label_dict[label])							# also appendign label of each image in labels array
        cnt += 1													# also keeping the count of no of images of each pokemon 
        
    print(cnt)



#visualiztion
print(len(image_data))
print(len(labels))
#Randomly see the images 
import random
random.seed(10)
X = np.array(image_data)
Y = np.array(labels)


from sklearn.utils import shuffle
X,Y = shuffle(X,Y,random_state=2)

#Normalisation
X = X/255.0  # normalisation of image data
# Draw some pokemons
def drawImg(img,label):
    plt.title(label2pokemon[label])
    # plt.imshow(img)
    # plt.show()

for i in range(1,20):
	drawImg(X[i].reshape(40,40,3),Y[i])									# resizing image as they reduce accuracy of model if 100*100

### Create Training and Testing Set
X_ = np.array(X)
Y_ = np.array(Y)

#Training Set
X = X_[:300,:]
Y = Y_[:300]

#Test Set
XTest = X_[300:,:]
YTest = Y_[300:]

print(X.shape,Y.shape)
print(XTest.shape,YTest.shape)

# our Neural Network


class NeuralNetwork:
    
    def __init__(self,input_size,layers,output_size):
        np.random.seed(0)
        
        model = {} #Dictionary
        
        #First Layer
        model['W1'] = np.random.randn(input_size,layers[0])
        model['b1'] = np.zeros((1,layers[0]))
        
        #Second Layer
        model['W2'] = np.random.randn(layers[0],layers[1])
        model['b2'] = np.zeros((1,layers[1]))
        
        #Third/Output Layer
        model['W3'] = np.random.randn(layers[1],output_size)
        model['b3'] = np.zeros((1,output_size))
        
        self.model = model

        self.activation_outputs = None
    
    def forward(self,x):
        
        W1,W2,W3 = self.model['W1'],self.model['W2'],self.model['W3']
        b1, b2, b3 = self.model['b1'],self.model['b2'],self.model['b3']
        
        z1 = np.dot(x,W1) + b1
        a1 = np.tanh(z1) 
        
        z2 = np.dot(a1,W2) + b2
        a2 = np.tanh(z2)
        
        z3 = np.dot(a2,W3) + b3
        y_ = softmax(z3)
        
        self.activation_outputs = (a1,a2,y_)
        return y_
        
    def backward(self,x,y,learning_rate=0.001):
        W1,W2,W3 = self.model['W1'],self.model['W2'],self.model['W3']
        b1, b2, b3 = self.model['b1'],self.model['b2'],self.model['b3']
        m = x.shape[0]
        
        a1,a2,y_ = self.activation_outputs
        
        delta3 = y_ - y
        dw3 = np.dot(a2.T,delta3)
        db3 = np.sum(delta3,axis=0)
        
        delta2 = (1-np.square(a2))*np.dot(delta3,W3.T)
        dw2 = np.dot(a1.T,delta2)
        db2 = np.sum(delta2,axis=0)
        
        delta1 = (1-np.square(a1))*np.dot(delta2,W2.T)
        dw1 = np.dot(X.T,delta1)
        db1 = np.sum(delta1,axis=0)
        
        
        #Update the Model Parameters using Gradient Descent
        self.model["W1"]  -= learning_rate*dw1
        self.model['b1']  -= learning_rate*db1
        
        self.model["W2"]  -= learning_rate*dw2
        self.model['b2']  -= learning_rate*db2
        
        self.model["W3"]  -= learning_rate*dw3
        self.model['b3']  -= learning_rate*db3
        
        # :)
        
    def predict(self,x):
        y_out = self.forward(x)
        return np.argmax(y_out,axis=1)
    
    def summary(self):
        W1,W2,W3 = self.model['W1'],self.model['W2'],self.model['W3']
        a1,a2,y_ = self.activation_outputs
        
        print("W1 ",W1.shape)
        print("A1 ",a1.shape)

def softmax(a):
    e_pa = np.exp(a) #Vector
    ans = e_pa/np.sum(e_pa,axis=1,keepdims=True)
    return ans        

def loss(y_oht,p):
    l = -np.mean(y_oht*np.log(p))
    return l

def one_hot(y,depth):
    print('look here')
    print(y)
    m = y.shape[0]
    print(m)
    y_oht = np.zeros((m,depth))
    # print(y_oht)
    # print( 'look here')
    # print(np.arange(m))
    # print(y_oht[np.arange(m),y])
    y_oht[np.arange(m)] = [1]
    y_oht[y] = [1]
    print(y_oht)
    return y_oht

def train(X,Y,model,epochs,learning_rate,logs=True):
	training_loss = []

	classes = 2
	Y_OHT = one_hot(Y,classes)

	for ix in range(epochs):

		Y_ = model.forward(X)
		l = loss(Y_OHT,Y_)

		model.backward(X,Y_OHT,learning_rate)
		training_loss.append(l)
		if(logs and ix%50==0):
			print("Epoch %d Loss %.4f"%(ix,l))

	return training_loss

# training the mmdoel
model = NeuralNetwork(input_size=4800,layers=[100,50],output_size=2)    # 4800 

print(X.shape)
X = X.reshape(X.shape[0],-1)
print(X.shape)

XTest = XTest.reshape(XTest.shape[0],-1)
print(XTest.shape)


l = train(X,Y,model,500,0.0002)

plt.style.use("dark_background")
plt.title("Training Loss vs Epochs")
plt.plot(l)

plt.show()


# Accuracy 
def getAccuracy(X,Y,model):
    outputs = model.predict(X)
    acc = np.sum(outputs==Y)/Y.shape[0]
    return acc,outputs
    
print("Train Acc %.4f"%getAccuracy(X,Y,model))
print("Test Acc %.4f"%getAccuracy(XTest,YTest,model))


a,outputs=getAccuracy(X,Y,model)

from sklearn.metrics import confusion_matrix
from visualize import plot_confusion_matrix
cnf_matrix = confusion_matrix(outputs,Y)
print(cnf_matrix)
cnf_matrix = confusion_matrix(outputs,Y)
print(cnf_matrix)
cnf_matrix = confusion_matrix(outputs,Y)
print(cnf_matrix)
cnf_matrix = confusion_matrix(outputs,Y)
print(cnf_matrix)
for i in range(Y.shape[0]):
    if Y[i] != outputs[i]:
        drawImg(X[i].reshape(100,100,3),Y[i])
        print("Prediction %d %s"%(i,label2pokemon[outputs[i]]))