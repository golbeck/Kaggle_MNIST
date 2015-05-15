####################################################################################
####################################################################################
import numpy as np
from scipy import special
import os
import pandas as pd

####################################################################################
####################################################################################
def NN_classifier(alpha,X_in,activation_fn,output_fn):
    #negative log likelihood function for classification
    #alpha: list of arrays of weights
    #X_in: np.array of inputs; dim (n,p+1) (each of the n rows is a separate observation, p is the number of features)
    #activation_fn: activation function
    #output_fn: output function
    
    #number of observations
    n=X_in.shape[0]
    #number of hidden layers
    n_layers=len(alpha)-1
    layer=0
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),activation_fn(np.dot(X_in,alpha[layer]))))

    for layer in range(1,n_layers):
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),activation_fn(np.dot(Z,alpha[layer]))))

    ##############################################################################################
    #output of last hidden layer
    layer=n_layers
    #outputs
    g=output_fn(np.dot(Z,alpha[layer]))
    return g
####################################################################################
####################################################################################
def cost_fn(eps_penalty,alpha,X_in,Y,activation_fn,output_fn):
    #negative log likelihood function for classification
    #eps_penalty: parameter for L2 regularization penalty term
    #alpha: np.array of weights for inputs; dim (p+1,M)
    #X_in: np.array of inputs; dim (n,p+1) (each of the n rows is a separate observation, p is the number of features)
    #Y: np.array of outputs; dim (n,K) (K is the number of classes)
    #activation_fn: activation function
    #output_fn: output function
    
    #number of observations
    n=X_in.shape[0]
    #number of hidden layers
    n_layers=len(alpha)-1
    layer=0
    #add bias vector to hidden layer; dim (batch_size,M[0]+1)
    Z=np.column_stack((np.ones(n),activation_fn(np.dot(X_in,alpha[layer]))))

    for layer in range(1,n_layers):
        #add bias vector to hidden layer 
        Z=np.column_stack((np.ones(n),activation_fn(np.dot(Z,alpha[layer]))))

    ##############################################################################################
    #output of last hidden layer
    layer=n_layers
    #outputs
    g=output_fn(np.dot(Z,alpha[layer]))

    #L2 regularization penalty term
    L2=0.5*eps_penalty*np.array([(alpha[i]**2).sum() for i in range(len(alpha))]).sum()
    #generate negative log likelihood function
    f=-(np.log(g)*Y).sum()+L2
    return f
####################################################################################
####################################################################################
def softmax_fn(T):
    #T: linear combination of hidden layer outputs
    #determine how many columns (classes) for the output
    K=T.shape[1]
    #iterate over each class to generate dividend matrix
    g=np.exp(T)
    #compute the denominator of the softmax function
    temp=g.sum(axis=1)
    #divide each column by the sum
    g=np.einsum('ij,i->ij',g,1/temp)
    #return a matrix of dim (n,K), where "n" is the number of observations
    return g

####################################################################################
####################################################################################
def grad_sigmoid(X):
    #X: linear combination of features
    g=special.expit(X)*(1-special.expit(X))
    return g
####################################################################################
####################################################################################
def grad_softmax(T):
    #T: linear combination of hidden layer outputs, dim (n,K)
    #number of observations in T
    n=T.shape[0]
    #determine how many columns (classes) for the output
    K=T.shape[1]
    #iterate over each class to generate dividend matrix
    g=np.exp(T)
    #compute the denominator of the softmax function
    temp=g.sum(axis=1)
    #divide each column by the sum
    g=np.einsum('ij,i->ij',g,1/temp)
    #generate a dim (n,K,K) matrix for derivatives of the softmax function with respect to T[i,l]
    f1=np.einsum('ij,ik->ijk',g,g)
    A=np.eye(K)
    f2=np.einsum('ij,jk->ijk',g,A)
    #matrix of derivatives: for each observation i, 
    #the derivative of the kth column of the softmax function
    #with respect to the jth output
    g=f2-f1
    return g
####################################################################################
####################################################################################  
def confusion_matrix_multi(y_out,y,n_class):
    #compute logistic function
    m=y.shape[0]
    tempTP=0
    tempTN=0
    tempFP=0
    tempFN=0
    
    #rows: actual class label
    #cols: predicted class label
    CF=np.zeros((n_class,n_class))
    
    for i in range(m):
        if(y_out[i]==y[i]):
            CF[y[i]-1,y[i]-1]+=1
        else:
            CF[y[i]-1,y_out[i]-1]+=1
            
    return CF        
####################################################################################
####################################################################################
####################################################################################
####################################################################################
####################################################################################
#GET INPUTS
####################################################################################
####################################################################################
####################################################################################
pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
dir1='/home/golbeck/Workspace/Kaggle_MNIST'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

#load dataset
dataset=np.array(pd.read_csv("train.csv"))
Y_train_array=dataset[:,0]
#convert output array to 0,1 matrix
K=Y_train_array.max(0)+1
Y_train=np.zeros((Y_train_array.shape[0],K))
for i in range(Y_train_array.shape[0]):
    Y_train[i,Y_train_array[i]]=1


X_in=dataset[:,1:]
n=X_in.shape[0]
X_image=np.zeros((n,28,28))
for j in range(28):
	for k in range(28):
		l=j*28+k
		X_image[:,j,k]=X_in[:,l]




