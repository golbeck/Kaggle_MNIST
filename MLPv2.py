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
X_train0=dataset[:,1:]
#number of features
p0=X_train0.shape[1]

#find the mean and std dev of each feature
X_mean=X_train0.mean(0)
X_sd=X_train0.std(0)
#track which features will be kept
X_features=[]
#initialize counter for number of features retained
p=0
#determine which features have non-zero elements
for i in range(p0):
    if X_sd[i,]>0.0:
        X_features.append(i)
        p+=1

#create new array of retained features, with features demeaned and standardized
X_train=np.zeros((X_train0.shape[0],p))
for i in range(p):
    j=np.int(X_features[i])
    X_train[:,i]=(X_train0[:,j]-X_mean[j,])/X_sd[j,]


Y_train_array=dataset[:,0]
#convert output array to 0,1 matrix
K=Y_train_array.max(0)+1
Y_train=np.zeros((Y_train_array.shape[0],K))
for i in range(Y_train_array.shape[0]):
    Y_train[i,Y_train_array[i]]=1

#test data
X_test0=np.array(pd.read_csv("test.csv"))
#use the train set demeaning and standardization 
X_test=np.zeros((X_test0.shape[0],p))
for i in range(p):
    j=np.int(X_features[i])
    X_test[:,i]=(X_test0[:,j]-X_mean[j,])/X_sd[j,]
####################################################################################
####################################################################################
####################################################################################
#TRAINING PARAMETERS
####################################################################################
#####################################################################################
#####################################################################################
#number of observations used in each gradient update
batch_size=50
#hyperparameters
eps_alpha=0.1
#learning rate adjustment factor
eps_adj=0.40
#L2 regularization parameter
eps_penalty=0.0001
#momentum smoothing parameter is min( x[0]+(x[1]-x[0])*epoch/x[2], x[1])
mom_param=np.array([0.50,0.99,200.0])
#at each epoch, learning rate decreases by a factor of (1-gamma)
gamma=0.01
#number of neurons in each hidden layer
M=np.array([1500,800,300])
#number of hidden layers
n_layers=M.shape[0]
#append the number of output units to M
M=np.append(M,K)
#list of network parameters
#choose starting values from a uniform distribution with left and right endpoints specified by:
weight_L=-4*np.sqrt(6./(p+M[0]))
weight_H=4*np.sqrt(6./(p+M[0]))
#generate non-zero parmaeters
count_zero=0.0
while count_zero==0.0:
    alpha=[]
    #input parameters for first layer activation function
    alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(p+1)*M[0]).reshape(p+1,M[0]))
    #parameters for inputs to all other hidden layer activation functions and the output function (K units)
    for layer in range(1,n_layers+1):
        alpha.append(np.random.uniform(low=weight_L,high=weight_H,size=(M[layer-1]+1)*M[layer]).reshape(M[layer-1]+1,M[layer]))
    count_zero=np.array([np.abs(alpha[0]).min() for i in range(n_layers+1)]).sum()

#threshold for improvement of the error over the validation_freq that needs to be achieved to continue training
improvement_threshold=0.95
#number of epochs to train before measuring error rate on test test
validation_freq=10
#min number of epochs for training
min_epochs=10
#max number of epochs for training
max_epochs=250
#dropout probabilities
#keep most of the inputs (p>0.7) and use a lower probability for hidden units (p~0.5)
prob_dropout=np.append([1.0],np.ones(n_layers)*1.0)
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#number of observations
n=X_train.shape[0]
#number of features
p=X_train.shape[1]
#number of mini-batches
n_mini_batch=n/batch_size
#add bias vector to inputs
X_train=np.column_stack((np.ones(n),X_train))
#number of rows in the dependent variable
n_Y=Y_train.shape[0]
#number of classes
K=Y_train.shape[1]

#list of gradient-based updates to parameters
update_param=[]
#parameters for inputs to all other hidden layer activation functions and the output function (K units)
for layer in range(n_layers+1):
    update_param.append(np.zeros(alpha[layer].shape))


##################################################################################################
##################################################################################################
##################################################################################################
#TRAIN NETWORK
##################################################################################################
##################################################################################################
##################################################################################################
#set epochs to minimum number
epochs=np.copy(min_epochs)
#initialize epoch counter
epoch_iter=0
#save validation scores in order to determine improvement
#save train scores in order to determine improvement
train_scores=[]
#best train accuracy initially set to 0.0
train_loss_best=1.0
#loop through epochs and train classifier provided that epoch counter is low enough and test set error shows improvement
while epoch_iter<max_epochs:
    #update learning rate via annealing
    # eps_alpha*=1/(1+epoch_iter*gamma)
    eps_alpha*=(1-gamma)
    #update momentum rate (starts at mom_param[0] and increases linearly to mom_param[1] over mom_param[2] iterations, after which is stays fixed)
    mom_rate=min(mom_param[1],mom_param[0]+(mom_param[1]-mom_param[0])*epoch_iter/mom_param[2])

    print "epoch iteration %s" %epoch_iter
    #save rng state to apply the same permutation to both X and Y
    rng_state = np.random.get_state()
    #randomly permuate the features and outputs using the same shuffle for each epoch
    np.random.shuffle(X_train)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_train)        
    ##############################################################################################
    #iterate through the entire observation set, updating the gradient via mini-batches
    for batch_number in range(n_mini_batch):
        # print "batch number %s" %batch_number
        #feedforward operation: generate activations, activation gradients
        grad=range(n_layers+1)
        grad_act=[]
        Z=[]
        ##############################################################################################
        #input to first layer
        layer=0
        #observations used to update alpha, beta        
        obs_index=range(batch_number*batch_size,(batch_number+1)*batch_size)
        #dropout array for input layer (save separately from out dropout arrays as it is used in backprob)
        dropout_array_input=np.random.binomial(1,prob_dropout[layer],size=batch_size)
        #current inputs for mini-batch
        X_train_temp=np.copy(X_train[obs_index,:])
        #only apply dropout to non-bias input units
        X_train_temp[:,range(1,p)]*=dropout_array_input[:,None]
        #linear combination of inputs
        T=np.dot(X_train_temp,alpha[layer])
        #dropout array
        dropout_array=np.random.binomial(1,prob_dropout[layer+1],size=batch_size)
        #gradient of activation function with respect to T
        grad_act.append(grad_sigmoid(T)*dropout_array[:,None])
        #add bias vector to hidden layer; dim (batch_size,M[0]+1)
        Z.append(np.column_stack((np.ones(batch_size),special.expit(T)*dropout_array[:,None])))

        for layer in range(1,n_layers):
            #linear combination of inputs
            T=np.dot(Z[layer-1],alpha[layer])
            #dropout array
            dropout_array=np.random.binomial(1,prob_dropout[layer+1],size=batch_size)
            #gradient of activation function with respect to T
            grad_act.append(grad_sigmoid(T)*dropout_array[:,None])
            #add bias vector to hidden layer 
            Z.append(np.column_stack((np.ones(batch_size),special.expit(T)*dropout_array[:,None])))

        #output of last hidden layer
        layer=n_layers
        #linear combination of hidden layer outputs
        T=np.dot(Z[layer-1],alpha[layer])
        #gradient of output function
        grad_softmax_fn=grad_softmax(T)
        #outputs
        g=softmax_fn(T)
        ##############################################################################################
        #backpropagation
        #outer layer (Y/g)*gradient of output (summed over classes)
        B_old=np.einsum('ij,ij,ijk->ik',Y_train[obs_index,:],1.0/g,grad_softmax_fn)
        #multiply by outer layer activations to obtain gradient and sum over all observations
        grad[layer]=-np.einsum('ij,ik->jk',Z[layer-1],B_old)/np.float(batch_size)

        for layer in range(n_layers,1,-1):
            B_old=np.einsum('ij,kj,ik->ik',B_old,alpha[layer][range(1,M[layer-1]+1),:],grad_act[layer-1])
            #multiply by activations of the layer to obtain gradient and sum over all observations
            grad[layer-1]=-np.einsum('ij,ik->jk',Z[layer-2],B_old)/np.float(batch_size)

        #input layer gradient
        layer=1
        B_old=np.einsum('ij,kj,ik->ik',B_old,alpha[layer][range(1,M[layer-1]+1),:],grad_act[layer-1])
        #multiply by inputs to obtain gradient and sum over all observations 
        grad[layer-1]=-np.einsum('ij,ik->jk',X_train_temp,B_old)/np.float(batch_size)
        ##############################################################################################
        #gradient descent updates with momentum smoothing
        for i in range(n_layers+1):
            #L2 regularization term
            M_temp=alpha[i].shape[0]
            M_range=range(1,M_temp)
            #use momentum smoothing for the updates
            update_param[i][M_range,:]=mom_rate*update_param[i][M_range,:]-eps_alpha*(grad[i][M_range,:]+eps_penalty*alpha[i][M_range,:])
            #apply smoothed update
            alpha[i][M_range,:]=alpha[i][M_range,:]+update_param[i][M_range,:]
            #no regularization for bias parameters
            update_param[i][0,:]=mom_rate*update_param[i][0,:]-eps_alpha*(grad[i][0,:])
            #apply smoothed update
            alpha[i][0,:]=alpha[i][0,:]+update_param[i][0,:]

    #update epoch iteration
    epoch_iter+=1
    ##############################################################################################
    ##############################################################################################
    #training set accuracy for previous epoch (before weights are updated using the gradient)
    y_pred_train=NN_classifier(alpha*prob_dropout,X_train,special.expit,softmax_fn)
    y_pred_train=y_pred_train.argmax(1)+1
    #convert class matrix to array of class labels (starting at 1) for use in confusion matrix
    y_dat_train=Y_train.argmax(1)+1
    #compute confusion matrix using predicted outputs (y_pred) and actual labels (y_dat)
    CF=confusion_matrix_multi(y_pred_train,y_dat_train,K)
    # print CF
    accuracy=CF.diagonal().sum(0)/y_pred_train.shape[0]
    train_scores.append(accuracy)
    print "training set accuracy rate %s" %accuracy
    ##############################################################################################
    ##############################################################################################
    #predict classes and compute accuracy rate on test set
    #predicted class for test set
    #weights are multiplied by dropout probability
    #if enough epochs have been iterated, through test if test set error has decreased sufficiently
    if (epoch_iter>=min_epochs) and (epoch_iter % validation_freq==0):
        #average train error over current set
        avg_loss=1.0-np.array(train_scores).mean()
        print "average loss: %s" %avg_loss
        #threshold for train error to pass below in order to continue training
        loss_threshold=improvement_threshold*train_loss_best
        print "threshold loss %s" %loss_threshold
        #test if loss has not decreased sufficiently, such that we can stop training
        if avg_loss>loss_threshold:
            eps_alpha*=eps_adj
        #update best test set loss
        train_loss_best=min(train_loss_best,1.0-np.array(train_scores).max())
        #new list of test set scores
        train_scores=[]
        print "best train set loss %s" %train_loss_best
        print "learning rate %s" %eps_alpha
##############################################################################################
##############################################################################################
##############################################################################################
X_test=np.column_stack((np.ones(X_test.shape[0]),X_test))
y_pred_test=NN_classifier(alpha*prob_dropout,X_test,special.expit,softmax_fn)
y_pred_test=y_pred_test.argmax(1).reshape((y_pred_test.shape[0],1))

columns = ['ImageId', 'Label']
index = range(1,y_pred_test.shape[0]+1) # array of numbers for the number of samples
df = pd.DataFrame(columns=columns)
df['ImageId']=index
df['Label']=y_pred_test

df.head(10)
df.to_csv("test_predictions1500x800x300.csv",sep=",",index=False)