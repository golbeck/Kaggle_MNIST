import pandas as pd
import numpy as np
import os
pwd_temp=os.getcwd()
# dir1='/home/sgolbeck/workspace/PythonExercises/NeuralNets'
dir1='/home/golbeck/Workspace/Kaggle_MNIST'
dir1=dir1+'/data' 
if pwd_temp!=dir1:
    os.chdir(dir1)

#load dataset
majority_vote=np.array(pd.read_csv("majority_vote.csv"))

from scipy.stats import mode
val, count = mode(majority_vote[:,1:5], axis=1)



columns = ['ImageId', 'Label']
index = range(1,28001) # array of numbers for the number of samples

df = pd.DataFrame(columns=columns)
df['ImageId']=index
df['Label']=val
df.Label = df.Label.astype(int)
df.head(10)
df.to_csv("majority_vote1.csv",sep=",",index=False)