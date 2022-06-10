import numpy as np
import random
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
#import matplotlib.pyplot as plt
import warnings
import time
import numba as nb
from numba import jit,guvectorize,set_num_threads
import os

dirName = 'Committor_Files'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")

def RelEnt(pred,prob):
    res = 0.
    count_0 = 0
    count_1 = 0
    for i in range(len(pred)):
        if(prob[i]<=0.):
            if(pred[i]==1):
                count_1 += 1
            res += pred[i]*np.log(10**(-14))+(1-pred[i])*np.log(1-prob[i])
        elif(prob[i]>=1.):
            if(pred[i]==0):
                count_0 += 1
            res += pred[i]*np.log(prob[i])+(1-pred[i])*np.log(10**(-14))
        else:
            res += pred[i]*np.log(prob[i])+(1-pred[i])*np.log(1-prob[i])
    print(count_1,count_0)
    return -res/len(pred)

global_time = time.time()

Temp = np.load('France_Anomalies_LONG_JJA_CG3D.npy') #Temperature France

T = np.load('Neighbors_Matrix_1000NN.npy') #Matrix analogues

thershold = np.array([2.6, 3.2, 3.6]) #Threshold defining committor

delay = np.array([0,3,6,9,12,15])  #time prediction

num_year = 1000

num_day_year = 90

Labels = np.zeros((len(thershold),num_year*num_day_year,len(delay)),dtype = 'int')

for i in range(len(thershold)):
    Labels[i] = np.load('Label%s.npy'%(str(i))) #File containg labels for each delay


#definition test data-sets
new_permutation = True

if(new_permutation==True):

    permutation = np.zeros((10,100),dtype = 'int')

    for i in range(10):
        permutation[i] = range(100*i,100*(i+1))

    for i in range(9):
        for j in range(i+1,10):
            if((permutation[i]==permutation[j]).all()==True):
                print('ERROR:Two identical dataset')
else:
    permutation = np.load('Perm1.npy')

permutation = permutation.astype(int)

print(len(permutation))

neighbors = [10]
#neighbors = [150,180,220,250,300,400]
#neighbors = [205,210,215,225,230]

num_Traj = 10000 #number of trajectories for monte-carlo sampling of committor

step_Traj = 5  #number of step in the markov chain for 15 days.

#Function for computing the committor at one point. Input: day=point where committor is computed (state of markov chain), ther= vector of threshold, dela= vector of delays, nn= number of neighbors, Matr=Matix which contains indeces of Markov chain, res= vector where results are stored
@guvectorize([(nb.int64,nb.float64[:],nb.int64[:],nb.int64,nb.int64[:,:],nb.float64[:,:])],'(),(n),(m),(),(l,j)->(n,m)',nopython=True,target="parallel")
def CommOnePoint(day,ther,dela,nn,Matr,res):
    z = np.zeros((len(ther),len(dela))) #auxiliary variable (result)
    key_day = 0     #parameter for initial condition
    for i in range(num_Traj):
        if(key_day>0):
            s_0 = day
        else:
            app = random.randint(0,nn-1)
            s_0 = Matr[day][app]
        
        A = np.zeros((len(ther),len(dela))) #auxiliary variable (integrated temperature)
        
        s = s_0
        for j in range(step_Traj+np.max(delay)//3):
            app = random.randint(0,nn-1) #analogue selection
            s = Matr[s][app] + 3         #evolution state s
            for l_2 in range(len(dela)):
                if(l_2==0):
                    if(j<step_Traj):
                        for l_1 in range(len(ther)):
                            A[l_1][l_2] += Temp[s]
                else:
                    if(j>=dela[l_2]//3 and j<dela[l_2]//3+step_Traj):
                        for l_1 in range(len(ther)):
                            A[l_1][l_2] += Temp[s]
        A = A / 5.
        
        #Check if A>a
        for l_1 in range(len(ther)):
            for l_2 in range(len(dela)):
                if(A[l_1][l_2]>ther[l_1]):
                    z[l_1][l_2] += 1.
    #fill res vector
    for l_1 in range(len(ther)):
        for l_2 in range(len(dela)):
            res[l_1][l_2] = z[l_1][l_2] / num_Traj
    

y = 0

M = T.copy()

#Compilation CommOnePoint Function
start_time = time.time()
nn = neighbors[0]
q_1 = CommOnePoint(y*90+89,thershold,delay,nn,M)
#print(y*90+89)
print(q_1[0][0])
print('--- %s seconds ---'%(time.time() - start_time))

#Compilation of vectorized version CommOnePoint Function
start_time = time.time()
days = np.array(range(y*90,(y+1)*90))
q = CommOnePoint(days,thershold,delay,nn,M)
print(q.shape)
print(q[len(q)-1][0][0])
print('--- %s seconds ---'%(time.time() - start_time))



#Function for removing states of test-data from matrix of analogue. Input: num_row=number of states, dum_vec = auxiliary vector for dimension, nn= number of neighbors, num_perm= number of test data-set (from 0 to 9),  new_row= vector where results are stored
@guvectorize([(nb.int64,nb.int64[:],nb.int64,nb.int64,nb.int64[:])],'(),(n),(),()->(n)',nopython=True,target="parallel")
def CreateMatrixForTest(num_row,dum_vec,nn,num_perm,new_row):
    count = 0
    place_holder = []
    while(len(place_holder)<nn and count<len(T[num_row])):
        if(T[num_row][count]//num_day_year not in permutation[num_perm]):
            place_holder.append(T[num_row][count])
        count +=1
    if(len(place_holder)!=nn):
        print('Error: Not enough neighbors')
    new_row[:] = np.array(place_holder)

#Compilation CreateMatrixForTest Function
Test_M = CreateMatrixForTest(0,np.zeros(nn,dtype='int'),nn,0)

#Function for filling labels for test data-set.
@guvectorize([(nb.int64,nb.float64[:],nb.int64[:],nb.int64[:,:])],'(),(n),(m)->(n,m)',nopython=True,target="parallel")
def Fill_True_Labels(day,ther,dela,res):
    for l_1 in range(len(ther)):
        for l_2 in range(len(dela)):
            res[l_1][l_2] = (Labels[l_1][day][l_2])


#Vectors for storing scores (B=brier, S=cross_entropy)
B = np.zeros((len(permutation),len(neighbors),len(thershold),len(delay)))
S = np.zeros((len(permutation),len(neighbors),len(thershold),len(delay)))

#compute committor for the 10 different test data-sets
for perm in range(len(permutation)):
    for k in range(len(neighbors)): #if one need to test different number of analogues
        start_time = time.time()
        M = CreateMatrixForTest(range(len(T)),np.zeros(neighbors[k],dtype='int'),neighbors[k],perm) #Matrix with analogues
        
        count = 0
        print('I am here')
        days = np.array([np.array(range(y*90,(y+1)*90)) for y in permutation[perm]]).reshape(-1) #points where committor is computed
        q = CommOnePoint(days,thershold,delay,neighbors[k],M) #committor computation
        print('I computed the committor')
        true_label = Fill_True_Labels(days,thershold,delay) #fill labels for computing scores
        np.save('Committor_Files/Committor_Analogues%s_Perm_%s.npy'%(str(neighbors[k]),str(perm)),q)
        #scores computation
        for l_1 in range(len(thershold)):
            for l_2 in range(len(delay)):
                B[perm][k][l_1][l_2] = brier_score_loss(true_label[:,l_1,l_2],q[:,l_1,l_2])
                S[perm][k][l_1][l_2] = RelEnt(true_label[:,l_1,l_2],q[:,l_1,l_2])
        print('I computed the scores')


#print scores 
Brier_File = open('Brier.dat','a')
Entropy_File = open('Entropy.dat','a')
B_mean = np.mean(B,axis=0)
B_std = np.std(B,axis=0)/np.sqrt(len(permutation))
S_mean = np.mean(S,axis=0)
S_std = np.std(S,axis=0)/np.sqrt(len(permutation))
print(B_mean,S_mean)
for k in range(len(neighbors)):
    for l_1 in range(len(thershold)):
        for l_2 in range(len(delay)):
            Brier_File.write('%d\t%lf\t%d\t%lf\t%lf\n'%(neighbors[k],thershold[l_1],delay[l_2],B_mean[k][l_1][l_2],B_std[k][l_1][l_2]))
            Entropy_File.write('%d\t%lf\t%d\t%lf\t%lf\n'%(neighbors[k],thershold[l_1],delay[l_2],S_mean[k][l_1][l_2],S_std[k][l_1][l_2]))
            Brier_File.flush()
            Entropy_File.flush()
Brier_File.close()
Entropy_File.close()

        
