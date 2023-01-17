import numpy as np
import numba as nb
from numba import jit,guvectorize,set_num_threads

Labels = np.random.randint(20, size=(2,2,3))

print(Labels)
print(f'{Labels.shape = }')

#Function for filling labels for test data-set. Input: day=point to be tested, ther= vector of threshold, dela= vector of delays, res=vector for results
@guvectorize([(nb.int64,nb.float64[:],nb.int64[:],nb.int64[:,:])],'(),(n),(m)->(n,m)',nopython=True,target="parallel")
def Fill_True_Labels(day,ther,dela,res):
    print("day = ",day)
    for l_1 in range(len(ther)):
        for l_2 in range(len(dela)):
            res[l_1][l_2] = (Labels[l_1][l_2][day])    #fill the vector with the labels of day at different tau and

true_label = Fill_True_Labels(0,np.array([2,3]),np.array([4,5])) #fill labels for computing scores
print(f'{true_label}')

true_label = Fill_True_Labels(1,np.array([2,3]),np.array([4,5])) #fill labels for computing scores
print(f'{true_label}')

true_label = Fill_True_Labels([0,1,2],np.array([2,3]),np.array([4,5])) #fill labels for computing scores
print(f'{true_label}')