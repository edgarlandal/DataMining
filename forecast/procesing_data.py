import numpy as np
from sklearn.model_selection import train_test_split

def separator(data, n_outs):
    inputs = []
    targets = []
    for i in range(n_outs):    
        targets.append(data[:,-(i+1)])
    targets = np.asarray(targets).T
    
    ins = data.shape[1] - n_outs
    for i in range(ins):
        inputs.append(data[:,i])
    inputs = np.asarray(inputs).T

    return inputs, targets
                        #4    2      2      1
def create_window(data, ins, outs, s_in, s_out):
    train = data.to_numpy();
    q = train.shape[0]
    array = []
    data = []
    for i in range(q):
        max_in = (ins*2)+s_in
        max_out = outs*(s_out + 1)
        if(q <= i+(max_in)+(max_out) + 1):
            break;
        j = 0
        while(len(array) != (outs+ins)):
            if(j == 0):
                array.append(float(train[i + j]))
            else:
                if(j < ((ins+(s_in*2)))):
                    j = j + s_in
                    array.append(float(train[i + j]))
                else:
                    j = j + s_out
                    array.append(float(train[i + j]))
            j+=1
        data.append(array)
        array = []
    data = np.asarray(data)
    return data

def separate_data(X, Y, train): 
    #X_train, X_test, Y_train, Y_test
    if((train >= 1.0)):
        return X, X, Y, Y
    return train_test_split(X, Y, train_size= train, shuffle = False) 
