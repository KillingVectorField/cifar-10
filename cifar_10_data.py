import pickle
import numpy as np

batch_size=100

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

batch_labels=np.array([],dtype=np.int8)
batch_data=np.empty(shape=[0,32*32*3],dtype=np.uint8)
for i in range(1,6):
    batch=unpickle(r'./cifar-10-batches-py/data_batch_'+str(i))
    batch_labels=np.concatenate((batch_labels,batch[b'labels']),axis=0)
    batch_data=np.concatenate((batch_data,batch[b'data']),axis=0)

label_names=unpickle(r'./cifar-10-batches-py/batches.meta')[b'label_names']

batch_data=batch_data.reshape(-1,3,32,32)
batch_data=np.rollaxis(batch_data,axis=1,start=4)#将channel移到最后一维，即NHWC

n_data=len(batch_labels)#50000个
n_batches=int(np.ceil(n_data /batch_size))#500个

test_batch=unpickle(r'./cifar-10-batches-py/test_batch')
test_labels=test_batch[b'labels']
test_data=test_batch[b'data'].reshape(-1,3,32,32)
test_data=np.rollaxis(test_data,axis=1,start=4)#NHWC

def fetch_batch(batch_index, batch_size):
    '''load the data from disk'''
    X_batch=batch_data[batch_index*batch_size:(batch_index+1)*batch_size,]
    y_batch=batch_labels[batch_index*batch_size:(batch_index+1)*batch_size]
    return X_batch, y_batch