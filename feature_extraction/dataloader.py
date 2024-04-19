import tarfile
import pickle
import requests
from io import BytesIO
import os

#directory path
directory = 'cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

#download the .tar.gz file
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
response = requests.get(url)
tar_bytes = BytesIO(response.content)

#extracting the .tar.gz file
tar = tarfile.open(fileobj=tar_bytes, mode="r:gz")
tar.extractall()
tar.close()

#unpickling
d = unpickle('cifar-10-batches-py/data_batch_1')
print(d[b'data'].shape)

contents = os.listdir(directory)

stored_data = {}
#reshaping the row major image to a matrix
for i in range(10000):
    stored_data[i] = {}
    stored_data[i][b'labels'] = d[b'labels'][i]
    stored_data[i][b'R'] = d[b'data'][i].reshape(96,32)[0:32]
    stored_data[i][b'G'] = d[b'data'][i].reshape(96,32)[32:64]
    stored_data[i][b'B'] = d[b'data'][i].reshape(96,32)[64:96]

