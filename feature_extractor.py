import tarfile
import pickle
import requests
from io import BytesIO
import os

# Directory path
directory = 'cifar-10-batches-py'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

# Download the .tar.gz file
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
response = requests.get(url)
tar_bytes = BytesIO(response.content)

# Extract the contents of the .tar.gz file
tar = tarfile.open(fileobj=tar_bytes, mode="r:gz")
tar.extractall()
tar.close()

# Assuming the .tar.gz file contains pickled data files, you can now unpickle them
d = unpickle('cifar-10-batches-py/data_batch_1')
print(d[b'data'].shape)

# List contents of the directory
contents = os.listdir(directory)

stored_data = {}
for i in range(10000):
    stored_data[i] = {}
    stored_data[i][b'labels'] = d[b'labels'][i]
    stored_data[i][b'R'] = d[b'data'][i].reshape(96,32)[0:32]
    stored_data[i][b'G'] = d[b'data'][i].reshape(96,32)[32:64]
    stored_data[i][b'B'] = d[b'data'][i].reshape(96,32)[64:96]
# Print the first stored dictionary as an example
print("Example of stored dictionary for row 0:")
print(stored_data[0])
print(stored_data[0][b'R'].shape)
# Assuming 'stored_data' contains the stored dictionaries

# Iterate over the keys of the first stored dictionary to detect potential keys containing image paths
for key in stored_data[0].keys():
    # Check if the value corresponding to the key is a string (which might be a file path)
    if isinstance(stored_data[0][key], str):
        print(f"Potential image path key: {key}")