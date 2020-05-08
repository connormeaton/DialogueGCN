import pickle

path = 'IEMOCAP_features.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f, encoding="latin1")

print(data)