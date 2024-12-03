import pickle

a = pickle.load(open("performance_results.pkl", "rb"))

print(a.shape)