import scipy.io as spio
mat = spio.loadmat('D1.mat', squeeze_me=True) 
d = mat['d']
Index = mat['Index'] #Index not in order
Class = mat['Class']

from scipy.stats import median_absolute_deviation
import pandas as pd
from scipy.signal import butter,filtfilt
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler 
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io

test_index = Index[2000:]


##########

#FILTER SIGNAL

#use butterworth filter to filter signal d
def butter_lowpass_filter(d, low_cutoff, high_cutoff, fs, order):
    b, a = butter(order, Wn= [low_cutoff,high_cutoff], btype='bandpass', analog=False, fs=fs)
    y = filtfilt(b, a, d)
    return y

sf = butter_lowpass_filter(d, 100, 1500, 25000, 2)

##########

#EXTRACT SPIKES (PANDAS)

# Extract the 60 voltage datapoints after an index (spike voltage) for every Index in d
spike_voltages = [sf[i:i+60] for i in (Index)]
# Create a dataframe with the list of spike voltages for each spike
df_spike_voltages = pd.DataFrame (spike_voltages) 
# Create a datframe with the index vector and its corresponding class vector
df = pd.DataFrame({'Index': Index, 'Class': Class}) 
# Combine both dataframes to create the training sample
df= pd.concat([df, df_spike_voltages], axis=1)


#to get rid of index column df = df.drop("Index", axis = 1)

print(df)

df.to_csv("extracted_spikes_pandas.csv")

###########

# PCA + KNN (to obtain class)

#load train data
df = pd.read_csv("extracted_spikes_pandas.csv")
test = df.to_numpy()
np.savetxt("extracted_spikes.csv",test, delimiter=",")


# Load the train and test MNIST data
train = np.loadtxt('extracted_spikes.csv', delimiter=',') 
test = np.loadtxt('extracted_spikes.csv', delimiter=',')

# Separate labels from training data
train_data = train[:2000, 3:] #from the 2nd column onwards (datapoints)
train_labels = train[:2000, 2] #the 1st column (class)
test_data = test[2000:, 3:] 
test_labels = test[2000:, 2]



# Select number of components to extract
pca = PCA(n_components = 10)
# Fit to the training data
pca.fit(train_data)

# Determine amount of variance explained by components
print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))


#KNN starts
# Extract the principal components from the training data
train_ext = pca.fit_transform(train_data)
# Transform the test data using the same components
test_ext = pca.transform(test_data)

# Normalise the data sets
min_max_scaler = MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_ext) 
test_norm = min_max_scaler.fit_transform(test_ext)

# Create a KNN classification system with k = 5 
# Uses the p2 (Euclidean) norm
knn = KNeighborsClassifier(n_neighbors=3, p=2) 
knn.fit(train_norm, train_labels)

# Feed the test data in the classifier to get the predictions
pred_class = knn.predict(test_norm) #variable 'pred' is the predicted classes (ie. the pred_class)

# Check how many were correct
scorecard = []

for i, sample in enumerate(test_data):
    # Check if the KNN classification was correct 
    if round(pred_class[i]) == test_labels[i]:
        scorecard.append(1) 
    else:
        scorecard.append(0) 
    pass

# Calculate the performance score, the fraction of correct answers
scorecard_array = np.asarray(scorecard)
print ("Predicted Class", (pred_class[:5]))
print ("Test Labels", (test_labels[:5]))
print("Performance = ", (scorecard_array.sum() / scorecard_array.size) * 100, ' % ')

D2 = {'d': d, 'Index': test_index, 'Class': pred_class}

scipy.io.savemat('D1test.mat', D2)

