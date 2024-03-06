import scipy.io as spio
import numpy as np
matD1 = spio.loadmat('D1.mat', squeeze_me=True) 
matD2 = spio.loadmat('D2.mat', squeeze_me=True) 
d_D1 = matD1['d']
Index_D1 = matD1['Index'] 

Class_D1 = matD1['Class']

d_D2 = matD2['d']


from scipy.stats import median_absolute_deviation
import pandas as pd
from scipy.signal import butter,filtfilt
import scipy.io

#########################

#FILTER Training SIGNAL (D1.mat)

#use butterworth filter to filter signal d
def butter_lowpass_filter(d_D1, low_cutoff, high_cutoff, fs, order):
    b, a = butter(order, Wn= [low_cutoff,high_cutoff], btype='bandpass', analog=False, fs=fs)
    y = filtfilt(b, a, d_D1)
    return y

sf_D1 = butter_lowpass_filter(d_D1, 100, 1500, 25000, 2) 

##################

#FILTER TEST SIGNAL (D2.mat)

#use butterworth filter to filter signal d
def butter_lowpass_filter(d_D2, low_cutoff, high_cutoff, fs, order):
    b, a = butter(order, Wn= [low_cutoff,high_cutoff], btype='bandpass', analog=False, fs=fs)
    y = filtfilt(b, a, d_D2)
    return y

sf_D2 = butter_lowpass_filter(d_D2, 100, 1500, 25000, 2)


##########################

#DETECT SPIKES from D2 (Create New index)

from scipy.signal import find_peaks

#calculate cutoff threshold value for spikes
threshold = 5*np.median(np.abs(sf_D2)/0.6745)
h = threshold

spikes,_ = find_peaks(sf_D2,h, distance=30)
Index_D2 = spikes #New_index is in order
#move the index backwards (as the peak is at the top of spike), so it matches given index location in the spikes
for i in range(len(Index_D2)):
    Index_D2[i] = Index_D2[i] - 15
print("Spikes detected =",Index_D2)


###############################

#EXTRACT SPIKES from D1 (Class and Index)

# Extract the 60 voltage datapoints after an index (spike voltage) for every Index in d
spike_voltages_D1=[]
for index in (Index_D1):
    # Extract the data for the current spike from the original signal where i is the index
    spike = sf_D1[index-5:index+100] 
    # Find the point of maximum gradient of the spike
    #max_gradient = np.argmax(np.gradient(spike))
    # Find the first significant zero crossing of the spike
    #zero_crossing = np.argmax(np.abs(spike) > 0)
    for i in range(len(spike)):
        if spike[i] > 0:
            zero_crossing = i
    #zero_crossing = next((j for j, n in enumerate(sf_D1) if n > 0), -1)
    aligned_spike = spike[zero_crossing:zero_crossing+60]
    spike_voltages_D1.append(aligned_spike)

# Create a dataframe with the list of spike voltages for each spike
df_spike_voltages = pd.DataFrame (spike_voltages_D1) 
# Create a datframe with the index vector and its corresponding class vector
df = pd.DataFrame({'Index': Index_D1, 'Class': Class_D1}) 
# Combine both dataframes to create the training sample
df= pd.concat([df, df_spike_voltages], axis=1)

#to get rid of index column df = df.drop("Index", axis = 1)

df.to_csv("extracted_spikes_D1.csv")

##############

#Extract Spikes from D2 (No Class, Only Index)

# Extract the 60 voltage datapoints after an index (spike voltage) for every Index in d
spike_voltages_D2=[]
for i in (Index_D2):
    # Extract the data for the current spike from the original signal where i is the index
    spike = sf_D2[i-10:i] 
    # Find the point of maximum gradient of the spike
    #max_gradient = np.argmax(np.gradient(spike))
    # Find the first significant zero crossing of the spike
    zero_crossing = np.argmax(np.abs(spike) > 0)
    aligned_spike = sf_D2[zero_crossing:zero_crossing+75]
    spike_voltages_D2.append(aligned_spike)
# Create a dataframe with the list of spike voltages for each spike
df_spike_voltages = pd.DataFrame (spike_voltages_D2) 
# Create a datframe with the index vector and its corresponding class vector
df = pd.DataFrame({'Index': Index_D2}) 
# Combine both dataframes to create the training sample
df= pd.concat([df, df_spike_voltages], axis=1)

#to get rid of index column df = df.drop("Index", axis = 1)

df.to_csv("extracted_spikes_D2.csv")

###########

##PCA + KNN 

# Numpy for useful maths
import numpy as np
# Sklearn contains some useful CI tools
# PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler 
# k Nearest Neighbour
from sklearn.neighbors import KNeighborsClassifier
# Matplotlib for plotting
import matplotlib.pyplot as plt
import pandas as pd

#load train data
df = pd.read_csv("extracted_spikes_D1.csv")
test = df.to_numpy()
np.savetxt("extracted_spikes_D1.csv",test, delimiter=",")

#load test data
df = pd.read_csv("extracted_spikes_D2.csv")
test = df.to_numpy()
np.savetxt("extracted_spikes_D2.csv",test, delimiter=",")


# Load the train and test data
train = np.loadtxt('extracted_spikes_D1.csv', delimiter=',') 
test = np.loadtxt('extracted_spikes_D2.csv', delimiter=',')

# Separate labels from training data
train_data = train[0:, 3:] #from the 2nd column onwards (voltage datapoints)
train_labels = train[0:, 2] #the 1st column (class)
test_data = test[0:, 2:]  #from the 2nd column as there is no index
# no test_labels as that is what we want to find 

#PCA Starts
# Select number of components to extract
pca = PCA(n_components = 10)
# Fit to the training data
pca.fit(train_data)

# Determine amount of variance explained by components
print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))


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


print ("Predicted Class", (pred_class[:10]))
print ("Index length", len(Index_D2))
print ("Class length",len(pred_class))
import scipy.io 

D2 = {'d': d_D2, 'Index': Index_D2, 'Class': pred_class}

scipy.io.savemat('D2.mat', D2)
