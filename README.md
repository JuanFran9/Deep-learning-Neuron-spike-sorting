# Neuron-spike-sorting

Problem: 

There are several spikes within the recordings that
represent extracellular action potentials from five different types of neurone. The algorithm was developed 
to detect the spikes within the signal and determine which type of neurone produces the spike (Type
1, 2, 3, 4 or 5).
The algorithms was be tested and optimised using the given recordings and then challenged using a
different and more complex dataset with more noise.

![image](https://github.com/JuanFran9/Neuron-spike-sorting/assets/58949950/93343a03-5f77-40f7-a987-e10b22071020)

Solution: 

For the classification of the spikes the following steps were taken:

1. Filtering: The raw signals were filtered using a butterworth filter as this is an IIR filter that has no phase shift (unlike FIR filter) and also doesn´t modify the shape of the spike.
2. Spike Detection: The index of the spikes was found using a threshold of 5σ of the filtered signal.
3. Alignment: The first zero-crossing of the spike voltage before the spike peak or index was used as the new start of the extracted spike. 
4. CI technique: A PCA + KNN algorithm was used and obtained a 92.63% performance on validation, but an MLP with one hidden layer with 40 perceptrons obtained 95.65%

A **performance of 92.63% ** was achieved for the validation dataset using the PCA+KNN method.

![image](https://github.com/JuanFran9/Neuron-spike-sorting/assets/58949950/fa3a0db5-2d0e-4678-a076-d0267a21bfef)
