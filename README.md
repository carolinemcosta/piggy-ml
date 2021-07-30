# About the project
This is a pilot research project aiming at classifying cardiac tissue as healthy or damaged using electrical signals measured inside the heart. The presence of damaged tissue can cause heart beat disturbances, know as arrhythmias, which can be fatal. Current clinical approaches using signal amplitude threshold are innacurate.

This study uses two additional morphological features of intra-cardiac electrical signals to classify underlying tissue as healthy or damage using Random Forest and SVM classifiers. One features is the minium slope of the signal, called DVDT. The other feature is the time between the mimum slope and signal recovery time, called activation-recovery interval (ref).

# The dataset
The dataset consists of intra-cardiac electrograms obtained from six pigs, where tissue damaged was artificially induced. An average of 4,000 electrograms were recorded from each pig, thus, the dataset comprises of multiple data points from the same individuals. The signals were pre-processed and analysed to compute features the signal amplitude **(AMP)**, the activation-recovery interval **(ARI)**, and the maximum downslope of the signal **(DVDT)**. Cardiac magnetic resonance images from the same pigs were used to **label** signals or damaged healthy or using image segmetation and registration techniques.

# Preparing the data
The dataset includes several data points from the same individual, thus, the two pigs were randomnly selected to be the testing set, and data from the remaining pigs was included in the training set.The data was transformed via standardization.  

# Modelling

Three classifier are currently being investigated: Random Forest, SVM, and XGboost. 
A dummy classifier that uses only an amplitude threshold, as often done in a clinical setting, was also created.

Model hyperparameters were tuned using a grid search and the models were evaluated on the training set using grouped 2-fold cross validation and F1 scoring. 

Currently, the Random Forest classifier has the highest F1 score on the training set (0.98). All models have a much lower F1 score on the test set, indicating overfitting.  

# Next steps
The data is highly heterogeneous, which might explain why the models do not generalize well on the test set. 
Regularization has improved the models, but the problem persists.
The use of K-means clusterization as a pre-processing step is being investigated to reduce
data complexity and improve the models.

# License
This is a research project using mostly unpublished data

