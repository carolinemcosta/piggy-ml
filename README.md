# About the project
This is a pilot research project aiming at classifying cardiac tissue as healthy or damaged using electrical signals measured inside the heart. The presence of damaged tissue can cause heart beat disturbances, know as arrhythmias, which can be fatal. Current clinical approaches using signal amplitude threshold are innacurate.

This study uses two additional morphological features of intra-cardiac electrical signals to classify underlying tissue as healthy or damage using Random Forest and SVM classifiers. One features is the minium slope of the signal, called DVDT. The other feature is the time between the mimum slope and signal recovery time, called activation-recovery interval (ref).

# The dataset
The dataset consists of intra-cardiac electrograms obtained from six pigs, where tissue damaged was artificially induced. An average of 4,000 electrograms were recorded from each pig, thus, the dataset comprises of multiple data points from the same individuals. The signals were pre-processed and analysed to compute features the signal amplitude **(AMP)**, the activation-recovery interval **(ARI)**, and the maximum downslope of the signal **(DVDT)**. Cardiac magnetic resonance images from the same pigs were used to **label** signals or damaged healthy or using image segmetation and registration techniques.

# Preparing the data
The dataset includes several data points from the same individual, thus, the two pigs were randomnly selected to be the testing set, and data from the remaining pigs was included in the training set.The data was transformed to have a normal distribution and the values were scaled to be within 0 and 1.  

# Training and evaluating the models
Model hyperparameters were tuned using a grid search and the models were evaluated on the training set using grouped 2-fold cross validation and accuracy scoring. 

The SVM classifier has an accuracy of 0.84 against 1.0 of the Random Forest classifier, indicating that the Random Forest classifier is overfitting. This is compared with an accuracy of 0.4 for the amplitude threshold classifier. 

# Next steps
Other morphological features that could improve classification are currently being investigated. Additional data from different studies might also be included. The use of neural networks to classify the electrical signals is also being considered.

# Latest update
The different in ARI between scar and healthy tissue is not statistically significant. After removing the ARI feature from the traning set, the Random Forest and SVM have accuracies of 0.78 and 0.84, respectively. The Random Forest is no longer overfitting, but has now has lower accuracy than SVM model, indicating that the latter is more approapriate for this problem.

# License
This is a research project using mostly unpublished data

