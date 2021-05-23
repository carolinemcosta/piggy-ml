# About the project
This is a pilot research project aiming at classifying cardiac tissue as healthy or damaged using electrical signals measured inside the heart. The presence of damaged tissue can cause heart beat disturbances, know as arrhythmias, which can be fatal. Current clinical approaches using signal amplitude threshold are innacurate.

This study uses two additional morphological features of intra-cardiac electrical signals to classify underlying tissue as healthy or damage using Random Forest and SVM classifiers. One features is the minium slope of the signal, called DVDT. The other feature is the time between the mimum slope and signal recovery time, called activation-recovery interval (ref).

# The dataset
The dataset consists of intra-cardiac electrograms obtained from six pigs, where tissue damaged was artificially induced. An average of 4,000 electrograms were recorded from each pig, thus, the dataset comprises of multiple data points from the same individuals. The signals were pre-processed and analysed to compute features the signal amplitude **(AMP)**, the activation-recovery interval **(ARI)**, and the maximum downslope of the signal **(DVDT)**. Cardiac magnetic resonance images from the same pigs were used to **label** signals or damaged healthy or using image segmetation and registration techniques.

# Preparing the data
The dataset includes several data points from the same individual, thus, the testing set should contain all data points from at least one pig. Since we  have 6 pigs, one pig was randomnly selected to be the testing set, and data from the remaining pigs was included in the training set.

Inspection of the distributions of the 3 features, as shown in the figure below (top row), reveals that these have different scales and that **AMP** and **DVDT** are tail-heavy. Thus, the data was transformed to have a normal distribution and the values were scaled to be within 0 and 1. This was done using the **QuantileTransformer** and the **MinMaxScaler** from scikit-learn. The resuling distributions are shown in the figure (bottom row).

![feature_scaling](https://user-images.githubusercontent.com/81109384/119149566-8ca40000-ba45-11eb-8ee9-a82bdc6c5753.png)


# Training and evaluating the models
Model hyperparameters were tuned using grid search and the models were evaluated on the training set using grouped 2-fold cross validation, and accuracy and ROC AUC scoring. 

The SVM classifier has slightly higher mean accuracy (0.83 *versus* 0.78) but smaller AUC (0.57 *versus* 0.58) than the Random Forest on the traning set, compared with an accuracy of 0.63 and an AUC of 0.5 for the amplitude threshold classifier.



