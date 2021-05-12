# About the project
This is a pilot research project aiming at electrical signals measured inside the heart from to classify cardiac tissue as healthy or damaged. Tissue damage is a consequence of myocardial infarction (heart attack), where blood supply to the heart is interrupted. Damaged tissue is replaced by scar tissue, which has different structural and physiological properties than healthy tissue. A key difference, is that the damaged tissue is less able to conduct electrical signals within the heart. Physiological electrical conduction is important for an effiecient heart beat. Thus, the presence of damaged tissue can cause heart beat disturbances, know as arrhythmias, which can be fatal.

One of the treatments for arrhythmias caused by the presence of damaged tissue is to effectively cause more tissue damage by burning around the damaged area. This is known as ablation. In this procedure, areas suspected of causing conduction disturbances are targeted and burned. However, identifying these areas is a major clinical challenge and a topic of intensive research. One approach is measure electrical signals inside the heart during the procedure and look at key features of these signals, such as their amplitude. This is because damaged tissue is known to create weaker (low amplitude) signals compared to the healthy tissue. However, this is not straightfoward, as damaged and healthy tissue are tipically mixed together, so the signals are difficult to interpret.

Other features of the electricals signals have been investigated, such as the duration of the signal and the number of peaks. However, none of these features alone are specific enough to identify ablation targets accurately. Thus, we propose to use machine learning models to combine known features of damaged tissue to improve signal classification.

# The dataset
The dataset consists of intra-cardiac electrograms obtained from pigs, where tissue damaged was artificially induced. These signals were pre-processed and analysed to compute features, such as the activation time, the recovery time, the activation-recovery interval, and the maximum downslope of the signal. Cardiac magnetic resonance images were also obtained from these pigs and used to classify tissue as healthy or damaged using image-processing techniques. This image-based classification was mapped onto the electrogram data and is used in this project as our gold-standard, or **target**.

# Initial strategy
## Training and testing sets
## Feature scaling
## Choosing a machine learning model
## Results

# Improving the model
