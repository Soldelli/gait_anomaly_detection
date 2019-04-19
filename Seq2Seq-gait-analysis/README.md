# Inference Code
In this folder can be found the inference scripts that analyze the collected data to solve the *Anomaly Detection* task.

## Getting started 
Our project takes advantage of Anaconda to create virtual environments to run the code.</br>
1 - Download and install [Anaconda](https://www.anaconda.com/distribution/#download-section).</br>
2 - Create the environment:
```bash
conda env create -n gait_analysis -f gait_analysis.yml
```
3 - Activate the environment:
```bash
conda activate gait_analysis 
```

## Data
Download the preprocessed data from this [link](https://drive.google.com/open?id=1Bw7bP53lxut0pTpHY35zL2dm6wXRkSV3).</br>
Unpack the downloaded zip and move the content into the folder `./data/`.</br>
Check that folder for more details about the files.


## Instructions
The first time you run the code you MUST follow these steps:</br></br>
**1 - Prepare the data:** This operation will prepare the data for the training operations that come after.
```bash
python data_utils.py 
```
**2 - Train the Seq2Seq encoder-decoder model:** It is recommended to perform this operation on a GPU to reduce the computational time to complete the step.
```bash
python bidirectional_autoencoder.py 
```
**3 - Train the CNN classifier:**
```bash
python conv_classifier_eval.py
```
**4 - Perform classification:** This step performs the classification operation on novel data, the test set.
```bash
python sequence_classification_app 
```
**5 - (OPTIONAL) Train the baseline model:**  SVM model.
```bash
python svm_classifier.py 
```

Please refer to the individual files for specific documentation and configurations.</br>
Once step 3 has been completed, you can perform step 4 anytime you want without having to retrain the models. Checkpoints of the trained models will be stored locally.
