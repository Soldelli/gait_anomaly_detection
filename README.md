# Seq2Seq RNN based Gait Anomaly Detection from Smartphone Acquired Multimodal Motion Data
Deep learning approach to anomaly detection in gait data acquired through smartphone's multimodal sensors.
The proposed architecture takes advantage of RNN and CNN layers to implement a Sequence-to-Sequence feature extractor and a Convolutional classifier, check the paper for more details.</br>
<p align="center">
<img src="https://github.com/Soldelli/gait_anomaly_detection/blob/master/ALV/images/teaser_gait_analysis.png">
</p>

## Welcome
If you find any piece of code valuable for your research please cite this work:</br>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2648530.svg)](https://doi.org/10.5281/zenodo.2648530)</br>

And don't forget to give us a :star: in the GitHub banner :wink:.

## Project structure
The folders are organized as follows:
- [ALV](https://github.com/Soldelli/gait_anomaly_detection/tree/master/ALV) contains the Android application **ActivityLoggerVideo** used for the data collection.
- [pre-processing](https://github.com/Soldelli/gait_anomaly_detection/tree/master/pre-processing) contains the scripts to apply the preprocessing transformations of the signals described in the paper. It is the necessary data preparation step for the later use in the deep learning framework.
- [Seq2Seq-gait-analysis](https://github.com/Soldelli/gait_anomaly_detection/tree/master/Seq2Seq-gait-analysis) contains the inference code and details on how to use it.
- [Seq2Seq-gait-analysis/Data](https://github.com/Soldelli/gait_anomaly_detection/tree/master/Seq2Seq-gait-analysis/data) is a placeholder for the preprocessed gait sequences used for training and testing of the proposed network. </br>

**Check each folder's README for more details**.


					
