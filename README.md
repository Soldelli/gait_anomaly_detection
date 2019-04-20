# Seq2Seq RNN based Gait Anomaly Detection from Smartphone Acquired Multimodal Motion Data
Deep learning approach to anomaly detection in gait data acquired through smartphone's multimodal sensors.
The proposed architecture takes advantage of RNN and CNN layers to implement a Sequence-to-Sequence feature extractor and a Convolutional classifier, check the paper for more details.</br>
<p align="center">
<img src="https://github.com/Soldelli/gait_anomaly_detection/blob/master/ALV/images/teaser_gait_analysis.png" width="700">
</p>

## Welcome
If you find any piece of code valuable for your research please cite this work:</br>

``` 
TBD when the paper is accepted 
```

And don't forget to give us a :star: in the github banner :wink:.

## Project structure
The folders are organized as follows:
- `ALV/` contains the Android application **ActivityLoggerVideo** used for the data collection.
- `pre-processing` contain the scripts to apply the preprocessing tranformations of the signals described in the paper. It is the necessary data preparation step for the later use in the deep learning framework.
- `Seq2Seq-gait-analysis/` contains the inference code and details on how to use it.
- `Seq2Seq-gait-analysis/Data/` is a placeholder for the preprocessed gait sequences used for training and testing of the proposed network. </br>

**Check each folder's README for more details**.


					
