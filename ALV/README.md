# Activity Logger Video Application

This folder contains the ALV android application used for data collection. </br>
The smartphone used in the paper submission is an "Asus Zenfone 2".</br>

## Setup
To install this application on a smartphone follow these steps:

- Download and install [Android Studio](https://developer.android.com/studio).
- Select `Open an existing Android Studio project` and chose the ALV folder of this repository.
- Connect the smartphone you want to use for data collection and pair it with the computer is necessary.
- Click on the green play button and when asked to select the smartphone as a testing device. This will install the application on the smartphone.

## Usage
<p align="center">
<img src="https://github.com/Soldelli/gait_anomaly_detection/blob/master/ALV/images/Screenshot_app.png" width="300" height="450" hspace="50">
<img src="https://github.com/Soldelli/gait_anomaly_detection/blob/master/ALV/images/chest_support.jpg" width="300" height="450" hspace="50">
</p>

Once the application is installed it is possible to start recording gait cycles.s

- Open the application and input the required information 
    - `Name` of the user
    - `Type` of activity
    - Keep `camera enabled` if you are planning to test our inference model.
    - Write any kind of `note` if needed, i.e. the kind of impairment movement you might want to mimic.
    - Select the `Start delay`: the amount of time (in seconds) the acquisitions process will be delayed after pressing the recording button. This gives the user the time to correctly place the smartphone on the ad-hoc made chest support.
    - Input the `Acquisition time`: the amount of time (in seconds) the acquisition process will last.
    - Push `START` and place the phone on the support.
- Record as many tracks as needed by repeating the above steps.
- Once the recording is terminated, connect the smartphone to a computer and download the data.

## Tips
For consecutive recordings in which the parameters do not change, use a Bluetooth remote control to trigger the data acquisition.</br>
If the inputted acquisition time is too long, a `STOP` button can be triggered to end the acquisition process.
