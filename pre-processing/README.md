# Preprocessing Code
The code in this folder provides the tool to preprocess the raw signals recorded through the smartphone application in order to use them with the deep learning framework. </br>

The operations involved in the preprocessing are:
- Visual flow computation from video data.
- Interpolation: deal with non-linear sampling time of smartphone sensors.
- Filtering: reduce noise.
- Cycle extraction: identify Gait Cycles using the *Continuous Walvelet Transform*.
- Signals de-trending: deal with sensors drift.
- Normalization: standard step for deep learning applications.

## Getting started 
Our project takes advantage of Anaconda to create virtual environments to run the code.</br>
1. Download and install [Anaconda](https://www.anaconda.com/distribution/#download-section).</br>
2. Create the environment:
```bash
conda env create -n gait_analysis -f gait_analysis.yml
```
3. Activate the environment:
```bash
conda activate gait_analysis 
```

## MATLAB integration
Several of the functions used in this part of the project have been implemented in MATLAB. To run the code it is mandatory that a version of MATLAB R2014b or later is installed on the machine you want to use.</br>

After installing a valid version of MATLAB, follow these steps to set up the **MATLAB Engine API for Python** ([source](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)):

- At a Windows operating system prompt: 
```bash
cd "matlabroot\extern\engines\python"
python setup.py install
```

- At a macOS or Linux operating system prompt: 
```bash
cd "matlabroot/extern/engines/python"
python setup.py install
```

- At the MATLAB command prompt: 
```bash
cd (fullfile(matlabroot,'extern','engines','python'))
system('python setup.py install')
```

NB: Administrative privileges might be needed to execute these commands. Tested on MacBook Pro with MATLAB R2019a and R2017b.

For more info visit the official [documentation](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

## How to use?
If you want to try out your data follow these steps:

- Download the recorded data from the smartphone to the folder `./data/raw_data/`. There should be one folder for each acquisition. NB: The ALV application takes care of creating unique folder identifiers ordered according to the date and time, therefore there is no need for renaming.
- Run the preprocessing script:
```bash
python preprocessing.py 
```
- The procedure will create, in the folder `./data/preprocessed_data/`:
    - One folder for each recorded activity with preprocessed data for that acquisition.
    - A csv file, `final_matrix.csv`, with the information from all the acquisitions.
    - A csv file, `cycles_per_acquisition.csv`, with the number of gait cycles for each acquisition.
    - A file containing a set of scalar to apply to future new data for normalization purpose.

## What's more? Visualizations!
The script is filled with (by default disabled) visualization routines for gaining more insight on the data. 
Check out the code for more details.
```
TODO: Give more information.
```
