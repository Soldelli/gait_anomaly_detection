"""
AUTHOR: Mattia Soldan
CONTACT: mattia.soldan@kaust.edu.sa

NOTE:
	NOTE:
	This library implements the preprocessing to apply to raw data acquired by the smatphone 
	before applying the Gait Anomaly Detection inference.
	Please refer to the paper for further details.
"""

import os
import sys
import csv
import time
import joblib
import argparse
import numpy as np
from utils import *
from config import *
from pathlib import Path
from sklearn import preprocessing
from camera_tracking import camera_tracking

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools

setup_folders()

num_channels = 9        # if set to 6, video info are not involved in the procedure.
Fs_prime = 200          # Hz, sampling frequency

# Parser setup  ---------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Preprocessing Gait Data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--folder', type=str, default='raw_data',
    help='Name of folder in which raw data is stored. \nAlways start with "raw_"')

args = parser.parse_args()
which_data = args.folder

# MAIN PROCESSING  ------------------------------------------------------------
print('--------------------  PREPROCESSING OF GAIT DATA  --------------------\n')
print('Selected working directory: data/{}'.format(which_data))

# Extraction of information regarding the video collected by the android app and placed in data/raw_data/* folders
camera_tracking(which_data, force=force)        

# Load of all raw data for preprocessing procedure  -----------------------
t = time.time()
data = read_data(VISUALIZE=VISUALIZE, which_data=which_data)

if len(data['acc_timestamp']) == 0:
    print('Data not found.')
    sys.exit(0)

print('Time to load the raw data: {:0.2f} seconds.\n'.format(time.time() - t))

# PSD visualization -------------------------------------------------------
if True:  # Experimental, not a sistematic visualization
    n=0
    f_cut = 40
    args = {'sig1':data['acc_x'][n],
            'sig2':data['acc_y'][n],
            'sig3':data['acc_z'][n],
            'sig_timestamp1':data['acc_timestamp'][n], 
            'sig_timestamp2':data['acc_timestamp'][n], 
            'sig_timestamp3':data['acc_timestamp'][n], 
            'dir':'raw'
            }
    psd_comparison(**args)

    names_list = ['acc_x', 'acc_y', 'acc_z', 
                    'gyro_x', 'gyro_y', 'gyro_z', 
                    'thx', 'thy', 'thz']
    for name in names_list:
        psd_visualization(sig=data[name][n], f_cut=f_cut, name=name, dir='raw')

# Preprocessing of raw data  ----------------------------------------------
cycles_tot = 0
cycles_feat = []
cluster_length = np.zeros(len(data['acc_x']))  

usr_height = 1.73
leg_length = 0.45 * usr_height

num_acquisitions = len(data['acc_x'])
already_preprocessed, k = 0, 0
total_cycles = []
cycles_per_acquisition = np.zeros(num_acquisitions, dtype=int)

time_preproc = time.time()
f = './data/preprocessed_{}/acquisition_'.format(which_data[4:])

for i in range(num_acquisitions):
    file = '{}{}/acc_x.csv'.format(f, i+1)

    if not (Path(file).is_file()) or force:
        if already_preprocessed != 0:
            already_preprocessed = 0

        args = {'Fs_prime':Fs_prime,
                'visualize':False,
                'sig0':data['acc_y'][i],
                'sig1':data['gyro_y'][i],
                'sig_cycles':data['acc_x'][i],
                'sig_timestamp0':data['acc_timestamp'][i],
                'sig_timestamp1':data['gyro_timestamp'][i],
                'cycles_timestamp':data['acc_timestamp'][i]
                }
        IC, FC, ICleft, ICright = preprocessing_y(**args)

        # -----------------------------------------------------------------
        """
        In this subsection the user can specify which variables are 
        taken in account for preprocessing 

        input_type : sygnals to be preprocessed
        timestamp_type : timestamps related to the input_type elements, 
        used for interpolation purposes.
        """

        ## 9 features acceletometer / gyroscope / video
        if num_channels == 9:
            input_type = [
                data['acc_x'][i] , data['acc_y'][i] , data['acc_z'][i], 
                data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i],
                data['thx'][i]   , data['thy'][i]   , data['thz'][i]
            ]
    
            timestamp_type = np.repeat(np.array([data['acc_timestamp'][i],
                                                data['gyro_timestamp'][i],
                                                data['frame_timestamp'][i]]), 3)
            input_names = ['acc_x', 'acc_y', 'acc_z', 
                            'gyro_x', 'gyro_y', 'gyro_z',
                            'thx', 'thy', 'thz']

        ## 6 features accelerometer / gyroscope
        elif num_channels == 6:
            input_type = [data['acc_x'][i], data['acc_y'][i], data['acc_z'][i], 
                            data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i]]

            timestamp_type = [data['acc_timestamp'][i], data['acc_timestamp'][i], 
                            data['acc_timestamp'][i], data['gyro_timestamp'][i], 
                            data['gyro_timestamp'][i], data['gyro_timestamp'][i]]

            input_names = ['acc_x', 'acc_y', 'acc_z', 
                            'gyro_x', 'gyro_y', 'gyro_z']
        
        else:
            raise ValueError('Specified number of channels is prohibited, \
                                            accepted values are 6 or 9.')

        # -----------------------------------------------------------------
        # # first and last 5 walk cycles are removed
        index_value = 0
        if len(ICleft) > 10:
            index_value = ICleft[5:-5]
        else:
            index_value = ICleft[0:]

        if gait_macro_parmeters:            # Not handled in current implementation, present for legacy purposes
                                            # Refer to Alberto Lanaro's Master Thesys for more details
            cluster_length[k] = len(index_value) - 1

            # stance computation, swing time of each cycle
            stance_time, swing_time = stance_swing_time(ICs=index_value, FCs=FC, Fs=Fs_prime)

            # step velocity computation
            args = {'IC':IC,
                'Fs':Fs_prime,
                'l':leg_length,
                'index':index_value,
                'acc_y':data['acc_y'][i],
                'acc_timestamp':data['acc_timestamp'][i],
                }
            mean_step_velocity = step_vel(**args)

            if VISUALIZE:
                args = {'msv':mean_step_velocity, 
                        'st_time':stance_time, 
                        'sw_time':swing_time, 
                        'i':i
                        }
                plot_walk_statistics(**args)

            for ii in range(len(stance_time)):
                cycles_feat.append(np.array([stance_time[ii], 
                                            swing_time[ii], 
                                            mean_step_velocity[ii]]))

        preprocessed_data = []
        for ii, j in zip(input_type, timestamp_type):
            args = {'input':ii, 
                    'timestamp':j, 
                    'index_values':index_value, 
                    'fixed_length':Fs_prime, 
                    'Fs_prime':Fs_prime,
                    'filter_vis':filter_vis}
            cycles_temp, _ = cycles_extraction(**args)
            cycles_tot  += len(cycles_temp)
            row_data = np.hstack(cycles_temp)
            preprocessed_data.append(row_data)

        total_cycles.append(preprocessed_data)
        cycles_per_acquisition[i] = int(len(preprocessed_data[0]) / Fs_prime)
        k += 1
    else:
        already_preprocessed += 1

    if already_preprocessed == num_acquisitions:
        print('All data have been already preprocessed. (set force=True to re-preprocess)')

""" Last step -- Normalization procedure, in this section the mean is removed from the data and then data 
is divided by the standard deviation."""

final_matrix = np.hstack(total_cycles)

# Trend removal ---------------------------------------------------------------
if num_channels == 9 and video_trick:
    final_matrix[6:, :] = video_flattening( final_matrix[6:, :], 
                                            cycles_per_acquisition, 
                                            which_data)

# Data normalization  ---------------------------------------------------------
if data_normalization:
    if not os.path.exists('./data/preprocessed_data/'):         
        os.makedirs('./data/preprocessed_data/')
    if not which_data == 'raw_data':
        scaler = joblib.load('./data/preprocessed_data/scaler')
        final_matrix = scaler.transform(final_matrix.transpose()).transpose()
    else:
        scaler = preprocessing.StandardScaler()
        final_matrix = scaler.fit_transform(final_matrix.transpose()).transpose()
        joblib.dump(scaler,'./data/preprocessed_data/scaler')

print('Preprocessing phase of data in {} executed in {:0.2f} seconds.\n'
                            .format(which_data,time.time() - time_preproc))

if which_data == 'raw_data':
    for i in range(len(final_matrix)):
        psd_visualization(  sig=final_matrix[i], 
                            f_cut=40, 
                            name=input_names[i], 
                            dir='final')
        # Experimental, not a sistematic visualization

path = './data/preprocessed_{}/'.format(which_data[4:])
if not os.path.exists(path):         # create directory is not present
    os.makedirs(path)

# Dump output  ----------------------------------------------
print('Dumping final_matrix.csv')
with open(path + 'final_matrix.csv', 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerows(final_matrix)

print('Dumping cycles_per_acquisition.csv')
with open(path + 'cycles_per_acquisition.csv', 'w') as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(cycles_per_acquisition)
    wr.writerow([len(input_names)])
counter = 0
for i in range(num_acquisitions):
    new_path = path + 'acquisition_{}/'.format(i + 1)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    with open('{}preprocessed.csv'.format(new_path), 'w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        init = Fs_prime * counter
        final = Fs_prime * cycles_per_acquisition[i] + counter * Fs_prime
        wr.writerows(final_matrix[:, init: final])
        counter += cycles_per_acquisition[i]

print('Total number of cycles extracted: {}\n'.format(np.sum(cycles_per_acquisition)))

# Output visualization  -------------------------------------------------------
args = {'which_data': which_data,
            'rescale':False, 
            'vis':0
            }
if preproc_data_inspection:
    [data_inspection(i+1,**args) for i in range(num_acquisitions)]