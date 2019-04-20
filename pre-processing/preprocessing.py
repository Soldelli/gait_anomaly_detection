"""
AUTHOR: Mattia Soldan
CONTACT: mattia.soldan@kaust.edu.sa

NOTE:
	NOTE:
	This library implements the preprocessing to apply to raw data acquired by the smatphone 
	before applying the Gait Anomaly Detection inference.
	Please refer to the paper for further details.
	
WARNING: 
	TBD

DEPENDENCIES: 
	Python >= 3.6
    os
    csv
    time
    numpy
    utils   (check it's dependencies)
    pathlib
    sklearn
    camera_tracking (matlab is required)
"""

import os
import sys
import csv
import time
import numpy as np
from utils import *
from pathlib import Path
from sklearn import preprocessing
#sys.path.append(os.path.abspath("pre-processing/camera_tracking/"))
from camera_tracking import camera_tracking

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools

save_fig_preproc = './images/preprocessing/preprocessed/acquisition_'
save_data = './data/preprocessed_data/acquisition_'
VISUALIZE = False                    # Enable visualization of several statistics
force = True
rescale = False                      # Set to true to force rescaling of preprocessed data to [0-1] range.
preprocessed_data_inspection = False # Set to true to produce pdf for visualization of preprocessed data.
video_trick = True                   # Set to true if you want to remove the drift from video features

num_channels = 9  # if set to 6, video info are not involved in the procedure.

if __name__ == '__main__':
    # Extraction of information regarding the video collected by 
    # the android app and placed in data/raw_data/* folders
    camera_tracking.camera_tracking(force=False)  

    # Load of all raw data for preprocessing procedure  ----------------------------------------------------------------
    t = time.time()
    data  = read_data(VISUALIZE=False, save_data=save_data)
    print('Time to load the raw data: {0:.2f} seconds.\n'.format(time.time() - t))

    # PSD visualization ------------------------------------------------------------------------------------------------
    if VISUALIZE:  # Experimental, not a sistematic visualization
        psd_visualization(sig=data['thx'][0], f_cut=40, name='esperimental')

    # Preprocessing of raw data  ---------------------------------------------------------------------------------------
    Fs_prime    = 200  # Hz, sampling frequency
    cycles_tot  = 0
    cycles_feat = []
    cluster_length = np.zeros(len(data['acc_x']))  

    usr_height = 1.73
    leg_length = 0.45 * usr_height

    num_acquisitions = len(data['acc_x'])
    already_preprocessed, k = 0, 0
    total_cycles = []
    cycles_per_acquisition = np.zeros(num_acquisitions, dtype=int)

    for i in range(num_acquisitions):
        file = './data/preprocessed_data/acquisition_{}/acc_x.csv'.format(i + 1)
        if not (Path(file).is_file()) or force:
            if already_preprocessed != 0:
                already_preprocessed = 0
            time_preproc = time.time()

            # acc_mag = np.sqrt(np.power(data['acc_x'][i], 2) + 
            #                   np.power(data['acc_y'][i], 2) +
            #                   np.power(data['acc_z'][i], 2))  # magnitude acc 
            # gyro_mag = np.sqrt(np.power(data['gyro_x'][i], 2) + 
            #                    np.power(data['gyro_y'][i], 2) +
            #                    np.power(data['gyro_z'][i], 2))  # magnitude gyro 

            args = {'Fs_prime':Fs_prime,
                    'visualize':VISUALIZE,
                    'sig0':data['acc_y'][i],
                    'sig1':data['gyro_y'][i],
                    'sig_cycles':data['acc_x'][i],
                    'sig_timestamp0':data['acc_tstamp'][i],
                    'sig_timestamp1':data['gyro_tstamp'][i],
                    'cycles_timestamp':data['acc_tstamp'][i]
                    }
            IC, FC, ICleft, ICright = preprocessing_y(**args)

            # del acc_mag, gyro_mag
            # ----------------------------------------------------------------------------------------------------------
            """In this subsection the user can specify which variables are taken in account for preprocessing 
            
            input_type : sygnals to be preprocessed
            tstamp_type : timestamps related to the input_type elements, used for interpolation purposes.
            """

            ## 9 features acceletometer / gyroscope / video
            if num_channels == 9:
                input_type = [
                    data['acc_x'][i] , data['acc_y'][i] , data['acc_z'][i], 
                    data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i],
                    data['thx'][i]   , data['thy'][i]   , data['thz'][i]
                ]
        
                tstamp_type = np.repeat(np.array([data['acc_tstamp'][i],
                                                  data['gyro_tstamp'][i],
                                                  data['frame_tstamp'][i]]), 3)
                input_names = ['acc_x', 'acc_y', 'acc_z', 
                               'gyro_x', 'gyro_y', 'gyro_z',
                               'thx', 'thy', 'thz']

            ## 6 features accelerometer / gyroscope
            elif num_channels == 6:
                input_type = [data['acc_x'][i], data['acc_y'][i], data['acc_z'][i], 
                              data['gyro_x'][i], data['gyro_y'][i], data['gyro_z'][i]]

                tstamp_type = [data['acc_tstamp'][i], data['acc_tstamp'][i], data['acc_tstamp'][i],
                               data['gyro_tstamp'][i], data['gyro_tstamp'][i], data['gyro_tstamp'][i]]

                input_names = ['acc_x', 'acc_y', 'acc_z', 
                               'gyro_x', 'gyro_y', 'gyro_z']
            
            else:
                raise ValueError('Specified number of channels is prohibited, accepted values are 6 or 9.')

            # ----------------------------------------------------------------------------------------------------------
            # first and last 5 walk cycles are removed
            index_value = ICleft[5:-5]
            cluster_length[k] = len(index_value) - 1

            # stance computation, swing time of each cycle
            stance_time, swing_time = stance_swing_time(ICs=index_value,
                                                        FCs=FC,
                                                        Fs=Fs_prime)

            # step velocity computation
            args = {'IC':IC,
                    'Fs':Fs_prime,
                    'l':leg_length,
                    'index':index_value,
                    'acc_y':data['acc_y'][i],
                    'acc_tstamp':data['acc_tstamp'][i],
                    }
            mean_step_velocity = step_vel(**args)

            #print('Stance: ',stance_time)
            #print('Swing: ', swing_time)
            #print('Step velocity: ' , mean_step_velocity)

            if VISUALIZE:
                args = {'msv':mean_step_velocity, 
                        'st_time':stance_time, 
                        'sw_time':swing_time, 
                        'i':i
                        }
                plot_walk_statistics(**args)

            for ii in range(len(stance_time)):
                walk_statistics = np.array([stance_time[ii], 
                                            swing_time[ii], 
                                            mean_step_velocity[ii]]
                                          )
                cycles_feat.append(walk_statistics)

            #cycles = np.zeros([len(index_value) - 1, Fs_prime * len(input_type)])

            preprocessed_data = []
            for ii, j, name in zip(input_type, tstamp_type, input_names):
                
                args = {'input':ii, 
                        'timestamp':j, 
                        'index_values':index_value, 
                        'fixed_length':Fs_prime, 
                        'Fs_prime':Fs_prime}
                cycles_temp, _ = cycles_extraction(**args)

                cycles_tot += len(cycles_temp)

                if not os.path.exists(save_data + str(i + 1)):
                    os.makedirs(save_data + str(i + 1))

                for h in range(len(cycles_temp)):
                    if VISUALIZE and (h % 5) == 0:
                        args = {'save_fig_preproc':save_fig_preproc,
                                'cycles_temp':cycles_temp,
                                'name':name,
                                'h':h,
                                'i':i,
                                }
                        plot_preprocessed_examples(**args)

                row_data = np.hstack(cycles_temp)
                preprocessed_data.append(row_data)

            total_cycles.append(preprocessed_data)
            cycles_per_acquisition[i] = len(preprocessed_data[0]) // Fs_prime
            k += 1
            print('Preprocessing acquisition {:02d} of {:02d} executed in {:0.2f} seconds.'.format(i + 1, num_acquisitions, time.time()-time_preproc))
        else:
            already_preprocessed += 1
        if already_preprocessed == num_acquisitions:
            print('All data have been already preprocessed. (set force=True to re-preprocess)')
    print()

    # Elimination of old variables
    #del data, IC, ICleft, ICright, FC, row_data, cycles_temp, input_type, tstamp_type
    """ Last step -- Normalization procedure, in this section the mean is removed from the data and then data 
    is divided by the standard deviation."""

    final_matrix = np.hstack(total_cycles)

    if num_channels == 9 and video_trick:
        final_matrix[6:, :] = video_flattening(final_matrix[6:, :],
                                               cycles_per_acquisition,
                                               Fs_prime)

    final_matrix = preprocessing.scale(final_matrix, axis=1)

    for i in range(len(final_matrix)):
        psd_visualization(sig=final_matrix[i][100:500],
                          f_cut=30,
                          name=input_names[i]
                          )  # Experimental, not a sistematic visualization

    ## Rescale data between [0,1] ---------------
    min = np.min(final_matrix, axis=1).reshape(len(input_names), 1)
    max = np.max(final_matrix, axis=1).reshape(len(input_names), 1)

    if rescale:  # this apply the [0-1] rescaling is rescale is set to True
        final_matrix -= min
        final_matrix /= max - min

    #determinazione di min e max
    temp = np.zeros(num_acquisitions)
    for i in range(num_acquisitions):
        temp[i] = np.sum(cycles_per_acquisition[:i + 1])

    for i in range(len(final_matrix)):
        if rescale:
            index_min = int(
                np.sort(np.where(final_matrix[i] == 0.0)) / Fs_prime)
            index_max = int(
                np.sort(np.where(final_matrix[i] == 1.0)) / Fs_prime)
        else:
            index_min = int(
                np.sort(np.where(final_matrix[i] == min[i]))[0][0] / Fs_prime)
            index_max = int(
                np.sort(np.where(final_matrix[i] == max[i]))[0][0] / Fs_prime)

        ii = 0
        while index_min > temp[ii]:
            ii += 1
        print('Min for signal {} found in acquisition {}'.format(input_names[i],ii + 1))
        ii = 0
        while index_max > temp[ii]:
            ii += 1
        print('Max for signal {} found in acquisition {}\n'.format(input_names[i],ii + 1))
    print()

    print('Dumping final_matrix.csv')
    with open('./data/preprocessed_data/final_matrix.csv','w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerows(final_matrix)

    print('Dumping cycles_per_acquisition.csv')
    with open('./data/preprocessed_data/cycles_per_acquisition.csv','w') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(cycles_per_acquisition)
        wr.writerow([len(input_names)])

    counter = 0
    for i in range(num_acquisitions):
        with open('{}{}/preprocessed.csv'.format(save_data, i + 1), 'w') as file:
            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
            init  = Fs_prime * counter
            final = Fs_prime * cycles_per_acquisition[i] + counter * Fs_prime
            wr.writerows(final_matrix[:, init:final])
            counter += cycles_per_acquisition[i]

    if preprocessed_data_inspection:
        [data_inspection(acquisition=i + 1, rescale=rescale, vis=0) 
                                    for i in range(num_acquisitions)]
