import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # import of package matplotlib.pyplot for plottools
import pandas as pd
import os
import time
import glob
from scipy import signal, io
import csv
import matlab.engine
import math
from pathlib import Path
from sklearn import preprocessing
eng = matlab.engine.start_matlab()


def read_data(VISUALIZE, save_data):
    """
    This function is used to upload all raw data from each diretory in data/raw_data
    absolute path are extensively used.

    Outputs: acc_timestamp, gyro_timestamp, frame_timestamp, acc_x, acc_y, acc_z,
                            gyro_x, gyro_y, gyro_z, deltax, deltay, deltaz
    Type of outpus is list of arrays, each array is a different acquisition.

    Note: There is no need to manually put usefull files in specific directories, 
    just copy all directory collected by the activity logger video application to 
    the  data/raw_data folder.
    """

    accelerometer   = "./data/raw_data/**/accellerometer_*.csv"
    gyroscope       = "./data/raw_data/**/gyroscope_*.csv"
    magnetometer    = "./data/raw_data/**/magnetometer_*.csv"
    video_path      = "./data/raw_data/**/*.mp4"


    # Load accelerometers data ------------------------------------------------
    acc_timestamp, acc_x, acc_y, acc_z = [], [], [], []

    num_acquisitions = 0
    for filename in glob.glob(accelerometer):
        acc = pd.read_csv(filename, sep='\t', header=0)
        acc_timestamp.append(np.array(acc.acc_timestamp, dtype=float))  # timestamps in ns
        acc_x.append(np.array(acc.acc_x))
        acc_y.append(np.array(acc.acc_y))
        acc_z.append(np.array(acc.acc_z))
        num_acquisitions += 1

    # Load gyroscope data -----------------------------------------------------
    gyro_timestamp, gyro_x, gyro_y, gyro_z = [], [], [], []
    for filename in glob.glob(gyroscope):
        gyro = pd.read_csv(filename, sep='\t', header=0)
        gyro_timestamp.append(np.array(gyro.gyro_timestamp, dtype=float))  # timestamps in ns
        gyro_x.append(np.array(gyro.gyro_x))
        gyro_y.append(np.array(gyro.gyro_y))
        gyro_z.append(np.array(gyro.gyro_z))

    # Load magnetometer data --------------------------------------------------
    magn_timestamp, magn_x, magn_y, magn_z = [], [], [], []
    for filename in glob.glob(magnetometer):
        magn = pd.read_csv(filename, sep='\t', header=0)
        magn_timestamp.append(np.array(magn.magn_timestamp, dtype=float))  # timestamps in ns
        magn_x.append(np.array(magn.magn_x))
        magn_y.append(np.array(magn.magn_y))
        magn_z.append(np.array(magn.magn_z))

    # Load video data ---------------------------------------------------------
    frame_timestamp =[]
    for filename in glob.glob("./data/raw_data/**/frame_timestamp.csv"):
        frame = pd.read_csv(filename, sep='\t', header=0)
        frame_timestamp.append(np.array(frame.frame_timestamp, dtype=float))  # timestamps in ns

    deltax, deltay, deltaz = [], [], []
    for filename in glob.glob(video_path):
        file = filename[0:filename.rfind('/')+1] + 'deltax.mat'
        deltax.append(io.loadmat(file)['deltax'].flatten())
        file = filename[0:filename.rfind('/')+1] + 'deltay.mat'
        deltay.append(io.loadmat(file)['deltay'].flatten())
        file = filename[0:filename.rfind('/')+1] + 'deltaz.mat'
        deltaz.append(io.loadmat(file)['deltaz'].flatten())

    thx, thy, thz, t = [], [], [], []
    for filename in glob.glob(video_path):
        file = filename[0:filename.rfind('/') + 1] + 'thx.mat'
        thx.append(io.loadmat(file)['thx'].flatten())
        file = filename[0:filename.rfind('/') + 1] + 'thy.mat'
        thy.append(io.loadmat(file)['thy'].flatten())
        file = filename[0:filename.rfind('/') + 1] + 'thz.mat'
        thz.append(io.loadmat(file)['thz'].flatten())

    # Pack all info into a dictionary -----------------------------------------
    data = {
        'acc_tstamp': acc_timestamp, 
        'gyro_tstamp': gyro_timestamp, 
        'frame_tstamp': frame_timestamp, 
        'acc_x': acc_x, 
        'acc_y': acc_y, 
        'acc_z': acc_z, 
        'gyro_x': gyro_x, 
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'deltax': deltax,
        'deltay': deltay,
        'deltaz': deltaz, 
        'thx': thx, 
        'thy': thy, 
        'thz': thz, 
    }

    if VISUALIZE:
        plot_raw_data(num_acquisitions, save_data, **data)

    return data


def write_data(name, value):  # DEPRECATED
    """
    This function is used to save all preprocessed data from each 
    diretory in data/preprocessed_data absolute path are extensively used.

    Outputs: acc_timestamp, gyro_timestamp, frame_timestamp, acc_x, acc_y, acc_z,
                            gyro_x, gyro_y, gyro_z, deltax, deltay, deltaz

    For each acquisition a different sub directory is 
    created /data/preprocessed_data/acquisition_*
    
    Type of outpus is csv files.

    Note: 
    """

    save_path = "./data/preprocessed_data/acquistion_"

    for i in range(len(name)):
        counter = 0
        for ii in range(len(value[i])):
            counter+=1
            current_path = save_path+str(ii)
            if not os.path.exists(current_path):
                os.makedirs(current_path)

            # -- Salvataggio su file --
            with open(current_path+'/'+name[i], 'wb') as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(value[i][ii])


def preprocessing_y(**kwargs):
    """
    This function extract the waling cycles using the accelerometer 
    and gyroscope information along y axis.
    
    The implemented procedure is explained in the Rochester paper, 
    provided with the documentation.
    """
    
    Fs_prime = kwargs['Fs_prime']
    visualize = kwargs['visualize']
    sig0 = kwargs['sig0']
    sig1 = kwargs['sig1']
    sig_cycles = kwargs['sig_cycles']
    sig_timestamp0 = kwargs['sig_timestamp0']
    sig_timestamp1 = kwargs['sig_timestamp1']
    cycles_timestamp = kwargs['cycles_timestamp']

    sig_prime0, sig_timestamp_prime0 = interpolation(sig0, sig_timestamp0, Fs_prime)
    sig_prime1, sig_timestamp_prime1 = interpolation(sig1, sig_timestamp1, Fs_prime)

    # ------------- STEP IDENTIFICATION ---------------------
    cwt_order = 40  # order of wavelet transform
    sig_prime0 = matlab.double(sig_prime0.tolist())
    sig_prime1 = matlab.double(sig_prime1.tolist())

    # determine star and end of each step
    IC, FC = eng.ICFC(sig_prime0, 
                      matlab.double([Fs_prime]), 
                      matlab.double([cwt_order]), 
                      matlab.double([visualize]),
                      nargout=2)  

    IC = np.array(IC[0], dtype=float)   # initial contact of each step
    FC = np.array(FC[0], dtype=float)   # final contact  of each step
    IC = IC[:]
    FC = FC[:]

    fc_lr = 1                           # cut frequency for IClr

    # determine left and right steps
    ICleft, ICright = eng.IClr(sig_prime1, 
                               matlab.double([Fs_prime]), 
                               matlab.double([fc_lr]),
                               matlab.double(IC.tolist()), 
                               matlab.double([visualize]), 
                               nargout=2)  

    ICleft  = np.array(ICleft[0], dtype=float)   # initial contact of left steps
    ICright = np.array(ICright[0], dtype=float)  # initial contact of right steps

    # ------ CYCLES EXTRACTION -------
    # determine cycles from the signal "sig_cycles" 
    index_values = ICleft.astype(int)
    fixed_length = Fs_prime  # lenght of each cycle

    args = {'input':sig_cycles, 
            'timestamp':cycles_timestamp, 
            'index_values':index_values, 
            'fixed_length':fixed_length, 
            'Fs_prime':Fs_prime
            }

    cycles, index_cycles = cycles_extraction(**args)
    m_cycles = np.mean(cycles, axis=0)  # average number of samples per cycle

    if visualize:
        plt.figure(figsize=(10, 6))
        for i in cycles:
            plt.plot(i)
        plt.plot(m_cycles, linewidth=2.5, color='k')
        plt.xlabel('samples')
        plt.show()
        for i in range(len(cycles)):
            plt.plot(index_cycles[i], cycles[i])
            plt.xlabel('timestamp')
            # plt.ylabel('m/s^2')
            # plt.ylim(np.array([4,19]))
            # plt.savefig('aaatest.pdf', format='pdf')
        plt.show()

    return IC, FC, ICleft, ICright


def butterworth(normalCutoff,b,a,order):
    w, h = signal.freqs(b, a)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.xscale('log')
    plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(normalCutoff, color='green')  # cutoff frequency
    name = './images/Butterworth_{}.pdf'.format(order)
    plt.savefig(name, dpi=150, transparent=False)
    plt.close()


def interpolation(input, tstamp, Fs_prime):
    """
    This functions is used to resample the raw data using Fs_prime as 
    frequency for the interpolation procedure.
    Moreover it estimates the average sampling frequency of the raw signal.
    """
    n_samples = tstamp.shape[0]
    Fs = 1 / (np.mean(tstamp[1: n_samples] - tstamp[0: n_samples - 1]) * 1e-9)  
    n_samples_prime = int(np.round(n_samples * Fs_prime / Fs))
    tstamp_prime    = np.linspace(tstamp.min(), tstamp.max(), n_samples_prime)  
    input_prime     = np.interp(x=tstamp_prime, xp=tstamp, fp=input)

    return input_prime, tstamp_prime


def cycles_extraction(**args):

    input = args['input']
    timestamp = args['timestamp']
    index_values = args['index_values']
    fixed_length = args['fixed_length'] 
    Fs_prime = args['Fs_prime']

    """This function performs the interpolation procedure, filtering of data and cycle extraction."""
    # Interpolation
    input_prime, timestamp_prime0 = interpolation(input, timestamp, Fs_prime)
    # LP filtering
    nyq = 0.5 * Fs_prime
    normalCutoff = 40 / nyq
    b, a = signal.butter(10, normalCutoff, btype='low', analog=False)
    VISUALIZE=False
    if VISUALIZE:
        butterworth(normalCutoff, b, a, 10)
        print('Butterworth visualization.')
    input_filt = signal.filtfilt(b, a, input_prime)
    # Cycles extraction ------------------------
    cycles, index_cycles = [], []
    for i in range(len(index_values) - 1):
        y_temp = input_filt[int(index_values[i]): int(index_values[i + 1])]
        x_temp = timestamp_prime0[int(index_values[i]): int(index_values[i + 1])]
        timestamp_prime1 = np.linspace(x_temp.min(), x_temp.max(), fixed_length)  # new timestamp
        cycles.append(np.interp(x=timestamp_prime1, xp=x_temp, fp=y_temp))
        index_cycles.append(timestamp_prime1)

    cycles = np.vstack(cycles)
    return cycles, index_cycles


def smooth(x, interval):    
    dft  = scipy.fftpack.rfft(x)[0: interval]
    idft = scipy.fftpack.irfft(dft, n=len(x))
    return idft


def stance_swing_time(ICs, FCs, Fs):
    FCstart = [n for n, i in enumerate(FCs) if i > ICs[0]][0]  # FC del primo passo
    FCtemp  = FCs[FCstart: len(FCs): 2]
    FCtemp  = FCtemp[0: len(ICs) - 1]
    stance_time = (FCtemp - ICs[:-1]) / Fs
    stride_time = (ICs[1:] - ICs[:-1]) / Fs
    swing_time  = stride_time - stance_time

    return stance_time, swing_time


def step_vel(**kwargs):

    l = kwargs['l']
    Fs = kwargs['Fs']
    IC = kwargs['IC']
    index = kwargs['index']
    acc_y = kwargs['acc_y']
    acc_tstamp = kwargs['acc_tstamp']

    ICstart = np.where(IC == index[0])[0]
    ICend = np.where(IC == index[-1])[0]

    ICvel = IC[ICstart[0]: (ICend[0] + 1)]
    acc_y_interp, _ = interpolation(acc_y, acc_tstamp, Fs)
    cutOff = 0.7  # high pass filter per eliminare drift in integrazione

    y_space = eng.space(matlab.double(acc_y_interp.tolist()), 
                        matlab.double([Fs]), 
                        matlab.double([cutOff]))

    y_space = np.array(y_space[0], dtype=float)

    # plt.plot(y_space)
    # plt.plot(IC, y_space[np.array(IC, dtype = int)],'ko')
    # plt.savefig('./images/y_space.pdf', dpi=150, transparent=False)
    # plt.close()

    step_velocity = np.zeros(len(ICvel) - 1)
    for i in range(len(ICvel) - 1):
        interval = np.arange(ICvel[i], ICvel[i + 1], dtype=int)
        max_val = np.max(y_space[interval])
        min_val = np.min(y_space[interval])
        h = math.fabs(max_val - min_val)
        step_length = 2 * math.sqrt(2 * l * h - math.pow(h, 2))
        step_time = (IC[i + 1] - IC[i]) / Fs
        step_velocity[i] = (float(step_length) / float(step_time))

    step_velocity = step_velocity.reshape((int(len(step_velocity) / 2), 2))
    mean_step_velocity = np.mean(step_velocity, axis=1)

    return mean_step_velocity


def psd_visualization(sig, f_cut, name):
    p = 20 * np.log10(np.abs(np.fft.rfft(sig)))     # psd of original signal
    nyq = 0.5 * 200                                 # Filter parameters
    normalCutoff = f_cut / nyq
    b, a = signal.butter(8, normalCutoff, btype='low', analog=False)    # filter definition
    sig1 = signal.filtfilt(b, a, sig)               # filtered signal
    pp = 20 * np.log10(np.abs(np.fft.rfft(sig1)))   # psd of filtered signal

    x = np.linspace(0, len(p), len(p))
    plt.subplot(2, 1, 1)
    plt.title('Power Spectral Density')
    plt.semilogx(x, p, linewidth=1)
    plt.ylabel('Signal dB')
    plt.subplot(2, 1, 2)
    plt.semilogx(x, pp, linewidth=1)
    plt.ylabel('Signal Filtered dB')
    plt.xlabel('Frequency f')
    plt.savefig('./images/PSD_'+name+'.pdf', 
                                        dpi=150, transparent=False)
    plt.close()

    init = 0
    final = 400
    x = np.linspace(0,len(sig[init:final]),len(sig[init:final]))
    plt.subplot(2, 1, 1)
    plt.title('Comparison Filtering')
    plt.plot(x, sig[init:final], linewidth=0.5)
    plt.ylabel('Signal')
    plt.subplot(2, 1, 2)
    plt.plot(x, sig1[init:final], linewidth=0.5)
    plt.ylabel('Signal Filtered')
    plt.xlabel('Time t')
    plt.savefig('./images/Filtering_comparison.pdf', 
                                        dpi=150, transparent=False)
    plt.close()


def data_inspection(acquisition, rescale, vis):
    """This function is used to upload all raw data from each diretory in data/raw_data
    absolute path are extensively used.

    Inputs: acquisition, the index of the acquisition to inspect,
            rescale, if 0-1 rescaling is applied to data, the visualization has 0-1 range.
            vis, [0,1] 0 to visualize both sections and whole acquisition, 1 just for whole acquisition

    Outputs: matrix containing, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, deltax, deltay, deltaz [9,200*num_cycles]
    Type of outpus is list of arrays, each array is a different acquisition.

    Note: There is no need to manually put usefull files in specific directories, just copy all directory collected by
    the activity logger video application to the  data/raw_data folder."""

    t = time.time()
    file_data = "./data/preprocessed_data/acquisition_{}/preprocessed.csv".format(acquisition)
    file_info = "./data/preprocessed_data/cycles_per_acquisition.csv"

    info = []
    with open(file_info, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            info.append(row)

    data = np.zeros([int(info[1][0]), int(info[0][acquisition - 1]) * Fs_prime], dtype=float)

    with open(file_data, 'r') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
        i = 0
        for row in reader:
            data[i][:] = row
            i += 1
    save_data = './images/preprocessed/acquisition_{}'.format(acquisition)
    if not os.path.exists(save_data):
        os.makedirs(save_data)

    if vis == 0:
        start = 0
        steps = 1000
        for i in range(int(len(data[0]) / steps)):
            x = np.linspace(start, start + steps, steps)

            data_visualization(data[0][start:start + steps],
                               data[1][start:start + steps], 
                               data[2][start:start + steps],
                               x, 'Accelerometer preprocessed', 
                               '/acc_section' + str(i), 
                               save_data, marker=True, 
                               fixed_range=rescale)

            data_visualization(data[3][start:start + steps], 
                               data[4][start:start + steps], 
                               data[5][start:start + steps],
                               x, 'Gyroscope preprocessed', 
                               '/gyro_section' + str(i), 
                               save_data, marker=True, 
                               fixed_range=rescale)
            start += steps

        start = 0
        if len(data) > 6:
            for i in range(int(len(data[0]) / steps)):
                x = np.linspace(start, start + steps, steps)
                data_visualization(data[6][start:start + steps], 
                                   data[7][start:start + steps], 
                                   data[8][start:start + steps],
                                   x, 'Angle preprocessed', 
                                   '/th_section' + str(i), 
                                   save_data, marker=True, 
                                   fixed_range=rescale)
                start += steps

    # WHOLE SEQUENCE
    start = 0
    steps = len(data[0])
    for i in range(int(len(data[0]) / steps)):
        x = np.linspace(start, start + steps, steps)

        data_visualization(data[0][start:start + steps], 
                           data[1][start:start + steps], 
                           data[2][start:start + steps],
                           x, 'Accelerometer preprocessed', 
                           '/whole_sequence_acc', 
                           save_data, marker=True,
                           fixed_range=rescale)

        data_visualization(data[3][start:start + steps], 
                           data[4][start:start + steps], 
                           data[5][start:start + steps],
                           x, 'Gyroscope preprocessed', 
                           '/whole_sequence_gyro', 
                           save_data, marker=True,
                           fixed_range=rescale)
        start += steps

    start = 0
    if len(data) > 6:
        for i in range(int(len(data[0]) / steps)):
            x = np.linspace(start, start + steps, steps)
            data_visualization(data[6][start:start + steps], 
                               data[7][start:start + steps], 
                               data[8][start:start + steps],
                               x, 'Angle preprocessed', 
                               '/whole_sequence_th', 
                               save_data, marker=True,
                               fixed_range=rescale)
            start += steps

    print('Data inspection of acquisition {:02d} performed in {0:.2f} seconds.'
                                            .format(acquisition,time.time()-t))


def video_flattening(video_vectors, cycles_per_acquisition, Fs_prime):
    t = time.time()
    ppc            = 1                 # points per cycle
    f_points       = np.zeros(shape=(video_vectors.shape[0], ppc * video_vectors.shape[1]//Fs_prime +len(cycles_per_acquisition)), dtype=float)
    f_points_prime = np.zeros(shape=(video_vectors.shape[0], video_vectors.shape[1] ), dtype=float)
    xx             = np.zeros(shape= ppc *video_vectors.shape[1] // Fs_prime +len(cycles_per_acquisition), dtype=float)

    current_path = './images/video_flattening'
    if not os.path.exists(current_path):         # create directory is not present
        os.makedirs(current_path)

    checkpoint = cycles_per_acquisition[0]
    filenames = ['','_check_1']
    for iii in range(1):
        index = 0
        checkpoint_index = 0
        for i in range(video_vectors.shape[1]//Fs_prime):
            for ii in range(ppc):
                xx[index] = i * Fs_prime + ii * Fs_prime//ppc
                f_points[:, index] = video_vectors[:,i * Fs_prime + ii * Fs_prime//ppc ]
                index += 1
            if i+1 == checkpoint and checkpoint_index < len(cycles_per_acquisition):
                xx[index] = (i+1) * Fs_prime -1
                f_points[:, index] = video_vectors[:, (i+1) * Fs_prime - 1]
                index += 1
                checkpoint_index += 1
                try:
                    checkpoint += cycles_per_acquisition[checkpoint_index]
                except IndexError:
                    print()

        # interpolation for better capturing of moving average
        x_prime = np.linspace(0, video_vectors.shape[1], video_vectors.shape[1])
        f_points_prime[0, :] = np.interp(x=x_prime, xp=xx, fp=f_points[0, :])
        f_points_prime[1, :] = np.interp(x=x_prime, xp=xx, fp=f_points[1, :])
        f_points_prime[2, :] = np.interp(x=x_prime, xp=xx, fp=f_points[2, :])

        # Visualization - f_p
        n = 1000
        plt.subplot(3, 1, 1)
        plt.plot(xx[0:ppc*n], f_points[0, 0:ppc*n], linewidth=0.15, 
                            linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], 
                    f_points_prime[0, 0:n * Fs_prime], linewidth=0.15)
        plt.title('Interpolation points')

        plt.subplot(3, 1, 2)
        plt.plot(xx[0:ppc*n], f_points[1, 0:ppc*n], linewidth=0.15, 
                            linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], 
                    f_points_prime[1, 0:n * Fs_prime], linewidth=0.15)

        plt.subplot(3, 1, 3)
        plt.plot(xx[0:ppc*n], f_points[2, 0:ppc*n], linewidth=0.15, 
                            linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], 
                    f_points_prime[2, 0:n * Fs_prime], linewidth=0.15)

        plt.savefig('{}/f_points_video{}.pdf'.format(current_path,filenames[iii]), 
                                                        dpi=150, transparent=False)
        plt.close()

        if iii == 0:
            threshold = [1.5, 1.5, 1.5]
            for i in range(3):
                 video_vectors[i, :] -= f_points_prime[i, :]

            for i in range(video_vectors.shape[1] // Fs_prime):
                index1 = Fs_prime * i
                index2 = Fs_prime * (i + 1)
                extreme = np.max(np.abs(video_vectors[:, index1: index2]), axis=1)
                for ii in range(3):
                    if extreme[ii] > threshold[ii]:
                        video_vectors[ii, index1: index2] /= (extreme[ii])

                #extreme1 = np.max(np.abs(video_vectors[:, index1: index2]), axis=1)
                #print(i, extreme, extreme1)

    print('Removal of trend in data collected from videos performed in {0:.2f} seconds.'.format(time.time()-t))
    return video_vectors


def plot_walk_statistics(**args):
    x = np.linspace(0, len(st_time), len(st_time))
    plt.subplot(3, 1, 1)
    plt.title('Walk statistics')
    plt.plot(x,st_time,linewidth=1,marker='o',markersize=3.5)
    plt.ylabel('Stance time')
    plt.subplot(3, 1, 2)
    plt.plot(x,sw_time,linewidth=1,marker='o',markersize=3.5)
    plt.ylabel('Swing time')
    plt.subplot(3, 1, 3)
    plt.plot(x,msv,linewidth=1,marker='o',markersize=3.5)
    plt.ylabel('z axis')
    ylim = ([0, 1, 0, 2])
    plt.xlabel('Step velocity')
    plt.savefig('./images/walk_statistics/stat_{}.pdf'.format(i),
                dpi=150, transparent=False)
    plt.close()


def plot_preprocessed_examples(**args):
    x = np.linspace(0, len(cycles_temp),len(cycles_temp))
    plt.title('Preprocessed data')
    plt.plot(x,cycles_temp,linewidth=1,marker='o',markersize=3.5)
    plt.savefig('{}{}/{}/cycle_{}.pdf'.format(save_fig_preproc,i + 1, name,h),
                dpi=150,transparent=False)
    plt.close()


def plot_raw_data(num_acquisitions, save_data, **args):
    steps = 1000
    for ii in range(num_acquisitions):
        save_data = './images/raw_data/acquisition_{}/'.format(ii + 1)
        if not os.path.exists(save_data):
            os.makedirs(save_data)

        start = 0
        for i in range(int(len(acc_x[ii]) / steps)):
            x = np.linspace(start, start + steps, steps)
            data_visualization(acc_x[ii][start:start + steps], 
                                acc_y[ii][start:start + steps], 
                                acc_z[ii][start:start+steps],
                                x, 'Accelerometer raw', 
                                '/acc_section' + str(i), 
                                save_data, 
                                marker=False, 
                                fixed_range=False)

            data_visualization(gyro_x[ii][start:start + steps], 
                                gyro_y[ii][start:start + steps], 
                                gyro_z[ii][start:start+steps],
                                x, 'Gyroscope raw', 
                                '/gyro_section' + str(i), 
                                save_data, 
                                marker=False, 
                                fixed_range=False)
            start += steps

        x = np.linspace(0, len(thx[ii]), len(thx[ii]))
        data_visualization(thx[ii][:], 
                            thy[ii][:], 
                            thz[ii][:], 
                            x, 'th raw', 
                            '/th_section',
                            save_data, 
                            marker=False, 
                            fixed_range=False)

        print('Saving representations of raw data, acquisition {} of {}.'
                                            .format(num_acquisitions,ii + 1))


def data_visualization(val_x, val_y, val_z, x, title,filename, path, marker, fixed_range):
    """
    This function produces an image file in the subdirectory Thesis/images
    containing a 3-dim subgraph representing part of the acquired signals

    Input values: val_x, val_y val_z represents the triaxial info of the sensors
    Ancillary variable: x is the dimension and position of the visualization window

    Function is used to gain more knowledge regarding the signals.
    """

    space = 25
    plt.subplot(3, 1, 1)
    plt.title(title)
    plt.plot(x, val_x, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green',linewidth=0.2, linestyle='--') 
                                        for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('x axis')
    plt.subplot(3, 1, 2)
    plt.plot(x, val_y, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green', linewidth=0.2, linestyle='--') 
                                        for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('y axis')
    plt.subplot(3, 1, 3)
    plt.plot(x, val_z, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green', linewidth=0.2, linestyle='--') 
                                        for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('z axis')
    plt.xlabel('Time t')
    plt.savefig(path + filename + '.pdf', dpi=150, transparent=False)
    plt.close()                                            