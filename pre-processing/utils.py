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

Fs_prime = 200                       # Hz, sampling frequency

def read_data(VISUALIZE, which_data):
    """This function is used to upload all raw data from each diretory in data/raw_data
    absolute path are extensively used.

    Outputs: acc_timestamp, gyro_timestamp, frame_timestamp, acc_x, acc_y, acc_z,
                            gyro_x, gyro_y, gyro_z, deltax, deltay, deltaz
    Type of outpus is list of arrays, each array is a different acquisition.

    Note: There is no need to manually put usefull files in specific directories, just copy all directory collected by
    the activity logger video application to the  data/raw_data folder."""

    accelerometer = ''
    gyroscope     = ''
    magnetometer  = ''
    video_path    = ''

    accelerometer = "./data/{}/**/accellerometer_*.csv".format(which_data)
    gyroscope     = "./data/{}/**/gyroscope_*.csv".format(which_data)
    magnetometer  = "./data/{}/**/magnetometer_*.csv".format(which_data)
    video_path    = "./data/{}/**/*.mp4".format(which_data)

    # Load accelerometers data -----------------------------------------------------------------------------------------
    acc_timestamp, acc_x, acc_y, acc_z = [], [], [], []

    num_acquisitions = 0
    for filename in glob.glob(accelerometer):
        acc = pd.read_csv(filename, sep='\t', header=0)
        acc_timestamp.append(np.array(acc.acc_timestamp, dtype=float))  # timestamps in ns
        acc_x.append(np.array(acc.acc_x))
        acc_y.append(np.array(acc.acc_y))
        acc_z.append(np.array(acc.acc_z))
        num_acquisitions += 1

    # Load gyroscope data ----------------------------------------------------------------------------------------------
    gyro_timestamp, gyro_x, gyro_y, gyro_z = [], [], [], []
    for filename in glob.glob(gyroscope):
        gyro = pd.read_csv(filename, sep='\t', header=0)
        gyro_timestamp.append(np.array(gyro.gyro_timestamp, dtype=float))  # timestamps in ns
        gyro_x.append(np.array(gyro.gyro_x))
        gyro_y.append(np.array(gyro.gyro_y))
        gyro_z.append(np.array(gyro.gyro_z))

    # Load magnetometer data -------------------------------------------------------------------------------------------
    magn_timestamp, magn_x, magn_y, magn_z = [], [], [], []
    for filename in glob.glob(magnetometer):
        magn = pd.read_csv(filename, sep='\t', header=0)
        magn_timestamp.append(np.array(magn.magn_timestamp, dtype=float))  # timestamps in ns
        magn_x.append(np.array(magn.magn_x))
        magn_y.append(np.array(magn.magn_y))
        magn_z.append(np.array(magn.magn_z))

    # Load video data --------------------------------------------------------------------------------------------------
    frame_timestamp =[]
    for filename in glob.glob("./data/{}/**/frame_timestamp.csv".format(which_data)):
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
        'acc_timestamp': acc_timestamp, 
        'gyro_timestamp': gyro_timestamp, 
        'frame_timestamp': frame_timestamp, 
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

    # Data visualization, folder Code/images/preprocessing -------------------------------------------------------------
    if VISUALIZE:
        plot_raw_data(num_acquisitions, which_data, **data)

    return data


def write_data(name,value,which_data):
    """This function is used to save all preprocessed data from each diretory in data/preprocessed_data
        absolute path are extensively used.

        Outputs: acc_timestamp, gyro_timestamp, frame_timestamp, acc_x, acc_y, acc_z,
                                gyro_x, gyro_y, gyro_z, deltax, deltay, deltaz

        For each acquisition a different sub directory is created /data/preprocessed_data/acquisition_*
        Type of outpus is csv files.

        Note: """

    if which_data:
        save_path = './data/preprocessed_events_data/acquistion_'
    else:
        save_path = './data/preprocessed_data/acquistion_'

    for i in range(len(name)):
        counter = 0
        for ii in range(len(value[i])):
            counter+=1
            current_path = save_path+str(ii)
            if not os.path.exists(current_path):
                os.makedirs(current_path)

            # -- Salvataggio su file --
            with open('{}/{}'.format(current_path,name[i]), 'wb') as file:
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(value[i][ii])


def preprocessing_y(sig0, sig_timestamp0, sig1, sig_timestamp1, Fs_prime, sig_cycles, cycles_timestamp, visualize):
    """This function extract the waling cycles using the accelerometer and gyroscope information along y axis.
     The implemented procedure is explained in the Rochester paper, provided with the documentation."""

    sig_prime0, sig_timestamp_prime0 = interpolation(sig0, sig_timestamp0, Fs_prime)
    sig_prime1, sig_timestamp_prime1 = interpolation(sig1, sig_timestamp1, Fs_prime)


    # ------------- STEP IDENTIFICATION ---------------------
    cwt_order = 40  # ordine della wavelet transform
    sig_prime0 = matlab.double(sig_prime0.tolist())
    sig_prime1 = matlab.double(sig_prime1.tolist())

    # richiamo funzioni matlab
    IC, FC = eng.ICFC(sig_prime0, matlab.double([Fs_prime]), matlab.double([cwt_order]), matlab.double([visualize]),
                      nargout=2)  # trova inizio e fine di ciascun passo
    IC = np.array(IC[0], dtype=float)  # initial contact di ogni passo
    FC = np.array(FC[0], dtype=float)  # final contact di ogni passo
    IC = IC[:]
    FC = FC[:]

    fc_lr = 1        # frequenza di taglio per IClr
    ICleft, ICright = eng.IClr(sig_prime1, matlab.double([Fs_prime]), matlab.double([fc_lr]),
                               matlab.double(IC.tolist()), matlab.double([visualize]), nargout=2)  # trova passi sx e dx

    ICleft = np.array(ICleft[0], dtype=float)  # initial contact dei passi di sx
    ICright = np.array(ICright[0], dtype=float)  # initial contact dei passi di dx

    # ------ CYCLES EXTRACTION -------
    # estraggo cicli da segnale "sig_cycles" per visualizzare risultato
    index_values = ICleft.astype(int)
    fixed_length = Fs_prime  # riporto ogni ciclo a lunghezza fissata di un secondo (= Fs campioni)

    args = {'input':sig_cycles, 
            'timestamp':cycles_timestamp, 
            'index_values':index_values, 
            'fixed_length':fixed_length, 
            'Fs_prime':Fs_prime,
            'filter_vis':False
            }

    cycles, index_cycles = cycles_extraction(**args)
    m_cycles = np.mean(cycles, axis=0)  # valor medio ciclo camminata

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
    w *= 100
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.xscale('log')
    #plt.title('Butterworth filter frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.xlim()
    plt.ylabel('Amplitude (dB)')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(normalCutoff*100, color='green')  # cutoff frequency
    plt.savefig('./images/Butterworth_{}.pdf'.format(order), dpi=150, transparent=False)
    plt.close()


def interpolation(input, timestamp, Fs_prime):
    """This functions is used to resample the raw data using Fs_prime as frequency for the interpolation procedure.
    Moreover it estimates the average sampling frequency of the raw signal."""

    n_samples = timestamp.shape[0]
    Fs = 1 / (np.mean(timestamp[1: n_samples] - timestamp[0: n_samples - 1]) * 1e-9)  # data mean sampling frequency
    n_samples_prime = int(np.round(n_samples * Fs_prime / Fs))
    timestamp_prime = np.linspace(timestamp.min(), timestamp.max(), n_samples_prime)  # new timestamp
    input_prime = np.interp(x=timestamp_prime, xp=timestamp, fp=input)

    return input_prime, timestamp_prime


def cycles_extraction(input, timestamp, index_values, fixed_length, Fs_prime, filter_vis):
    """This function performs the interpolation procedure, filtering of data and cycle extraction."""
    # Interpolation
    input_prime, timestamp_prime0 = interpolation(input, timestamp, Fs_prime)
    # LP filtering
    nyq = 0.5 * Fs_prime
    normalCutoff = 40 / nyq
    b, a = signal.butter(10, normalCutoff, btype='low', analog=False)
    if filter_vis:
        butterworth(normalCutoff, b, a, 10)
    input_filt = signal.filtfilt(b, a, input_prime)

    # Cycles extraction ------------------------
    cycles = []
    index_cycles = []
    for i in range(len(index_values) - 1):
        y_temp = input_filt[int(index_values[i]): int(index_values[i + 1])]
        x_temp = timestamp_prime0[int(index_values[i]): int(index_values[i + 1])]
        timestamp_prime1 = np.linspace(x_temp.min(), x_temp.max(), fixed_length)  # new timestamp
        cycles.append(np.interp(x=timestamp_prime1, xp=x_temp, fp=y_temp))
        index_cycles.append(timestamp_prime1)

    cycles = np.vstack(cycles)

    return cycles, index_cycles


def smooth(x, interval):
    dft = scipy.fftpack.rfft(x)[0: interval]
    idft = scipy.fftpack.irfft(dft, n=len(x))
    return idft


def stance_swing_time(ICs, FCs, Fs):
    FCstart = [n for n, i in enumerate(FCs) if i > ICs[0]][0]  # FC del primo passo
    FCtemp = FCs[FCstart: len(FCs): 2]
    FCtemp = FCtemp[0: len(ICs) - 1]
    stance_time = (FCtemp - ICs[:-1]) / Fs
    stride_time = (ICs[1:] - ICs[:-1]) / Fs
    swing_time = stride_time - stance_time

    return stance_time, swing_time


def step_vel(IC, index, acc_y, acc_timestamp, l, Fs):
    ICstart = np.where(IC == index[0])[0]
    ICend = np.where(IC == index[-1])[0]

    ICvel = IC[ICstart[0]: (ICend[0] + 1)]
    acc_y_interp, _ = interpolation(acc_y, acc_timestamp, Fs)
    cutOff = 0.7  # high pass filter per eliminare drift in integrazione

    # funzione matlab per ricavare segnale di spostamento verticale
    y_space = eng.space(matlab.double(acc_y_interp.tolist()), matlab.double([Fs]), matlab.double([cutOff]))
    y_space = np.array(y_space[0], dtype=float)

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


def psd_comparison(sig1,sig2,sig3,sig_timestamp1,sig_timestamp2,sig_timestamp3,dir):
    sig1_prime1, sig_timestamp_prime1 = interpolation(sig1, sig_timestamp1, Fs_prime)
    sig2_prime2, sig_timestamp_prime2 = interpolation(sig2, sig_timestamp2, Fs_prime)
    sig3_prime3, sig_timestamp_prime3 = interpolation(sig3, sig_timestamp3, Fs_prime)

    perseg = 200
    fs=Fs_prime
    noverlap = fs//2
    # p = 20 * np.log10(np.abs(np.fft.rfft(sig)))     # psd of original signal
    x1, p1 = signal.welch(x=sig1_prime1, fs=fs, window="hanning", nperseg=perseg, axis=0, detrend=False, noverlap = noverlap)
    x2, p2 = signal.welch(x=sig2_prime2, fs=fs, window="hanning", nperseg=perseg, axis=0, detrend=False, noverlap = noverlap)
    x3, p3 = signal.welch(x=sig3_prime3, fs=fs, window="hanning", nperseg=perseg, axis=0, detrend=False, noverlap = noverlap)
    p1 = 20 * np.log10(p1)
    p2 = 20 * np.log10(p2)
    p3 = 20 * np.log10(p3)
    # x, p = signal.periodogram(x=sig, fs=100, window='hann', nfft=200 )


    path = './images/Filtering/{}/'.format(dir)
    if not os.path.exists(path):
        os.makedirs(path)

    #x= np.linspace(0,2*len(p1),len(p1))
    plt.semilogx(x1, p1, linewidth=1)
    plt.semilogx(x2, p2, linewidth=1, color='red')
    plt.semilogx(x3, p3, linewidth=1, color='green')
    plt.ylabel('Power Spectral Density [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.legend(['x-axis acc','y-axis acc','z-axis acc'])
    plt.grid(True,which='both')
    plt.axis([0, 100, -100, 40])
    plt.savefig(path + 'PSD_accelerometer.pdf', dpi=150, transparent=False)

    plt.close()


def psd_visualization(sig, f_cut, name, dir):

    perseg = 130
    #p = 20 * np.log10(np.abs(np.fft.rfft(sig)))     # psd of original signal
    x, p = signal.welch(x=sig, fs=100, window="hanning",nperseg=perseg, axis=0,detrend=False)
    p = 20 * np.log10(p)
    #x, p = signal.periodogram(x=sig, fs=100, window='hann', nfft=200 )

    nyq = 0.5 * Fs_prime                                # Filter parameters
    normalCutoff = f_cut / nyq
    b, a = signal.butter(10, normalCutoff, btype='low', analog=False)    # filter definition
    sig1 = signal.filtfilt(b, a, sig)               # filtered signal
    xx, pp = signal.welch(x=sig1, fs=50, window="hanning",nperseg=perseg, axis=0,detrend=False)
    pp = 20 * np.log10(pp)
    #xx, pp = signal.periodogram(x=sig1, fs=0.1, window='hann', nfft=200)
    #pp = 20 * np.log10(np.abs(np.fft.rfft(sig1)))   # psd of filtered signal


    path = './images/Filtering/{}/'.format(dir)
    if not os.path.exists(path):
        os.makedirs(path)

    x = np.linspace(0, len(p), len(p))
    xx = np.linspace(0, len(pp), len(pp))
    plt.subplot(2, 1, 1)
    plt.title('Power Spectral Density')
    plt.semilogx(x, p, linewidth=1)
    plt.ylabel('Signal dB')
    plt.axis([0, 60, -60, 40])
    plt.subplot(2, 1, 2)
    plt.semilogx(xx, pp, linewidth=1)
    plt.ylabel('Signal Filtered dB')
    plt.xlabel('Frequency (Hz)')
    plt.axis([0, 60, -60, 40])
    plt.savefig(path +'PSD_'+name+'.pdf', dpi=150, transparent=False)

    plt.close()


    x = np.linspace(0,Fs_prime,Fs_prime)
    i=5
    plt.subplot(2, 1, 1)
    plt.title('Comparison Filtering')
    plt.plot(x, sig[Fs_prime*i:Fs_prime*(i+1)], linewidth=0.5)
    plt.ylabel('Signal')
    plt.subplot(2, 1, 2)
    plt.plot(x, sig1[Fs_prime*i:Fs_prime*(i+1)], linewidth=0.5)
    plt.ylabel('Signal Filtered')
    plt.xlabel('Time (s)')
    plt.savefig(path +'Filtering_comparison_'+name+'.pdf', dpi=150, transparent=False)
    plt.close()

    x = np.linspace(0, Fs_prime, Fs_prime)
    i = 5
    plt.plot(x, sig[Fs_prime * i:Fs_prime * (i + 1)], linewidth=0.5)
    plt.plot(x, sig1[Fs_prime * i:Fs_prime * (i + 1)], linewidth=0.5, color='red')
    plt.legend(['Signal','Signal Filtered'])
    plt.xlabel('Time t')
    plt.savefig(path + 'Filtering_comparison_sovrapposti' + name + '.pdf', dpi=150, transparent=False)
    plt.close()


def data_inspection(acquisition, rescale, vis, which_data):
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
    file_data = ''
    file_info = ''

    root = './data/preprocessed_{}'.format(which_data[4:])
    file_data = '{}/acquisition_{}/preprocessed.csv'.format(root,acquisition)
    file_info = '{}/cycles_per_acquisition.csv'.format(root)

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

    save_data = ''
    if which_data =='raw_events_data':
        save_data = './images/preprocessed_events/acquisition_' + str(acquisition)
    elif which_data == 'raw_data':
        save_data = './images/preprocessed/acquisition_'+str(acquisition)
    elif which_data == 'raw_data_classifier':
        save_data = './images/preprocessed_classifier/acquisition_' + str(acquisition)
    elif which_data == 'raw_data_test':
        save_data = './images/preprocessed_test/acquisition_' + str(acquisition)
    elif which_data == 'raw_data_test_events':
        save_data = './images/preprocessed_test_events/acquisition_' + str(acquisition)

    if not os.path.exists(save_data):
        os.makedirs(save_data)

    if vis == 0:
        start = 0
        steps = 1000
        for i in range(int(len(data[0]) / steps)):
            x = np.linspace(start, start + steps, steps)

            data_visualization( data[0][start:start + steps], 
                                data[1][start:start + steps], 
                                data[2][start:start + steps],
                                x, 'Accelerometer preprocessed', 
                                '/acc_section' + str(i), 
                                save_data, marker=True, 
                                fixed_range=rescale)

            data_visualization( data[3][start:start + steps], 
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
                data_visualization( data[6][start:start + steps], 
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

        data_visualization( data[0][start:start + steps], 
                            data[1][start:start + steps], 
                            data[2][start:start + steps],
                            x, 'Accelerometer preprocessed', 
                            '/whole_sequence_acc', 
                            save_data, marker=True,
                            fixed_range=rescale)

        data_visualization( data[3][start:start + steps], 
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
            data_visualization( data[6][start:start + steps], 
                                data[7][start:start + steps], 
                                data[8][start:start + steps],
                                x, 'Angle preprocessed', 
                                '/whole_sequence_th', 
                                save_data, marker=True,
                                fixed_range=rescale)
            start += steps

    print('Data inspection of acquisition {:02d} performed in {:02.2f} seconds.'
                                                .format(acquisition,time.time()-t))


def video_flattening(video_vectors, cycles_per_acquisition, which_data):
    sum_cycles = np.zeros(shape=(len(cycles_per_acquisition)), dtype=float)
    for i in range(len(cycles_per_acquisition)-1):
        sum_cycles[i+1] = sum_cycles[i]+cycles_per_acquisition[i]
    sum_cycles *= Fs_prime

    ppc = 1        # points per cycle
    f_points       = np.zeros(shape=(video_vectors.shape[0], \
                                     ppc * video_vectors.shape[1]//Fs_prime +len(cycles_per_acquisition)), dtype=float)
    f_points_prime = np.zeros(shape=(video_vectors.shape[0], video_vectors.shape[1] ), dtype=float)
    xx             = np.zeros(shape= ppc *video_vectors.shape[1] // Fs_prime +len(cycles_per_acquisition), dtype=float)

    current_path = ''
    if which_data =='raw_events_data':
        current_path = './images/video_events_flattening'
    elif which_data =='raw_data':
        current_path = './images/video_flattening'
    elif which_data =='raw_data_classifier':
        current_path = './images/video_classifier_flattening'
    elif which_data =='raw_data_test':
        current_path = './images/video_test_flattening'
    elif which_data =='raw_data_test_events':
        current_path = './images/video_test_events_flattening'

    if not os.path.exists(current_path):         # create directory is not present
        os.makedirs(current_path)

    checkpoint = cycles_per_acquisition[0]
    filenames = ['','_check_1']
    for iii in range(2):
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
                    pass

        # interpolation for better capturing of moving average
        x_prime = np.linspace(0, video_vectors.shape[1], video_vectors.shape[1])
        f_points_prime[0, :] = np.interp(x=x_prime, xp=xx, fp=f_points[0, :])
        f_points_prime[1, :] = np.interp(x=x_prime, xp=xx, fp=f_points[1, :])
        f_points_prime[2, :] = np.interp(x=x_prime, xp=xx, fp=f_points[2, :])


        # Visualization - f_p
        n = 500
        if n*Fs_prime > sum_cycles[-2]:
            n = int(sum_cycles[-2]//Fs_prime)
        space = 10
        plt.subplot(3, 1, 1)
        plt.plot(xx[0:ppc*n], f_points[0, 0:ppc*n], linewidth=0.15, linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], f_points_prime[0, 0:n * Fs_prime], linewidth=0.15)
        idx=0
        while  sum_cycles[idx] <= n*Fs_prime and idx<len(sum_cycles):
            plt.axvline(xx[0]+sum_cycles[idx], color='green', linewidth=0.2, linestyle='--')
            idx +=1
        plt.xlim((-space, n*Fs_prime+space))
        if iii == 1:
            plt.ylim((-0.01,0.01))
        plt.ylabel('Roll')

        plt.subplot(3, 1, 2)
        plt.plot(xx[0:ppc*n], f_points[1, 0:ppc*n], linewidth=0.15, linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], f_points_prime[1, 0:n * Fs_prime], linewidth=0.15)
        idx = 0
        while sum_cycles[idx] <= n * Fs_prime and idx<len(sum_cycles):
            plt.axvline(xx[0] + sum_cycles[idx], color='green', linewidth=0.2, linestyle='--')
            idx += 1
        plt.ylabel('Pitch')
        plt.xlim((-space, n * Fs_prime + space))
        if iii == 1:
            plt.ylim((-0.01,0.01))
        plt.subplot(3, 1, 3)
        plt.plot(xx[0:ppc*n], f_points[2, 0:ppc*n], linewidth=0.15, linestyle='', marker='o', markersize=0.2)
        plt.plot(x_prime[0:n * Fs_prime], f_points_prime[2, 0:n * Fs_prime], linewidth=0.15)
        idx = 0
        while sum_cycles[idx] <= n * Fs_prime and idx<len(sum_cycles):
            plt.axvline(xx[0] + sum_cycles[idx], color='green', linewidth=0.2, linestyle='--')
            idx += 1
        plt.ylabel('Yaw')
        plt.xlim((-space, n * Fs_prime + space))
        if iii == 1:
            plt.ylim((-0.01,0.01))
        plt.savefig(current_path + '/f_points_video'+filenames[iii]+'.pdf', dpi=150, transparent=False)
        plt.close()

        if iii == 0:
            threshold = [1.5, 1.5, 1.5]
            for i in range(3):
                 video_vectors[i, :] -= f_points_prime[i, :]

            # for i in range(video_vectors.shape[1] // Fs_prime):
            #     index1 = Fs_prime * i
            #     index2 = Fs_prime * (i + 1)
            #     extreme = np.max(np.abs(video_vectors[:, index1: index2]), axis=1)
            #     for ii in range(3):
            #         if extreme[ii] > threshold[ii]:
            #             video_vectors[ii, index1: index2] /= (extreme[ii])

                #extreme1 = np.max(np.abs(video_vectors[:, index1: index2]), axis=1)
                #print(i, extreme, extreme1)



    return video_vectors


def plot_walk_statistics(msv,st_time,sw_time,i):
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


def plot_preprocessed_examples(save_fig_preproc, cycles_temp, name, h, i):
    x = np.linspace(0, len(cycles_temp),len(cycles_temp))
    plt.title('Preprocessed data')
    plt.plot(x,cycles_temp,linewidth=1,marker='o',markersize=3.5)
    root = '{}{}/{}/'.format(save_fig_preproc,i + 1,name)
    if not os.path.exists(root):
        os.makedirs(root)
    plt.savefig('{}/cycle_{}.pdf'.format(root,h),
                dpi=150,transparent=False)
    plt.close()


def plot_raw_data(num_acquisitions, which_data, **kwargs):
    acc_x = kwargs['acc_x'] 
    acc_y = kwargs['acc_y'] 
    acc_z = kwargs['acc_z']  
    gyro_x = kwargs['gyro_x']  
    gyro_y = kwargs['gyro_y'] 
    gyro_z = kwargs['gyro_z']  
    thx = kwargs['thx']  
    thy = kwargs['thy']  
    thz = kwargs['thz']

    steps = 1000
    for ii in range(num_acquisitions):
        save_data = './data/preprocessed_{}/acquisition_{}/'.format(which_data[4:],ii+1)
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
                                        .format(ii + 1,num_acquisitions))
    print()


def data_visualization(val_x, val_y, val_z, x, title,filename, path, marker, fixed_range):
    """This function produces an image file in the subdirectory Thesis/images
    containing a 3-dim subgraph representing part of the acquired signals

    Input values: val_x, val_y val_z represents the triaxial info of the sensors
    Ancillary variable: x is the dimension and position of the visualization window

    Function is used to gain more knowledge regarding the signals."""

    space = 5
    plt.subplot(3, 1, 1)
    plt.title(title)
    plt.plot(x, val_x, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green',linewidth=0.2, linestyle='--') for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('x axis')
    plt.subplot(3, 1, 2)
    plt.plot(x, val_y, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green', linewidth=0.2, linestyle='--') for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('y axis')
    plt.subplot(3, 1, 3)
    plt.plot(x, val_z, linewidth=0.2)
    if fixed_range:
        plt.axis([x[0] - space, x[-1] + space, 0, 1])
    if marker:
        [plt.axvline(vertical*200+x[0], color='green', linewidth=0.2, linestyle='--') for vertical in range(int(len(val_x)/200)+1)]
    plt.ylabel('z axis')
    plt.xlabel('Time t')
    plt.savefig(path + filename + '.pdf', dpi=150, transparent=False)
    plt.close()


def setup_folders():
    root_images = './images/'
    folder_list = [ root_images, 
                    root_images + 'walk_statistics/',
                    root_images + 'matlab images/',
                    root_images + 'preprocessed/'
                    ]
    for f in folder_list:
        if not os.path.exists(f):
            os.makedirs(f)