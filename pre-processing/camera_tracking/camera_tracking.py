# coding: utf-8

import sys
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import scipy.io as sio
from scipy import interpolate
import os
import glob
import time as time
from pathlib import Path

# from mpl_toolkits.mplot3d import Axes3D

help_message = '''\nUSAGE: camera_tracking.py [no input, sweep all videos in folders /data/raw_data/*] '''

# VISUALIZE RESULTS
VISUALIZE = 0

# COSTANTS
MIN_FEATURES = 1200  # min number of features  (previous value 1200)
THRESH_VAL = 0.9  # RANSAC threshold - optimal value depends on the video (try 1->5)
OUTLIERS_VAL = 10
ALPHA = 1  # ALPHA = 1 -> only new values
BETA = 1  # alto BETA -> meno sensibile a curve

# Intrinsic camera parameters

# RES: 576 X 720
mtx = np.array([[875.8470374, 0., 290.22010515],
                [0., 618.67901671, 348.7369155],
                [0., 0., 1.]], dtype=float)
invmtx = np.linalg.inv(mtx)
dist = np.array([[0.21486217, -0.47784865, -0.00159837, 0.00146397, -0.17732472]], dtype=float)


# Calculate optical flow and get rid of bad points
def featureTracking(old, new, p0, **lk_params):
    # Calculate optical flow - KLT
    p1, st, err = cv2.calcOpticalFlowPyrLK(old, new, p0, None, **lk_params)
    # Getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    indexCorrection = 0
    for i in range(len(st)):
        pt = p1[i - indexCorrection][0]
        if ((st[i] == 0) or (pt[0] < 0) or (pt[1] < 0)):
            if ((pt[0] < 0) or (pt[1] < 0)):
                st[i] = 0
            p0 = np.delete(p0, (i - indexCorrection), axis=0)
            p1 = np.delete(p1, (i - indexCorrection), axis=0)
            indexCorrection += 1

    return p0, p1


# Camera calibration + coordinates normalization
def calibration(frame, mtx, dist):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]  # crop the frame

    h, w = dst.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 270, 1.0)
    dst = cv2.warpAffine(dst, M, (w, h))

    return dst


# Return yaw, pitch and roll in degrees
def decomposeR(R):
    thx = math.atan2(R[2, 1], R[2, 2])
    thy = math.atan2(-R[2, 0], math.sqrt(math.pow(R[2, 1], 2) + math.pow(R[2, 2], 2)))
    thz = math.atan2(R[1, 0], R[0, 0])

    return math.degrees(thx), math.degrees(thy), math.degrees(thz)


# Return normalized coordinates (----- NOT NEEDED ------)
def pointNorm(p):
    p = cv2.convertPointsToHomogeneous(p).dot(invmtx)

    return cv2.convertPointsFromHomogeneous(p)


def checkVideoFrames(cap, n_frames):
    toKeep = []
    for i in range(n_frames):
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret == True and frame_gray.sum() < frame_gray.shape[0] * frame_gray.shape[1] * 255 * 0.9:
            toKeep.append(i)

    return np.array(toKeep)


# Valore dei frame con problemi viene rimpiazzato con il valore del frame accettato piu' vicino
def postProc(n_frames, toKeep, inputz):
    temp = np.zeros(n_frames)
    temp[toKeep] = np.array(inputz)
    index0 = np.where(temp == 0)[0]
    index1 = np.where(temp != 0)[0]
    for i in index0:
        idx = ((index1 - i)).min()
    if idx >= 0:
        temp[i] = temp[idx + i]
    else:
        idx = ((index1 - i)).max()
        temp[i] = temp[idx + i]
    return temp


def camera_tracking(force):
    """This function is used to retrieve the information from the video inside data/raw_data

    Input: force is used to force extraction on all acquisition events (True--> extraction on all videos, False -->
            extraction on new videos)

    Outpur: the output are the deltax/y/z and t, thx/y/z vectors

    Note: Output files are placed in the same folder of the video for latter preprocessing
          The format of the output is .mat """

    # Close all existing plots
    plt.close("all")

    if len(sys.argv[1:]) < 1:
        print(help_message)
    # input_video = raw_input("\nChoose video file: ")
    else:  # input passato da riga di comando
        input_video = sys.argv[1]

    # retrieve path where to find data
    head_path = os.path.abspath("camera_tracking.py")
    video_path = head_path[0:head_path.rfind('Code')] + "data/raw_data/**/*.mp4"

    num_dir = 0
    current_dir = 0
    for filename in glob.glob(video_path):
        num_dir += 1
    print("Total number of directory/video: " + str(num_dir))
    print("")
    already_processed = 0
    for filename in glob.glob(video_path):
        current_dir += 1
        t = time.time()
        deltafile = filename[0:filename.rfind('/') + 1] + 'deltax.mat'

        if not (Path(deltafile).is_file()) or force:
            if already_processed != 0:
                print("Video " + str(already_processed) + " of " + str(num_dir) + " already processed")
                already_processed = 0
            # Input video
            cap = cv2.VideoCapture(filename)
            n_frames0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            video_info = pd.read_csv(filename[0:filename.rfind('/') + 1] + 'video_info.txt', sep='\t', header=None)
            n_frames = int(video_info.iat[2, 1])

            # Check which frames are good
            toKeep = checkVideoFrames(cap, n_frames0)
            counter = 0

            # Take first two useful frames and extract keypoints

            cap.set(1, toKeep[counter]);
            counter += 1
            ret, old_frame0 = cap.read()
            old_frame0 = calibration(old_frame0, mtx, dist)

            cap.set(1, toKeep[counter]);
            counter += 1
            ret, old_frame1 = cap.read()
            old_frame1 = calibration(old_frame1, mtx, dist)

            # color --> gray
            old_gray0 = cv2.cvtColor(old_frame0, cv2.COLOR_BGR2GRAY)
            old_gray1 = cv2.cvtColor(old_frame1, cv2.COLOR_BGR2GRAY)

            # Keypoints extraction - SIFT/FAST
            sift = cv2.xfeatures2d.SIFT_create()
            # fast = cv2.FastFeatureDetector_create( threshold = 20, nonmaxSuppression = True)
            # kp = fast.detect(old_frame0, None)
            kp = sift.detect(old_frame0, None)
            #print ("\n" + str(len(kp)) + " features extracted\n")
            p0 = np.zeros((len(kp), 1, 2), dtype=np.float32)
            for i in range(len(kp)):
                p0[i, 0, 0], p0[i, 0, 1] = kp[i].pt

            # Optical flow parameters
            # ORIGINAL: winSize = (21,21) maxLevel = 2
            lk_params = dict(winSize=(21, 21), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            # Track features
            p0, p1 = featureTracking(old_gray0, old_gray1, p0, **lk_params)
            # Compute fundamental matrix for first 2 frames
            F, maskr = cv2.findFundamentalMat(p0, p1, method=cv2.FM_RANSAC, param1=THRESH_VAL, param2=0.999)
            # Compute essential matrix for first 2 frames
            Er = mtx.transpose().dot(F).dot(mtx)  # good for R
            Et, maskt = cv2.findEssentialMat(p0, p1, method=cv2.RANSAC, focal=1, threshold=THRESH_VAL,
                                             prob=0.999)  # good for t
            # Recover R and t from E
            points, Rr, tr, mask = cv2.recoverPose(Er, p0, p1, focal=1, mask=maskr)
            points, Rt, tt, mask = cv2.recoverPose(Et, p0, p1, focal=1, mask=maskt)
            # Variables initialization
            R_f = Rr  # rotation matrix
            t_f = tr  # tt  # translation matrix

            R_frpred = Rr.copy()
            Rr_pred = Rr.copy()
            tr_pred = tr.copy()
            Ralways = Rr.copy()  #  always updated

            R_ft = Rt.copy()
            tt_pred = tt.copy()  # transaltion matrix of the previous frame
            R_ftpred = Rt.copy()  # rotation matrix of the previous frame
            Rt_pred = Rt.copy()

            # Decompose rotation matrix -> pitch, roll, yaw
            thx_old, thy_old, thz_old = decomposeR(R_f)
            deltax = 0
            deltay = 0
            deltaz = 0
            thx = 0
            thy = 0
            thz = 0
            deltax_fin = []
            deltax_fin.append(0)
            deltay_fin = []
            deltay_fin.append(0)
            deltaz_fin = []
            deltaz_fin.append(0)
            thx_fin = []
            thx_fin.append(0)
            thy_fin = []
            thy_fin.append(0)
            thz_fin = []
            thz_fin.append(0)
            t_fin = []
            t_fin.append(t_f)

            # Update variables
            old_gray0 = old_gray1.copy()
            p0 = p1

            # Do the same for all other frames
            while (counter < len(toKeep)):
                cap.set(1, toKeep[counter])
                ret, frame = cap.read()
                frame = calibration(frame, mtx, dist)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Tracks extracted features
                p0, p1 = featureTracking(old_gray0, frame_gray, p0, **lk_params)
                F, maskr = cv2.findFundamentalMat(p0, p1, method=cv2.FM_RANSAC, param1=THRESH_VAL, param2=0.999)
                # Recover essential matrix E,R and t
                Er = mtx.transpose().dot(F).dot(mtx)  # good for R
                Et, maskt = cv2.findEssentialMat(p0, p1, method=cv2.RANSAC, focal=1, threshold=THRESH_VAL,
                                                 prob=0.999)  # good for t
                points, Rr, tr, mask = cv2.recoverPose(Er, p0, p1, focal=1, mask=maskr)
                points, Rt, tt, mask = cv2.recoverPose(Et, p0, p1, focal=1, mask=maskt)
                # Update R_f and t_f
                '''
                if ((tt[2] > BETA * tt[0]) and (tt[2] >  BETA * tt[1])):
                    t_f = t_f - (ALPHA * R_ft.dot(tt) + (1 - ALPHA) * R_ftpred.dot(tt_pred))
                    R_ft = Rt.dot(R_ft)
                R_f = Rr.dot(R_f) # final rotation matrix
                '''
                if ((tr[2] > BETA * tr[0]) and (tr[2] > BETA * tr[1])):
                    t_f = t_f + (ALPHA * R_f.dot(tr) + (1 - ALPHA) * R_frpred.dot(tr_pred))
                    R_f = Rr.dot(R_f)
                Ralways = Rr.dot(Ralways)

                # decompose rotation matrix
                thx, thy, thz = decomposeR(Ralways)  #  Ralways

                t_fin.append(t_f)
                thx_fin.append(thx)
                thy_fin.append(thy)
                thz_fin.append(thz)

                # Check if I have some strange values
                if math.fabs(thx - thx_old) < OUTLIERS_VAL:
                    deltax = thx - thx_old
                    deltax_fin.append(deltax)
                else:
                    deltax_fin.append(deltax)  # append old value
                if math.fabs(thy - thy_old) < OUTLIERS_VAL:
                    deltay = thy - thy_old
                    deltay_fin.append(deltay)
                else:
                    deltay_fin.append(deltay)  # append old value
                if math.fabs(thz - thz_old) < OUTLIERS_VAL:
                    deltaz = thz - thz_old
                    deltaz_fin.append(deltaz)
                else:
                    deltaz_fin.append(deltaz)  # append old value

                # Triggered if more keypoints needed
                if len(p0) < MIN_FEATURES:
                    # kp = fast.detect(frame_gray, None)
                    kp = sift.detect(old_gray0, None)
                    #print("\n" + str(len(kp)) + " features extracted\n")
                    p0 = np.zeros((len(kp), 1, 2), dtype=np.float32)
                    for i in range(len(kp)):
                        p0[i, 0, 0], p0[i, 0, 1] = kp[i].pt
                    p0, p1 = featureTracking(old_gray0, frame_gray, p0, **lk_params)

                # Visualize results if VISUALIZE is flagged
                if VISUALIZE:
                    # Plot translation in XZ plane
                    plt.figure(1)
                    plt.plot(t_f[0], t_f[2], 'ko')
                    plt.xlabel('x')
                    plt.ylabel('z')
                    plt.title('XZ translation')
                    plt.pause(0.001)
                    # Plot delta roll, pitch, yaw
                    ylims = np.array([-5, 5])
                    plt.figure(2, figsize=(16, 8))
                    plt.subplot(311)
                    # plt.ylim(ylims)
                    plt.plot(counter, deltax, 'ko')  # yaw
                    plt.title('delta_Yaw')
                    plt.subplot(312)
                    # plt.ylim(ylims)
                    plt.plot(counter, deltay, 'ko')  # pitch
                    plt.title('delta_Pitch')
                    plt.subplot(313)
                    # plt.ylim(ylims)
                    plt.plot(counter, deltaz, 'ko')  # roll
                    plt.title('delta_Roll')
                    plt.pause(0.001)
                    cv2.imshow('Video', frame)
                    k = cv2.waitKey(1)
                    #print ('frame ' + str(toKeep[counter]) + ' of ' + str(n_frames))
                #else:
                #    print ('frame ' + str(toKeep[counter]) + ' of ' + str(n_frames))

                # Update variables
                counter += 1
                old_gray0 = frame_gray.copy()
                p0 = p1.copy()

                thx_old = thx
                thy_old = thy
                thz_old = thz

                R_frpred = R_f.copy()
                tr_pred = tr.copy()
                Rr_pred = Rr.copy()

                R_ftpred = R_ft.copy()
                tt_pred = tt.copy()
                Rt_pred = Rt.copy()

            # post processing
            deltax_fin = postProc(n_frames, toKeep[1:], deltax_fin)
            deltay_fin = postProc(n_frames, toKeep[1:], deltay_fin)
            deltaz_fin = postProc(n_frames, toKeep[1:], deltaz_fin)
            thx_fin = postProc(n_frames, toKeep[1:], thx_fin)
            thy_fin = postProc(n_frames, toKeep[1:], thy_fin)
            thz_fin = postProc(n_frames, toKeep[1:], thz_fin)

            # save matrices

            sio.savemat(filename[0:filename.rfind('/') + 1] + 'deltax.mat', {'deltax': deltax_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 'deltay.mat', {'deltay': deltay_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 'deltaz.mat', {'deltaz': deltaz_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 'thx.mat', {'thx': thx_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 'thy.mat', {'thy': thy_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 'thz.mat', {'thz': thz_fin})
            sio.savemat(filename[0:filename.rfind('/') + 1] + 't.mat', {'t_f': t_fin})
            print("Video " + str(current_dir) + " of " + str(num_dir) + " processed in " +"{0:.2f}".format(time.time()-t) + "s")
            # if not(VISUALIZE):
            # 	t_fin = np.array(t_fin)
            # 	t_fin.flatten()
            # 	cv2.destroyAllWindows()
            # 	plt.figure(3)
            # 	plt.plot(t_fin[:,0], t_fin[:,2],'ko')
            # 	plt.show()
        else:
            already_processed += 1

        if already_processed == num_dir:
            print('All videos have been already processed. (set force=True to re-process)\n')


if __name__ == '__main__':
    """The information extraction function can be launched in this stand alone program, or
    (as it is done in the preprocessing.py) can be imported for remote run."""
    camera_tracking(force=False)

