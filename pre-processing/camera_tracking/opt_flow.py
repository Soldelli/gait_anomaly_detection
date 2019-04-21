# coding: utf-8
# %load MotionFlowProva.py

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import simplejson

# input video
cap = cv2.VideoCapture('video.mp4')

n_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# Harris corner detection parameters
max_corners = 50
feature_params = dict( useHarrisDetector = True, k = 0.04,
                      maxCorners = max_corners, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
# Optical flow parameters
lk_params = dict( winSize  = (15,15), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# CHOOSE WHETHER TO USE HARRIS OR SIFT KEYPOINTS:
HARRIS = 0
SIFT = 1

if HARRIS:
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	color = np.random.randint(0,255,(max_corners,3))

if SIFT:
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(old_gray, None)
	p0 = np.zeros((len(kp),1,2), dtype = np.float32)
	for i in range(len(kp)):
		p0[i,0,0] , p0[i,0,1] = kp[i].pt
	color = np.random.randint(0,255,(len(kp),3))

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# array che conterranno tutte le variazioni di angolo e norma scostamento
theta_tot = []
norm_tot = []

counter = 1
while(counter <= n_frames ):
	ret,frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	# Calculate optical flow of the frame
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	diff = good_new - good_old # delta_y - delta_x
	#print np.round(np.mean(diff, axis = 0)[1])

	theta = np.zeros(len(diff))
	norm = np.zeros(len(diff))

	for i in range(len(diff)):
		theta[i] = math.degrees(math.atan(diff[i,1] / diff[i,0]))
		norm[i] = math.sqrt( math.pow(diff[i,1] , 2) + math.pow(diff[i,0] , 2) )

	# scrivo il valor medio della variazione di angolo e di spostamento per un dato frame
	theta_tot.append(np.mean(theta))
	norm_tot.append(np.mean(norm))

	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		cv2.line(frame, (a,b),(c,d), color[i].tolist(), 2)
		cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

	img = cv2.add(frame,mask)

	cv2.imshow('Optical Flow',img)

	k = cv2.waitKey(30) & 0xff
	if k == 30:
		break

	old_gray = frame_gray.copy()

	if len(p0) < 5:  # mi servono altri punti da tracciare
		if HARRIS:
			p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
		elif SIFT:
			kp = sift.detect(frame_gray, None)
			p0 = np.zeros((len(kp),1,2), dtype = np.float32)
			for i in range(len(kp)):
				p0[i,0,0] , p0[i,0,1] = kp[i].pt
	else:
		p0 = good_new.reshape(-1,1,2)

	counter += 1

cv2.destroyAllWindows()
cap.release()

# scrivo in un file .txt
if SIFT:
	out_file = open('theta_SIFT.txt','w')
	simplejson.dump(theta_tot, out_file)
	out_file.close()

	out_file = open('norm_SIFT.txt','w')
	simplejson.dump(norm_tot, out_file)
	out_file.close()

if HARRIS:
	out_file = open('theta_HARRIS.txt','w')
	simplejson.dump(theta_tot, out_file)
	out_file.close()

	out_file = open('norm_HARRIS.txt','w')
	simplejson.dump(norm_tot, out_file)
	out_file.close()

