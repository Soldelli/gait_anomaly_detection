import pandas as pd
import os 
import re

files = filter(os.path.isfile, os.listdir( os.curdir ) )

types = ['accellerometer','gravity','gyroscope','linearaccellerometer','magnetometer','rotvec']

template = files[0]
if template == '.DS_Store':
	template = files[1]
	
template =  re.sub(template.split("_")[0], '', template)
template = re.sub('.txt', '', template) 
template = re.sub('.csv', '', template)

for i in types:
	txt_file = pd.read_csv( i + template + ".txt", sep = '\t', header = 0)
	frame_timestamp_txt= pd.read_csv( "frame_timestamp.txt", sep = '\t', header = 0)

	txt_file.to_csv(i + template + ".csv", index = False)
	frame_timestamp_txt.to_csv("frame_timestamp.csv", sep = '\t', index = False)

