import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from datetime import timedelta
from day import day
#from VTEC import clean

rc("text", usetex = True)

def TOW2UT(TOW):
	TOW = TOW%86400
	x = str(timedelta(seconds = TOW))
	return x
	
def clean(df, GPS = False, elevation = False, TEC = False, VTEC = False, locktime = False):
	if elevation == True:
		df = df[df['elevation'] > 30]
	if TEC == True:
		df = df[df['TEC'] > 0]
	if VTEC == True:
		df = df[df['VTEC'] > 0]
	if locktime == True:
		df = df[df['locktime'] > 180]
	if GPS == True:
		df = df[(df['SVID'] >= 1) & (df['SVID'] <= 37)]
	return df
	
all_dfs = day(glob.glob("PUNE323?.17_.ismr"))

for day in all_dfs:
	all_dfs[day] = clean(all_dfs[day], elevation = True, TEC = True, locktime = True, GPS = True)
	print(day)
	year = day[4:]
	d, y = day.split("_")
	df_day = all_dfs[day]
	TOW = np.unique(df_day['TOW'])
	SVID = np.unique(df_day['SVID'])

	#df_occurrences =  df_day['SVID'].value_counts(dropna = False)
	#plt.bar(df_occurrences.index, df_occurrences)
	min_occurrences = np.array([])
	for time in TOW:
		df_time = df_day.loc[df_day['TOW'] == time]
		min_occurrence = len(df_time.index)
		min_occurrences = np.append(min_occurrences, min_occurrence)
		#print(time)
		#print(df_time)
	print(np.shape(TOW))
	print(np.shape(min_occurrences))
	plt.plot(TOW, min_occurrences)
	plt.xlabel("SVID")
	plt.ylabel("Number")
	plt.title("No of occurrences of each satellite on day {}, 20{}".format(d, y))
	plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/sat_occurrences_{}.png".format(day, day))
	plt.close()
	'''
	min_count = []
	#el_min_count = []
	for time in TOW:
		df_temp = df_day.loc[df_day['TOW'] == time, 'SVID']
		id = np.asarray(df_temp)
		id = np.unique(id)
		min_count.append(len(id))
	min_count = np.asarray(min_count)
	#el_min_count = np.asarray(el_min_count)

	h = (np.max(TOW) - np.min(TOW))/5
	t0 = np.min(TOW)
	t1 = np.min(TOW) + h
	t2 = np.min(TOW) + 2*h
	t3 = np.min(TOW) + 3*h
	t4 = np.min(TOW) + 4*h
	t5 = np.max(TOW)

	plt.plot(TOW, min_count)
	plt.xlabel("Time (UT)")
	plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
	plt.ylabel("Number")
	plt.title("No of satellites seen on each minute on day {}, 20{}".format(d, y))
	plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/GPS_sat_min_count_{}.png".format(day, day))
	plt.close()
	print("Saved")
	'''
	

	
	
	
	
	
       
	
