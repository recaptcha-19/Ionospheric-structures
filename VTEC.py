import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from day import day
import datetime
import iri2016 as ion
import mapping_functions as mapf
from itertools import combinations

rc("text", usetex = True)
R_earth = 6371
h_ion = 350
d = 100
lat = 19.0919   #latitude of GMRT
long = 74.0506  #longitude of GMRT

def TOW2UT(TOW):
	TOW = TOW%86400
	x = str(datetime.timedelta(seconds = TOW))
	return x

def rms(array):
	return np.sqrt(np.mean(array**2))

def VTEC(STEC, elevation, map_func):
	map_method = getattr(mapf, map_func)
	VTEC = STEC/map_method(elevation)
	return VTEC

def clean(df, elevation = False, TEC = False, VTEC = False, locktime = False):
	if elevation == True:
		df = df[df['elevation'] > 30]
	if TEC == True:
		df = df[df['TEC'] > 0]
	if VTEC == True:
		df = df[df['VTEC'] > 0]
	if locktime == True:
		df = df[df['locktime'] > 180]
	return df	

def VTEC(STEC, elevation, map_func):
	map_method = getattr(mapf, map_func)
	VTEC = STEC/map_method(elevation)
	return VTEC

def VTEC_time(all_dfs, map_fn):
	
	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])
		
		l = 0
		for satellite in SVID:
			df_sat = df_day.loc[df_day['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
			#df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			df_p = clean(df_sat, elevation = True, TEC = True)
			
			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation', 'locktime']]
			minlock['VTEC'] = VTEC(minlock['TEC'], minlock['elevation'], map_func = map_fn)
			minlock_VTEC = clean(minlock, VTEC = True)
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation', 'locktime']]
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = clean(maxlock, VTEC = True)
			
			plt.figure(0)
			plt.scatter(minlock_VTEC['TOW'], minlock_VTEC['VTEC'], marker = 's', c = minlock_VTEC['elevation'], label = "Locktime $\leq$ 3 min" if l == 0 else "")
			plt.scatter(maxlock_VTEC['TOW'], maxlock_VTEC['VTEC'], marker = 'o', s = 3, c = maxlock_VTEC['elevation'], label = "Locktime $>$ 3 min" if l == 0 else "")
			l += 1
			
		h = (np.max(TOW) - np.min(TOW))/5
		t0 = np.min(TOW)
		t1 = np.min(TOW) + h
		t2 = np.min(TOW) + 2*h
		t3 = np.min(TOW) + 3*h
		t4 = np.min(TOW) + 4*h
		t5 = np.max(TOW)
		
		plt.figure(0)
		x = plt.clim(30,90)
		plt.colorbar()
		plt.xlabel("Time (UT)")
		plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("Vertical TEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_VTEC_{}.png".format(day, map_fn, day))
		print("Saved")
		plt.close()
		
	
def VTEC_STEC(all_dfs, map_fn):

	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])
		
		l = 0
		for satellite in SVID:
			df_sat = df_day.loc[df_day['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
			#df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			df_p = clean(df_sat, elevation = True, TEC = True)

			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
			minlock['VTEC'] = VTEC(minlock['TEC'], minlock['elevation'], map_func = map_fn)
			minlock_VTEC = clean(minlock, VTEC = True)
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = clean(maxlock, VTEC = True)

			plt.figure(1)
			plt.scatter(minlock_VTEC['TEC'], minlock_VTEC['VTEC'], marker = 's', c = minlock_VTEC['elevation'], label = "Locktime $\leq$ 3 min" if l == 0 else "")
			plt.scatter(maxlock_VTEC['TEC'], maxlock_VTEC['VTEC'], marker = 'o', s = 3, c = maxlock_VTEC['elevation'], label = "Locktime $>$ 3 min" if l == 0 else "") 
			l += 1
	
		plt.figure(1)
		x = plt.clim(30,90)
		plt.colorbar()
		plt.xlabel("Slant TEC (TECU)")
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("VTEC vs STEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_VTEC_STEC_{}.png".format(day, map_fn, day))
		print("Saved")
		plt.close()
	
	
def VTEC_averaged(all_dfs, map_fn, iri = False):	
	
	for day in all_dfs:
		print(day)
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])
		el = np.unique(df_day['elevation'])

		mean_VTEC = np.array([])
		median_VTEC = np.array([])
		RMS_VTEC = np.array([])
		for time in TOW:
			
			df_sat = df_day.loc[df_day['TOW'] == time, ['TOW', 'elevation', 'TEC', 'locktime']]
			df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']]  
			#maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock = clean(df_sat, elevation = True, TEC = True, locktime = True)
			#maxlock['VTEC'] = maxlock['TEC']/map_method(maxlock['elevation'])
			maxlock['VTEC'] = VTEC(maxlock['TEC'], maxlock['elevation'], map_func = map_fn)
			#maxlock_VTEC = maxlock.loc[maxlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			maxlock_VTEC = clean(maxlock, VTEC = True)

			m_VTEC = np.mean(maxlock_VTEC['VTEC'])
			mean_VTEC = np.append(mean_VTEC, m_VTEC)
			med_VTEC = np.median(maxlock_VTEC['VTEC'])
			median_VTEC = np.append(median_VTEC, med_VTEC)
			R_VTEC = rms(maxlock_VTEC['VTEC'])
			RMS_VTEC = np.append(RMS_VTEC, R_VTEC)
			
		plt.plot(TOW, mean_VTEC, c = 'blue', label = "$VTEC_{mean}$")
		plt.plot(TOW, median_VTEC, c = 'red', label = "$VTEC_{median}$")
		plt.plot(TOW, RMS_VTEC, c = 'black', label = "$VTEC_{RMS}$")

		if iri == True:
			dayOfYear, Year = day.split("_")
			Year = '20' + year
			d = datetime.datetime.strptime('{} {}'.format(dayOfYear, Year),'%j %Y')
			calendar_day = d.strftime('%Y-%m-%d')
			start_time = TOW2UT(np.min(TOW))
			end_time = TOW2UT(np.max(TOW))
			start = calendar_day + "T{}".format(start_time)
			end = calendar_day + "T{}".format(end_time)
			time_profile = ion.timeprofile((start, end), datetime.timedelta(minutes = 30), [0, 2000, 10], lat, long)
			plt.plot(time_profile.time, time_profile.TEC, '--', c = 'g', label = 'IRI')

		h = (np.max(TOW) - np.min(TOW))/5
		t0 = np.min(TOW)
		t1 = np.min(TOW) + h
		t2 = np.min(TOW) + 2*h
		t3 = np.min(TOW) + 3*h
		t4 = np.min(TOW) + 4*h
		t5 = np.max(TOW)
		
		plt.xlabel("Time (UT)")
		plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
		plt.ylabel("Vertical TEC (TECU)")
		d, y = day.split("_")
		plt.title("Vertical TEC ({}-{})".format(d, y))
		plt.legend()
		plt.grid()
		plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_VTEC_averaged_{}.png".format(day, map_fn, day))
		print("Saved")
		plt.close()


def VTEC_comparison(all_dfs, map_fn):

	for day in all_dfs:
		year = day[4:]
		df_day = all_dfs[day]
		
		SVID = np.unique(df_day['SVID'])

		#df = df_day.loc[(df_day['elevation'] > 30) & (df_day['elevation'] < 35) & (df_day['locktime'] > 180) & (df_day['TOW'] == 600), ['SVID', 'TOW', 'TEC', 'elevation']]
		#df = df_day.loc[(df_day['locktime'] > 180) & (df_day['elevation'] > 30) & (df_day['TEC'] > 0), ['SVID', 'TOW', 'TEC', 'elevation']]
		df = clean(df_day, elevation = True, TEC = True, locktime = True)
		el = np.unique(df['elevation'])
		el = el[~np.isnan(el)]
		df['VTEC'] = VTEC(df['TEC'], df['elevation'], map_func = map_fn)
		bin_angles = np.arange(np.min(el), np.max(el) + 2, 2)
		bin_labels = ['{}-{}'.format(bin_angles[i], bin_angles[i+1]) for i in range(len(bin_angles)-1)]
		df['elevation_bins'] = pd.cut(df['elevation'], bins = bin_angles, labels = bin_labels)
		
		delta_VTEC_info = {}
		for bin in bin_labels:
			#print(bin)
			df_bin = df.loc[df['elevation_bins'] == bin, ['SVID', 'TOW', 'TEC', 'VTEC', 'elevation']]
			#print("Bin size: {}".format(df_bin.size))
			TOW = np.asarray(df_bin['TOW'])

			idx_sort = np.argsort(TOW)
			sorted_TOW = TOW[idx_sort]
			vals, idx_start, count = np.unique(sorted_TOW, return_counts = True, return_index = True)
			res = np.split(idx_sort, idx_start[1:])
			vals = vals[count > 1]
			res = filter(lambda x: x.size > 1, res)
			list_indices = list(res)

			mean_els = []
			delta_VTECs = []
			for indices in list_indices:
				comb = list(combinations(indices, 2))
				#print(comb)
				for index in comb:
					a,b = index
					el1 = df_bin['elevation'].iloc[a]
					el2 = df_bin['elevation'].iloc[b]
					mean_el = (el1 + el2)/2
					mean_els.append(mean_el)
					VTEC1 = df_bin['VTEC'].iloc[a]
					VTEC2 = df_bin['VTEC'].iloc[b]
					delta_VTEC = abs(VTEC1 - VTEC2)
					delta_VTECs.append(delta_VTEC)

			delta_VTECs = np.asarray(delta_VTECs)
			mean_els = np.asarray(mean_els)
			delta_VTEC_info['{}_delta_VTECs'.format(bin)] = delta_VTECs
			delta_VTEC_info['{}_mean_els'.format(bin)] = mean_els

		np.save("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/{}_delta_VTEC_info.npy".format(day, day), delta_VTEC_info)
		print("Saved!")
