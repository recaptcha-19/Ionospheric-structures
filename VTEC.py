import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from day import day
import datetime
import iri2016 as ion
import mapping_functions as mapf

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

def map_comparison(el, maps):
	plt.figure()
	for map_fn in maps:
		map_method = getattr(mapf, map_fn)
		plt.plot(el, map_method(el), label = map_fn)
	plt.xlabel("Elevation ($^\circ$)")
	plt.ylabel("Map function")
	plt.title("Map function comparison")
	plt.legend()
	plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/map_fn_comparison.png")
	plt.show()
	plt.close()

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
			df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			
			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
			minlock['VTEC'] = VTEC(STEC = minlock['TEC'], elevation = minlock['elevation'], map_func = map_fn)
			minlock_VTEC = minlock.loc[minlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock['VTEC'] = VTEC(STEC = maxlock['TEC'], elevation = maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = maxlock.loc[maxlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			
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
			df_p = df_sat.loc[(df_sat['TEC']>0) & (df_sat['elevation']>30), ['TOW', 'elevation', 'TEC', 'locktime']] 
			
			minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
			minlock['VTEC'] = VTEC(STEC = minlock['TEC'], elevation = minlock['elevation'], map_func = map_fn)
			minlock_VTEC = minlock.loc[minlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock['VTEC'] = VTEC(STEC = maxlock['TEC'], elevation = maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = maxlock.loc[maxlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]

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
			maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
			maxlock['VTEC'] = VTEC(STEC = maxlock['TEC'], elevation = maxlock['elevation'], map_func = map_fn)
			maxlock_VTEC = maxlock.loc[maxlock['VTEC']>0, ['TEC', 'VTEC', 'TOW', 'elevation']]
			
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
	t = 300

	for day in all_dfs:
		year = day[4:]
		df_day = all_dfs[day]
		TOW = np.unique(df_day['TOW'])
		SVID = np.unique(df_day['SVID'])

		df = df_day.loc[(df_day['elevation'] > 30) & (df_day['locktime'] > 180), ['SVID', 'TOW', 'TEC', 'elevation']]
		df['VTEC'] = VTEC(df['TEC'], df['elevation'], map_func = "map3")
		ind = np.asarray(df.index)

		combinations = []
		for i in range(len(ind)):
			current = ind[i]
			rest = ind[i+1:]
			current_list = current*np.ones(len(rest))
			current_comb = (np.vstack((rest, current_list))).T
			for element in current_comb:
				combinations.append(element)
		combinations = np.asarray(combinations)
		combinations = combinations.astype('int64')

		delta_els = []
		delta_VTECs = []
		for indices in combinations:
			p, q = indices
			el1 = df.loc[p].at['elevation']
			el2 = df.loc[q].at['elevation']
			delta_el = abs(el1 - el2)
			delta_els.append(delta_el)
			VTEC1 = df.loc[p].at['VTEC']
			VTEC2 = df.loc[q].at['VTEC']
			delta_VTEC = abs(VTEC1 - VTEC2)
			delta_VTECs.append(delta_VTEC)

		plt.figure()
		plt.scatter(delta_els, delta_VTECs)
		plt.xlabel("$\Delta \epsilon$")
		plt.ylabel("$\Delta VTEC$")
		plt.grid()
		plt.show()
		plt.close()


all_dfs = day(glob.glob("PUNE323?.17_.ismr"))
VTEC_comparison(all_dfs, map_fn = 'map3')
