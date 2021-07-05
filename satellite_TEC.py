import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from day import day
from datetime import timedelta

rc("text", usetex = True)
R_earth = 6371
h_ion = 350

def TOW2UT(TOW):
	TOW = TOW%86400
	x = str(timedelta(seconds = TOW))
	return x

all_dfs = day(glob.glob("*.ismr"))
#print(len(years))
#print(all_year_days)
#print(all_dfs["272_17"])

for day in all_dfs:
	print(day)
	year = day[4:]
	df_day = all_dfs[day]
	TOW = np.unique(df_day['TOW'])
	SVID = np.unique(df_day['SVID'])
	
	l = 0
	n_SVID = []
	for satellite in SVID:
		df_sat = df_day.loc[df_day['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
		df_p = df_sat[df_sat['TEC']>0]
		ds1 = set([tuple(line) for line in df_sat.values])
		ds2 = set([tuple(line) for line in df_p.values])
		df_n = pd.DataFrame(list(ds1.difference(ds2)))
		n_SVID.append(len(df_n))
		
		minlock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
		maxlock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
		plt.scatter(minlock['TOW'], minlock['TEC'], marker = 's', c = minlock['elevation'], label = "Locktime $\leq$ 3 min" if l == 0 else "")
		plt.scatter(maxlock['TOW'], maxlock['TEC'], marker = 'o', s = 3, c = maxlock['elevation'], label = "Locktime $\geq$ 3 min" if l == 0 else "") 
		l += 1    
		
	
	h = (np.max(TOW) - np.min(TOW))/5
	t0 = np.min(TOW)
	t1 = np.min(TOW) + h
	t2 = np.min(TOW) + 2*h
	t3 = np.min(TOW) + 3*h
	t4 = np.min(TOW) + 4*h
	t5 = np.max(TOW)

	x = plt.clim(0,90)
	plt.colorbar()
	plt.xlabel("Time (UT)")
	plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
	plt.ylabel("Slant TEC (TECU)")
	d, y = day.split("_")
	plt.title("Slant TEC ({}-{})".format(d, y))
	plt.legend()
	#plt.show()
	plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/STEC_{}.png".format(day, day))
	print("Saved")
	plt.close()
	
	d, y = day.split("_")
	n_SVID = np.asarray(n_SVID)
	plt.bar(SVID, n_SVID)
	plt.xlabel("SVID")
	plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
	plt.ylabel("Number")
	plt.title("Number of error values for each satellite on day {}, 20{}".format(d, y))
	plt.savefig("/Data/rpriyadarshan/ismr/sat_TEC_plots/{}/sat_error_nos_{}.png".format(day, day))
	print("Saved")
	plt.close()

		
