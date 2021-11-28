from VTEC import VTEC_averaged, clean
from day import day
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from pickle import load
import time

#2018
all_dfs = day(glob.glob("*.18_.ismr"))
#all_dfs_18 = day(glob.glob("*.18_.ismr"))
#all_dfs = dict(all_dfs_17)
#all_dfs.update(all_dfs_18)
for day in all_dfs:
	all_dfs[day] = clean(all_dfs[day], GPS = True, elevation = True, TEC = True, locktime = True)
print("Cleaned")
VTEC_dict = VTEC_averaged(all_dfs, "map3")
#print(VTEC_dict['103_18'])

main_diff_array = []
main_VTEC_array = []
main_IRI_array = []
#diff_VTEC_dict = {}
for day in VTEC_dict:
	print(day)
	day_VTEC_dict = VTEC_dict[day]
	median_VTEC = day_VTEC_dict['{}_median_VTEC'.format(day)]
	print(len(median_VTEC))
	TOW = day_VTEC_dict['{}_TOW'.format(day)]
	TOW = TOW%86400
	#median_VTEC = median_VTEC[2:-2]
	iri_time_part = day_VTEC_dict['{}_IRI_time'.format(day)]
	iri_TEC_part = day_VTEC_dict['{}_IRI_TEC'.format(day)]
	#print(len(median_VTEC))
	print(iri_time_part)
	print(iri_TEC_part)
	if len(iri_time_part) > 1:
		f = interp1d(iri_time_part, iri_TEC_part)
		iri_time = np.arange(iri_time_part[0], iri_time_part[-1] + 60, 60)
		iri_TEC = f(iri_time)
		#print(iri_time[0] == iri_time_part[0])
		#print(iri_TEC[0] == iri_TEC_part[0])
		mask = np.isin(TOW[2:-2], iri_time)
		indices = np.where(mask == True)
		print(indices)
		#print(iri_time)
		#print(TOW)
		VTEC = np.empty(len(iri_time))
		VTEC[:] = np.nan
		print(median_VTEC[indices])
		VTEC[indices] = median_VTEC[indices]
		#VTEC[mask] = median_VTEC[mask]
		#print(VTEC[indices])
		diff = iri_TEC - VTEC
		print(diff)
		main_diff_array.append(diff)
		main_VTEC_array.append(VTEC)
		main_IRI_array.append(iri_TEC)
	else:
		iri_time = np.arange(iri_time_part[0], iri_time_part[-1] + 60, 60)
		print(len(iri_time))
		VTEC = np.empty(len(iri_time))
		VTEC[:] = np.nan
		print(VTEC)
		main_diff_array.append(VTEC)
		main_VTEC_array.append(VTEC)
		
img = plt.imshow(main_diff_array, cmap = 'seismic', aspect = 'auto')
plt.xlabel("Minute of the day")
plt.ylabel("Day of year 2018")
bar = plt.colorbar(img)
bar.set_label("VTEC(IRI) - VTEC(GPS) in TEC units")
plt.savefig('/Data/rpriyadarshan/ismr/yearly_trends_18_seismic.png')
plt.close()

img1 = plt.imshow(main_VTEC_array, aspect = "auto")
plt.xlabel("Minute of the day")
plt.ylabel("Day of year 2018")
bar1 = plt.colorbar(img1)
bar1.set_label("VTEC(GPS) in TEC units")
plt.savefig('/Data/rpriyadarshan/ismr/yearly_trends_VTEC_18_final.png')
plt.close()

img1 = plt.imshow(main_IRI_array, aspect = "auto")
plt.xlabel("Minute of the day")
plt.ylabel("Day of year 2018")
bar2 = plt.colorbar(img1)
bar2.set_label("VTEC(IRI) in TEC units")
plt.savefig('/Data/rpriyadarshan/ismr/yearly_trends_IRI_18_final.png')
plt.close()
#main_diff_array = np.asarray(main_diff_array)
#print(np.shape(main_diff_array))

