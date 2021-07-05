import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import iri2016 as ion
import pandas as pd
import datetime
rc('text', usetex=True)

#a few plots using iri2016
lat = 19.0919   #latitude of GMRT
long = 74.0506  #longitude of GMRT
lat1 = 45.5017  #latitude of Montreal
long1 = -73.5673    #longitude of Montreal
now = datetime.datetime.now()
date = str(now.year) + '-' + str(now.month) + '-' + str(now.day) + 'T{}:{}:{}'.format(now.hour, now.minute, now.second)
#print(date)

#sample plot of TEC variation on a particular day above GMRT
time_profile = ion.timeprofile(('2019-03-25T00:00:00','2019-03-25T23:59:59'), datetime.timedelta(minutes = 30), [0,2000,10], lat, long)
#print(time_profile)
plt.plot(time_profile.time, time_profile.TEC)
plt.xlabel("Time of the day")
plt.ylabel("Total Electron Content")
plt.title("Variation of TEC above GMRT on 25th March, 2019")
plt.savefig("IRI2016_TEC_2019_03_25.png")
plt.show()

#TEC variation on same day from 6.00 am to 7.00 am above GMRT
time_profile_67 = ion.timeprofile(('2019-03-25T06:00:00','2019-03-25T07:00:00'), datetime.timedelta(minutes = 2), [0,2000,10], lat, long)
#print(time_profile)
plt.plot(time_profile_67.time, time_profile_67.TEC)
plt.xlabel("Time of the day (6 am to 7 am)")
plt.ylabel("Total Electron Content")
plt.title("Variation of TEC above GMRT on 25th March, 2019 from 6 am to 7 am")
plt.savefig("IRI2016_TEC_2019_03_25_6am_7am.png")
plt.show()

#altitude vs temperature at 12 noon above GMRT
alt_profile_12 = ion.IRI('2019-03-25T12', [0,2000,10], lat, long)
#print(alt_profile)
plt.plot(alt_profile_12.Te, alt_profile_12.alt_km)
plt.xlabel("Temperature (K)")
plt.ylabel("Altitude (km)")
plt.yscale('log')
plt.title("Variation of temperature with altitude above GMRT at 12 pm on 25th March, 2019")
plt.savefig("IRI2016_Te_2019_03_25_12pm.png")
plt.show()
#altitude vs electron density at 12 noon and 8 pm above GMRT
alt_profile_20 = ion.IRI('2019-03-25T20', [0,2000,10], lat, long)
plt.plot(alt_profile_12.ne, alt_profile_12.alt_km, label = '12 noon (Day)')
plt.plot(alt_profile_20.ne, alt_profile_20.alt_km, label = '8 pm (Night)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Electron density $(cm^{-3})$")
plt.ylabel("Altitude (km)")
plt.title("Variation of electron density with altitude above GMRT on 25th March, 2019")
plt.legend()
plt.savefig("IRI2016_ne_2019_03_25_12pm_8pm.png")
plt.show()

#empirical observation of TEC above GMRT from 6 am to 7 am on 2019-03-25

#0 - GPS week no
#1 - GPS time of week in seconds
#2 - Satellite Vehicle ID (SVID)
#4 - Azimuth (deg)
#5 - Elevation (deg)
#22 - TEC at TOW
#41 - Lock time
R_earth = 6371
h_ion = 350
sat_data = np.loadtxt("GMRT084G.19_.ismr", delimiter = ',', usecols = [0,1,2,4,5,23,41])
#print(sat_data)
titles = ['week', 'TOW', 'SVID', 'azimuth', 'elevation', 'TEC', 'locktime']
df = pd.DataFrame(sat_data, columns = titles)
df.to_csv("GMRT084G.19_.csv", header = True, sep = ',')
#print(df)

def TOW2UT(TOW):

    days = TOW/86400
    days = days.astype(int)
    h = TOW - days
    hours = h/3600
    hours = hours.astype(int)
    m = h - hours
    min = m/60
    min = min.astype(int)
    return [days, hours, min]

#simple cosine mapping function
def map1(el):
    f = R_earth/(R_earth + h_ion)
    map_function = []
    for theta in el:
        a = f*np.cos(np.deg2rad(theta))
        map_function.append(np.cos(np.arcsin(a)))
    map_function = np.asarray(map_function)
    return map_function

#mapping function from Smith et al
#def map2(el):
    
'''
x = df.loc[df['TOW'] == 108060, 'TEC']
index = [i for i in range(len(x)) if x[i]>0]
print(index)
print(x[index])
'''
STEC_final = []
elevations = []
time_indices = np.unique(df['TOW'])
for time in time_indices:
    x = df.loc[df['TOW'] == time, 'TEC']
    x = x[x>0]
    STEC_final = np.append(STEC_final, np.mean(x))
    e = df.loc[df['TOW'] == time, 'elevation']
    elevations = np.append(elevations, e.iloc[0])
STEC_final *= 10**16
UT = TOW2UT(time_indices)
#print(UT)

VTEC_final = STEC_final/map1(elevations)
plt.scatter(time_indices, VTEC_final)
plt.show()