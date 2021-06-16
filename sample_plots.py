import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd
import glob
from datetime import timedelta

rc("text", usetex = True)
R_earth = 6371
h_ion = 350

sat_data1 = np.loadtxt("GMRT084G.19_.ismr", delimiter = ',', usecols = [0,1,2,4,5,22,41])
sat_data2 = np.loadtxt("GMRT084H.19_.ismr", delimiter = ',', usecols = [0,1,2,4,5,23,41])
sat_data3 = np.loadtxt("GMRT084I.19_.ismr", delimiter = ',', usecols = [0,1,2,4,5,23,41])
titles = ['week', 'TOW', 'SVID', 'azimuth', 'elevation', 'TEC', 'locktime']
df1 = pd.DataFrame(sat_data1, columns = titles)
df2 = pd.DataFrame(sat_data2, columns = titles)
df3 = pd.DataFrame(sat_data3, columns = titles)
dfs = [df1, df2, df3]
df = pd.concat(dfs)

def TOW2UT(TOW):

    days = TOW/86400
    days = int(days)
    h = TOW - days*86400
    hours = h/3600
    hours = int(hours)
    m = h - hours*3600
    min = m/60
    min = int(min)
    s = m - min*60
    sec = int(s)
    return "{}:{}:{}".format(hours, min, sec)

def map1(el):
    f = R_earth/(R_earth + h_ion)
    map_function = []
    for theta in el:
        a = np.cos(90 - theta)
        map_function.append(a)
    map_function = np.asarray(map_function)
    return map_function

SVID = np.unique(df['SVID'])
n_SVID = []

df_sat = df.loc[df['SVID'] == SVID[0], ['TOW', 'elevation', 'TEC', 'locktime']]
df_sat = df_sat[df_sat['TEC']>0]
t = []
'''
for TOW in df_sat['TOW']:
    t.append(TOW2UT(TOW))
plt.scatter(df_sat['TOW'], df_sat['TEC']*10**16)
plt.show()
'''

for satellite in SVID:
    
    df_sat = df.loc[df['SVID'] == satellite, ['TOW', 'elevation', 'TEC', 'locktime']]
    df_p = df_sat[df_sat['TEC']>0]
    df_n = df_sat[~df_sat.index.isin(df_p)]
    n_SVID.append(len(df_n))
    less_lock = df_p.loc[df_p['locktime']<180, ['TOW', 'TEC', 'elevation']]
    more_lock = df_p.loc[df_p['locktime']>180, ['TOW', 'TEC', 'elevation']]
    plt.scatter(less_lock['TOW'], less_lock['TEC']*10**16, marker = 's', c = less_lock['elevation'])
    plt.scatter(more_lock['TOW'], more_lock['TEC']*10**16, marker = 'o', s = 5, c = more_lock['elevation'])
h = (np.max(df['TOW']) - np.min(df['TOW']))/5
t0 = np.min(df['TOW'])
t1 = np.min(df['TOW']) + h
t2 = np.min(df['TOW']) + 2*h
t3 = np.min(df['TOW']) + 3*h
t4 = np.min(df['TOW']) + 4*h
t5 = np.max(df['TOW'])
plt.colorbar()
plt.clim(0,90)
plt.xlabel("Time")
plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
plt.ylabel("Slant TEC")
plt.title("TEC on 25th March, 2019")
plt.savefig("STEC.png")
plt.show()

plt.bar(SVID, df['SVID'].value_counts(dropna = False))
plt.xlabel("SVID")
plt.ylabel("No of occurrences")
plt.title("No of occurrences of each satellite in a day")
plt.savefig("satellite_occurrences_day.png")
plt.show()

min = np.unique(df['TOW'])

min_count = []
el_min_count = []
for time in min:
    #elevation < 30 deg
    df_temp = df.loc[df['TOW'] == time, 'SVID']
    id = np.asarray(df_temp)
    id = np.unique(id)
    min_count.append(len(id))

    #elevation > 30 deg
    el_df_temp = df.loc[(df['TOW'] == time) & (df['elevation']>30), 'SVID']
    el_id = np.asarray(el_df_temp)
    el_id = np.unique(el_id)
    el_min_count.append(len(el_id))
    
min_count = np.asarray(min_count)
plt.plot(min, min_count)
plt.xlabel("Time of the day")
plt.ylabel("No of satellites")
plt.title("No of satellites seen each minute in a day")
plt.savefig("satellite_occurrences_min.png")
plt.show()

el_min_count = np.asarray(el_min_count)
plt.plot(min, el_min_count)
plt.xlabel("Time of the day")
plt.ylabel("No of satellites")
plt.title("No of satellites seen each minute in a day with elevation more than 30$^\circ$")
plt.savefig("satellite_occurrences_min_30.png")
plt.show()

n_SVID = np.asarray(n_SVID)
plt.bar(SVID, n_SVID)
plt.xlabel("Satellite id")
plt.ylabel("Number of negative/NaN occurrences")
plt.title("Number of error values in a day for each satellite")
plt.savefig("sat_error_nos.png")
plt.show()

VTEC = []
for time in min:
    df_temp = df.loc[(df['TOW'] == time) & (df['elevation']>30), ['TEC', 'elevation']]
    VTEC_min = np.mean(df_temp['TEC']/map1(df_temp['elevation']))
    VTEC.append(VTEC_min)

VTEC = np.asarray(VTEC)
plt.plot(min, VTEC)
plt.xlabel("Time of the day")
plt.ylabel("Total Electron Content (TECU)")
plt.title("Estimated VTEC variation as a function of time")
plt.xticks([t0, t1, t2, t3, t4, t5], [TOW2UT(t0), TOW2UT(t1), TOW2UT(t2), TOW2UT(t3), TOW2UT(t4), TOW2UT(t5)])
plt.savefig("VTEC_estimate.png")
plt.show()

