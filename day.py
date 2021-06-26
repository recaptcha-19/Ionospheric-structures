import numpy as np
import pandas as pd
import glob
import re
import itertools
import os

pattern1 = r'[0-9]+'
pattern = r'[0-9]{3}'

def day(files):
	titles = ['week', 'TOW', 'SVID', 'azimuth', 'elevation', 'TEC', 'locktime']
	years = []	#list of years in the files
	all_dfs = {}	#dictionary with all dataframes 
	all_year_days = {}	#nested list with all days in each year
	
	for file in files:
		x = re.findall(pattern1, file)
		#print(x[1])
		years.append(x[1])

	years = list(set(years))
	years.sort()
	year_files = {}	#list of all files in a year
	for i in range(len(years)):
		year_file = [file for file in files if file[9:11] == years[i]]
		year_files[years[i]] = year_file
	
	for i in year_files:
		year_days = []	#list of all days in a year
		for file in year_files[i]:
			x = re.findall(pattern, file)
			year_days.append(x)
		year_days = np.unique(np.asarray(year_days))
		#year_days.sort()
		all_year_days[i] = year_days

	#print(all_year_days)
		
	for year in all_year_days:
		days = all_year_days[year]
		for day in days:
			day_files = [file for file in year_files[year] if day in file]	#list of all files corresponding to a particular day
			list_day = []	#list of dataframes in a day
			for x in day_files:
				sat_data = np.loadtxt(x, delimiter = ',', usecols = [0,1,2,4,5,22,41])
				df = pd.DataFrame(sat_data, columns = titles)
				list_day.append(df)	
			#print(len(list_day))
			df_day = pd.concat(list_day)	#single dataframe for an entire day
			all_dfs["{}_{}".format(day, year)] = df_day	#master dataframe for the entire dataset	

	return all_dfs
	


