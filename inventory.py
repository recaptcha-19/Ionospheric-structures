import re
import os
import numpy as np
import glob
import string

files_17 = []
files_18 = []
files_19 = []
days = np.linspace(1, 365, 365)
pattern = r'[0-9]+'

for file in glob.glob("*.ismr"):
    day, year = re.findall(pattern, file)
    if year == '17':	
        files_17.append(file)
    if year == '18':	
        files_18.append(file)
    if year == '19':
        files_19.append(file)
'''
print(len(files_17))
print(len(files_18))
print(len(files_19))
'''
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWX"

days_17 = []
for file in files_17:
    day, _ = re.findall(pattern, file)
    days_17.append(day)
days_17 = set(days_17)
days_17 = list(days_17)
days_17.sort()
print(days_17)
for day in days_17:
    temp = []
    x = glob.glob("PUNE{}?.17_.ismr".format(day))
    for file in x:
        temp.append(file[7])
    absent_hours = set(temp).symmetric_difference(alphabet)
    absent_hours = list(absent_hours)
    absent_hours.sort()
    if absent_hours == []:
        continue
    absent_hours = ''.join(absent_hours)
    print("2017, Day {}:".format(day) + absent_hours + '\n')

days_18 = []
for file in files_18:
    day, _ = re.findall(pattern, file)
    days_18.append(day)
days_18 = set(days_18)
days_18 = list(days_18)
days_18.sort()
for day in days_18:
    temp = []
    x = glob.glob("PUNE{}?.18_.ismr".format(day))
    for file in x:
        temp.append(file[7])
    absent_hours = set(temp).symmetric_difference(alphabet)
    absent_hours = list(absent_hours)
    absent_hours.sort()
    if absent_hours == []:
        continue
    absent_hours = ''.join(absent_hours)
    print("2018, Day {}:".format(day) + absent_hours + '\n')

days_19 = []
for file in files_19:
    day, _ = re.findall(pattern, file)
    days_19.append(day)
days_19 = set(days_19)
days_19 = list(days_19)
days_19.sort()
for day in days_19:
    temp = []
    x = glob.glob("PUNE{}?.19_.ismr".format(day))
    for file in x:
        temp.append(file[7])
    absent_hours = set(temp).symmetric_difference(alphabet)
    absent_hours = list(absent_hours)
    absent_hours.sort()
    if absent_hours == []:
        continue
    absent_hours = ''.join(absent_hours)
    print("2019, Day {}:".format(day) + absent_hours + '\n')
        

