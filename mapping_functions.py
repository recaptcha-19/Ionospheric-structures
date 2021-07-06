import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex = True)
R_earth = 6371
h_ion = 350
d = 10

def map3(el):
	f = R_earth/(R_earth + h_ion)
	el = np.deg2rad(el)
	map_function = 1/np.sqrt(1 - (f*np.cos(el))**2)
	return map_function

def map2(el):
	R = R_earth + h_ion
	p = 90 - el
	z = np.deg2rad(p)
	map_function = 1/np.cos(z) + (np.cos(z)**2 - 1)*d**2/(8*R**2*np.cos(z)**5)
	return map_function

def map1(el):
	el = np.deg2rad(el)
	map_function = 1/np.sin(el)
	return map_function
