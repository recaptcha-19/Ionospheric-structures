import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc("text", usetex = True)
R_earth = 6371
h_ion = 350
d = 10
#sample comment

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
	
def map4(el):
	a0 = 64.4297
	a1 = 0.0942437
	a2 = 1.39436
	a3 = 19.6357
	b0 = 64.3659
	b1 = 0.104974
	b2 = 1.41152
	b3 = -0.0463341
	z = 90 - el
	z_rad = np.deg2rad(z)
	za = (z - a0)*a1
	za = np.deg2rad(za)
	zb = (z - b0)*b1
	zb = np.deg2rad(zb)
	a = (-np.arctan(za) - a2)*a3
	b = (-np.arctan(zb) - b2)*b3
	p = a + b*d
	map_function = 1/((1 - p/100)*np.cos(z_rad))
	return map_function
