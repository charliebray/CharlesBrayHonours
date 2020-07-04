# importing our class from classes.py
from classes import *

# the .fits file that contains our absorption spectrum
filename = "0527-6549.1419.fits"
# pixel position (index is relative to python i.e. first pixel has index of zero)
x_pixel = 150
y_pixel = 150

# create a absorption spectrum class
abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
# creates a spectrum from .fits file with attributes self.vrad (radial velocity array) and self.T_B (spectral flux array) - see classes.py
abs_obj.raw_spectra()
# determine the background temperature along the velocity range (-50, -20) km/s
abs_obj.T_bg(-50, -20)
# create our optical depth spectrum (both smoothed and raw) - see classes.py
abs_obj.optical_depth()

# here we plot both the raw and smoothed spectra
plt.xlim(-100, 75)
plt.plot(abs_obj.vrad, abs_obj.tau_raw, label="Raw")
plt.plot(abs_obj.vrad, abs_obj.tau, label="Smoothed")
plt.xlabel(r"Radial Velocity, $v$ (km/s)")
plt.ylabel(r"Optical Depth, $\tau$")
plt.grid()
plt.show()
