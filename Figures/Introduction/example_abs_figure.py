from classes import *

filename = "0527-6549.1419.fits"
x_pixel = 150
y_pixel = 150

abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
abs_obj.raw_spectra()
abs_obj.T_bg(-50, -20)
abs_obj.optical_depth()

plt.xlim(-100, 75)
plt.plot(abs_obj.vrad, abs_obj.tau_raw, label="Raw")
plt.plot(abs_obj.vrad, abs_obj.tau, label="Smoothed")
plt.xlabel(r"Radial Velocity, $v$ (km/s)")
plt.ylabel(r"Optical Depth, $\tau$")
plt.grid()
plt.show()
