from classes import *

filename_gass = "gass_smallest.fits"

filename = "0527-6549.1419.fits"
x_pixel = 150
y_pixel = 150

abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
#abs_obj.raw_spectra()
#abs_obj.T_bg(-50, -20)
#abs_obj.optical_depth()

em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
em_obj.raw_spectra()

plt.xlim(-100, 75)
plt.plot(em_obj.vrad, em_obj.T_B)
plt.xlabel(r"Radial Velocity, $v$ (km/s)") #, fontsize=12, weight='medium', stretch='normal', family='Times New Roman', style='normal', variant='normal')
plt.ylabel(r"Brightness Temperature, $T_{B}$") #, fontsize=12, weight='medium', stretch='normal', style='normal', variant='normal')
plt.grid()
#plt.xlim(-50,50)
#plt.savefig("example_em_intro.pdf")
plt.show()
