from classes import *

# the emission .fits file, replace with your own here :)
filename_gass = "gass_smallest.fits"

# the name of our absorption .fits file, and the pixel position of the line of sight (relative to python indexing)
filename = "0527-6549.1419.fits"
x_pixel = 150
y_pixel = 150

# creating the absorption spectrum instance, this has an attribute .ra and .dec that is the right ascension and declination of the line of sight
abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)

# create the emission spectrum instance, using the emission .fits file, and the (RA,DEC) of the line of sight
em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
# create the emission spectrum
em_obj.raw_spectra()

# plotting the emission spectrum
plt.xlim(-100, 75)
plt.plot(em_obj.vrad, em_obj.T_B)
plt.xlabel(r"Radial Velocity, $v$ (km/s)") 
plt.ylabel(r"Brightness Temperature, $T_{B}$")
plt.grid()
#plt.xlim(-50,50)
#plt.savefig("example_em_intro.pdf")
plt.show()
