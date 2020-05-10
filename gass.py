import astropy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from classes import *
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib.patches import Rectangle

# This is an example of using the classes.py file to create the image in my thesis, with the peak optical depths placed over
#	the GASS column density image.

# This is the name of the fits file from GASS.
fits_dir = "gass_smallest.fits"

# Processing it to remove all points in the spectrum below 3*sigma of the noise.
hdu = fits.open(fits_dir)
hdr = hdu[0].header
data = hdu[0].data
fits.setval(fits_dir, "CUNIT1", value="deg")
fits.setval(fits_dir, "CUNIT2", value="deg")

wcs = WCS(hdr, naxis=2)

noise = np.zeros_like(data[0])

# Noise is the average of the brightness temperature from index 0 to 50, at each pixel (array)
for z_ind in range(0, len(data[:50])):
	noise += np.absolute(data[z_ind,:,:])/len(range(0, len(data[:50])))

# Sets all elements less than 3 times the noise to zero.
data[data < 3.0*noise] = 0

# Column density map: N = 1.823*(10**18) * sum(data) * dv (km/s)
dv = hdr["CDELT3"]/1000
column_density = 1.823*(10**18)*np.sum(data, axis=0)*dv

# These are two lists, one containing all the sources with detections (along with their pixel position) "LMC_list",
#	and one containing sources without detection "LMC_no_abs_list".

LMC_list = [["0452-6823.1419.fits", 150, 150],\
		["0507-6657.1419.fits", 128, 148], ["0520-7312.1419.fits", 150, 149],\
		["0552-6814.1419.fits", 148, 152],\
		["0516-7336.1419.fits", 151, 150], ["0450-6940.1419.fits", 150, 150],\
		["0512-7233.1419.fits", 150, 150],\
		["0527-6549.1419.fits", 150, 150],\
		["0518-6617.1419.fits", 150, 150], ["0506-6555.1419.fits", 150, 150],\
		["0521-6960.1419.fits", 149, 150],\
		["0522-7038.1419.fits", 150, 149],\
		["0547-7206.1419.fits", 150, 151]]

LMC_no_abs_list = [["0514-6707.1419.fits", 149, 150], ["0454-7040.1419.fits", 150, 150],\
		["0548-6745.1419.fits", 149, 150], ["0502-6625.1419.fits", 150, 150],\
		["0507-7144.1419.fits", 148, 149], ["0528-6759.1419.fits", 150, 150],\
		["0525-674.1419.fits", 150, 150], ["0459-6658.1419.fits", 150, 150],\
		["0540-6906.1419.fits", 151, 149]]

# These are the criteria emission spectra, we require SNR>5 and FWHM>5.
em_min = 5.0
em_sig = 5.0

# These are the lists we will use to store all the individual information/properties.
name_list = []
ra_list = []
dec_list = []
tau_list = []
noise_list = []

# We loop through each source in our list that showed detection.
for source in LMC_list:

	# Our initial fitting requirements for absorption features, SNR>5 and FWHM>1.
	abs_sig = 5.0
	abs_min = 1.0

	# The name of the fits file, along with the (x,y) pixel of the source.
	filename, x_pixel, y_pixel = source

	# We create a new instance of the absorption spectrum class
	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	# This method provides us with the radial velocity array, and raw brightness temperature/flux array as attributes
	abs_obj.raw_spectra()
	# This method provides us with the average background temperature in the velocity range (-50, -20) km/s.
	abs_obj.T_bg(-50, -20)
	# This method provides us with the all optical depth quantities/arrays as attributes (for all attributes see "classes.py"
	abs_obj.optical_depth()

	# We create a new instance of Plotting_Spectra, where we insert our radial velocity and raw optical depth arrays
	# 	along with our fitting criteria ("abs_min" = minimum FWHM and "abs_sig" = minimum SNR)
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	# This runs the GaussPy+ fitting process and returns two attributes, one with the parameters for each component,
	#	and the number of components.
	tau_spec.gpp_parameters()

	# The number of fitted components.
	n = tau_spec.components
	# The assigning of this variable is a bit weird, but essentially it's used to remember the number of components
	#	when we use our other fitting criteria for absorption spectra (which are SNR>3 and FWHM>3).
	m = n
	# The amplitudes of each component are stored in this list
	tau_amp = tau_spec.params[:n]
	# Similarly for their FWHM
	tau_fwhm = tau_spec.params[n:2*n]
	# Similarly for their position/mean
	tau_mean = tau_spec.params[2*n:]

	# The following code now repeats the above GaussPy+ decomposition but with the alternative fitting criteria
	#	of SNR>3 and FWHM>3. If a feature satisfies both criteria then the former component is chosen (i.e. SNR>5 and FWHM>1).
	
	# this gives the rough position of each component that satisfied the first criteria.
	tau_check = np.round(tau_mean)

	# new fitting criteria
	abs_sig = 3.0
	abs_min = 3.0
	# performs gausspy+ fitting
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	# number of new components fitted
	n = tau_spec.components

	# for each component, check if we have already fitted a feature before. If not, add its parameters to the lists.
	for i in range(0,n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n + i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1

	# looks through all features that have been fitted and stores the parameters/information of interest.
	for i in range(0, m):
		amp = tau_amp[i]
		mu = tau_mean[i]
		fwhm = tau_fwhm[i]
		if np.abs(mu) > 50.:
			continue
		name_list.append(filename)
		ra_list.append(abs_obj.ra)
		dec_list.append(abs_obj.dec)
		tau_list.append(amp)
		noise_list.append(abs_obj.tau_noise)

# Plotting the figure

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = plt.subplot(projection=wcs)

# Plots the peak optical depths
marker_size = np.array(tau_list)*1000
im1 = ax.scatter(ra_list, dec_list, c=tau_list, transform=ax.get_transform("world"), edgecolors='b', s=marker_size, cmap="autumn", alpha = 0.5)

# Creates the colorbar for optical depth
cax1 = fig.add_axes([0.425, 0.83, 0.27, 0.03])
cbar = plt.colorbar(im1, orientation="horizontal", cax=cax1, ticks=np.arange(0., 3.0, 0.3), extend='max', extendrect=True, label=r"$\tau_{peak}$", alpha=1)
cbar.set_alpha(1)
cbar.draw_all()

# Plots the column density map
column_density = np.array(column_density)/(10.**20)
im2 = ax.imshow(column_density, origin='lower', cmap="Greys")

# Labels etc
plt.grid(color="black", ls="dotted")
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")

# Creates colorbar for column density
cax2 = fig.add_axes([0.72, 0.11, 0.015, 0.77])
plt.colorbar(im2, orientation="vertical", cax=cax2, label=r"$N_{H}(\tau \ll 1)$ ($10^{20}$ cm$^{-2}$)")

# Creates a list of all sources (RA, DEC) that did not show detection
noabs_ra_list = list()
noabs_dec_list = list()
for source in LMC_no_abs_list:
	filename, x_pixel, y_pixel = source
	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	noabs_ra_list.append(abs_obj.ra)
	noabs_dec_list.append(abs_obj.dec)
	print(str(filename) + " : " + str(abs_obj.ra) + " : " + str(abs_obj.dec))

# Adds markers of all sources that didn't show absorption
im3 = ax.scatter(noabs_ra_list, noabs_dec_list, transform=ax.get_transform("world"), marker="x")

plt.savefig("GASS_column_density.pdf")
#plt.show()
