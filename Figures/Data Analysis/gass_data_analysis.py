import astropy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from classes import *
from astropy.wcs import WCS
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from matplotlib.patches import Rectangle

# importing data etc

fits_dir = "gass_smallest.fits"
hdu = fits.open(fits_dir)

hdr = hdu[0].header
data = hdu[0].data

fits.setval(fits_dir, "CUNIT1", value="deg")
fits.setval(fits_dir, "CUNIT2", value="deg")

wcs = WCS(hdr, naxis=2)

noise = np.zeros_like(data[0])

# noise is the average of brightness temperature from index 0 to 50, at each pixel (array)

for z_ind in range(0, len(data[:50])):
	noise += np.absolute(data[z_ind,:,:])/len(range(0, len(data[:50])))

# sets all elements less than 3 times the noise to zero.

data[data < 3.0*noise] = 0

# column density map: N = 1.823*(10**18) * sum(data) * dv (km/s)

dv = hdr["CDELT3"]/1000
column_density = 1.823*(10**18)*np.sum(data, axis=0)*dv

# get peak optical depth and (RA, DEC) of each source in LMC

# IF we get GaussPy+ fixed we also need to recheck the "good" spectra that didn't work.

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

LMC_no_abs_list = [["0514-6707.1419.fits", 149, 150],\
		["0502-6625.1419.fits", 150, 150],\
		["0507-7144.1419.fits", 148, 149], ["0528-6759.1419.fits", 150, 150],\
		["0525-674.1419.fits", 150, 150], ["0459-6658.1419.fits", 150, 150],\
		["0540-6906.1419.fits", 151, 149]]


# sig and FWHM min values
abs_min = 1.0
abs_sig = 3.0
em_min = 5.0
em_sig = 5.0

name_list = []
ra_list = []
dec_list = []
tau_list = []
noise_list = []

for source in LMC_list:

	abs_sig =  5.0
	abs_min = 1.0

	filename, x_pixel, y_pixel = source

	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	abs_obj.raw_spectra()
	abs_obj.T_bg(-50, -20)
	abs_obj.optical_depth()

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	# code for checking features

	tau_check = np.round(tau_mean)

	abs_sig = 3.0
	abs_min = 3.0
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	n = tau_spec.components

	for i in range(0,n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n + i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1

	# looks through all features

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

# plotting stuff (position colour bars?)

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = plt.subplot(projection=wcs)

column_density = np.array(column_density)/(10.**20)
im2 = ax.imshow(column_density, origin='lower', cmap="Greys")
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")

cax2 = fig.add_axes([0.72, 0.11, 0.015, 0.77])
plt.colorbar(im2, orientation="vertical", cax=cax2, label=r"$N_{H}(\tau \ll 1)$ ($10^{20}$ cm$^{-2}$)")

noabs_ra_list = list()
noabs_dec_list = list()

for source in LMC_no_abs_list:
	filename, x_pixel, y_pixel = source
	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	noabs_ra_list.append(abs_obj.ra)
	noabs_dec_list.append(abs_obj.dec)
	print(str(filename) + " : " + str(abs_obj.ra) + " : " + str(abs_obj.dec))


im3 = ax.scatter(noabs_ra_list, noabs_dec_list, transform=ax.get_transform("world"), marker="x", color="lime")
im4 = ax.scatter(ra_list, dec_list, transform=ax.get_transform("world"), marker="x", color="lime")

plt.savefig("GASS_data_analysis.pdf")
#plt.show()
