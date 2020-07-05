import astropy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate

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

LMC_list = [["0527-6549.1419.fits", 150, 150]]

# sig and FWHM min values
abs_min = 1.0
abs_sig = 3.0
em_min = 3.0
em_sig = 5.0

name_list = []
ra_list = []
dec_list = []
tau_list = []
noise_list = []
spin_list = []
EW_list = list()
N_list = list()

for source in LMC_list:

	abs_sig = 3.0
	abs_min = 3.0

	filename, x_pixel, y_pixel = source

	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	abs_obj.raw_spectra()
	abs_obj.T_bg(-50, -20)
	abs_obj.optical_depth()
	ra_list.append(abs_obj.ra)
	dec_list.append(abs_obj.dec)

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	em_obj = Em_Spectra(fits_dir, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	em_spec = Plotting_Spectra(fits_dir, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()

	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	# checking for other features

	tau_check = np.round(tau_mean)

	abs_sig = 5.0
	abs_min = 1.0
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	n = tau_spec.components

	for i in range(0, n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n + i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1

	def gauss_func(amp, fwhm, mu, x):
		return (amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2)))

	def N_func(x):
		return (1.823*(10**18)*sum(em_spec.gaussian_list(x)))

	def EW_func(amp_list, fwhm_list, mean_list, x):
		a_sum = 0
		for i in range(0, len(amp_list)):
			amp = amp_list[i]
			fwhm = fwhm_list[i]
			mean = mean_list[i]
			a_sum += gauss_func(amp, fwhm, mean, x)
		return (1 - np.exp(-a_sum))

	for i in range(0, m):
		amp = tau_amp[i]
		mu = tau_mean[i]
		fwhm = tau_fwhm[i]
		if np.abs(mu) > 50.:
			continue
		N_uncor = integrate.quad(lambda x: N_func(x), -np.inf, np.inf)[0]
		EW = integrate.quad(lambda x: EW_func(tau_amp, tau_fwhm, tau_mean, x), -np.inf, np.inf)[0]
		spin = N_uncor/(1.823*(10**(18)) * EW)
		spin_list.append(spin)
		name_list.append(filename)
		ra_list.append(abs_obj.ra)
		dec_list.append(abs_obj.dec)
		tau_list.append(amp)
		N_list.append(N_uncor)
		EW_list.append(EW)
		noise_list.append(abs_obj.tau_noise)


# plotting stuff (position colour bars?)

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = plt.subplot(projection=wcs)

marker_size = 2*np.array(spin_list)
im1 = ax.scatter(ra_list, dec_list, c=spin_list, transform=ax.get_transform("world"), edgecolors='b', s=marker_size, cmap="plasma", alpha=0.5)

#
for index in range(0, len(name_list)):
	print(str(name_list[index]) + ": " + str(spin_list[index]) + ", " + str(N_list[index]) + ", " + str(EW_list[index]))
#

#cax1 = fig.add_axes([0.327, 0.83, 0.25, 0.03])
cax1 = fig.add_axes([0.425, 0.83, 0.27, 0.03])
cbar = plt.colorbar(im1, orientation="horizontal", cax=cax1, ticks=np.arange(0., 600., 75.), extend='max', extendrect=True, label=r"$<T_S> (K)$", alpha=1)
cbar.set_alpha(1)
cbar.draw_all()

column_density = np.array(column_density)/(10.**20)
im2 = ax.imshow(column_density, origin='lower', cmap="Greys")
plt.grid(color="black", ls="dotted")
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


im3 = ax.scatter(noabs_ra_list, noabs_dec_list, transform=ax.get_transform("world"), marker="x")

from matplotlib.patches import Rectangle
#r = Rectangle((80.89375-(10.75/2), -69.75611111-(9.17/2)), 10.75, 9.17, transform=ax.get_transform("world"), edgecolor="yellow", facecolor='none')
#ax.add_patch(r)

plt.savefig("GASS_spintemp.pdf")
#plt.show()
