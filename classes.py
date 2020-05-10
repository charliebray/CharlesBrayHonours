import os
import astropy
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import mlab
from astropy.io import fits
from astropy.table import Table
from scipy import stats
from scipy.interpolate import *
from scipy.stats import norm
from astropy.wcs import WCS
from scipy.optimize import curve_fit

from astropy.convolution import convolve, Gaussian1DKernel, CustomKernel

from astropy.utils.data import get_pkg_data_filename

from gausspyplus.prepare import GaussPyPrepare
from gausspyplus.decompose import GaussPyDecompose

# Creates absorption spectra attributes, radial velocity, raw flux,
#	background source temp, and optical depth
class Abs_Spectra(object):
	
	# filename, x_px = x pixel of source, y_px = y pixel of source w.r.t python ind
	def __init__(self, filename, x_px, y_px):
		self.filename = filename
		self.x_px = x_px
		self.y_px = y_px
		self.directory = os.path.abspath(filename)
		self.hdu = fits.open(self.directory)
		self.data = self.hdu[0].data[0]
		self.vdel = self.hdu[0].header["CDELT3"]
		self.vref = self.hdu[0].header["CRVAL3"]
		self.ra = self.hdu[0].header["CRVAL1"]
		self.dec = self.hdu[0].header["CRVAL2"]

	# method assigns radial velocity in km/s and T_B as attributes
	def raw_spectra(self):
		flux_list = list()
		vrad_list = list()
		for z_int in range(0,len(self.data)):
			flux = self.data[z_int][self.y_px][self.x_px]
			vrad = (self.vdel*z_int + self.vref)/1000
			flux_list.append(flux)
			vrad_list.append(vrad)
		self.vrad = np.array(vrad_list)
		self.T_B = np.array(flux_list)

	# method assigns T_bg attribute averaged  between v_min and v_max km/s (i.e T_bg)
	def T_bg(self, v_min, v_max):
		for a_ind in range(0, len(self.T_B)):
			if int(self.vrad[a_ind]) == v_min:
				min_ind = a_ind
			if int(self.vrad[a_ind]) == v_max:
				max_ind = a_ind

		self.vmin_ind = min_ind
		self.vmax_ind = max_ind
		
		T_bg = 0
		for a_ind in range(min_ind, max_ind+1):
			T_bg += self.T_B[a_ind]
		T_bg = T_bg/len(range(min_ind, max_ind+1))

		self.T_bg = T_bg

	# method assigns optical depth = T_B/T_bg as attribute .tau
	def optical_depth(self):
		tau_array = -np.log(self.T_B/self.T_bg)
		tau_noise = round(np.std(tau_array[self.vmin_ind:self.vmax_ind+1]),3)

		self.tau_peak = max(tau_array)
		self.tau = tau_array
		self.tau_noise = tau_noise
		self.tau_raw = tau_array
		self.opacity = 1 - np.exp(-1*self.tau_raw)

		# Katie's awesome smoothing

		hanning_sz = 9
		hann_window = np.hanning(hanning_sz)
		hann_kernel = CustomKernel(hann_window)
		self.tau = convolve(self.tau, hann_kernel, boundary='extend')

# Produces all data attributes for the emission spectra, given (RA, DEC). Being a subclass
#	of Abs_Spectra, we can use the parent methods.
class Em_Spectra(Abs_Spectra):
	
	def __init__(self, filename, ra_float, dec_float):

		fits.setval(filename, "CUNIT1", value="deg")
		fits.setval(filename, "CUNIT2", value="deg")
		fits.setval(filename, "CUNIT3", value="m/s")

		self.filename = filename
		self.ra = ra_float
		self.dec = dec_float
		self.directory = os.path.abspath(filename)
		self.hdu = fits.open(self.directory)
		self.data = self.hdu[0].data
		self.wcs = WCS(self.hdu[0].header, naxis=2)
		self.x_px = int(self.wcs.all_world2pix(self.ra, self.dec, 0)[0])
		self.y_px = int(self.wcs.all_world2pix(self.ra, self.dec, 0)[1])
		self.vdel = self.hdu[0].header["CDELT3"]
		self.vref = self.hdu[0].header["CRVAL3"]

# The class Plotting_Spectra includes the ability to decompose
#	a set of data into Gaussian components, and then plots both
class Plotting_Spectra(object):
	# enter the filename, and the x and y data you would like to plot
	def __init__(self, filename, x_data, y_data, min, sig):
		self.filename = filename
		self.x_data = x_data
		self.y_data = y_data
		self.min = min
		self.sig = sig
		self.params = 0

	# Does a gaussian decomposition using GaussPyPlus
	def gpp_parameters(self):

		# JAmes Dempsey is a legend
		self.y_data = np.reshape(self.y_data, (self.y_data.shape[0],1,1))
		self.y_data.shape
		hdu = fits.PrimaryHDU(self.y_data)
		CRVAL3 = self.x_data[0]
		CDELT3 = self.x_data[1] - self.x_data[0]
		hdu1 = fits.HDUList([hdu])
		hdu1.writeto("gpp-temp.fits", overwrite=True)

		prepare = GaussPyPrepare()
		prepare.path_to_file = os.path.abspath("gpp-temp.fits")
		prepare.p_limit = 0.02
		prepare.pad_channels = 2	#5
		prepare.signal_mask = True
		prepare.min_channels = 100	# 100
		prepare.mask_out_ranges = []
		prepare.snr = self.sig		# 3.0
		prepare.significance = 5.0	# 5.0
		prepare.snr_noise_spike = self.sig	# 5.0
		data_location = (0, 0)
		prepared_spectrum = prepare.return_single_prepared_spectrum(data_location)

		decompose = GaussPyDecompose()

		decompose.two_phase_decomposition = True
		decompose.alpha1 = 2.58 # 2.58
		decompose.alpha2 = 5.14 # 5.14

		decompose.improve_fitting = True	#false

		# increase (decrease) snr and significance to decrease (increase) fittings.

		decompose.exclude_mean_outside_channel_range = True
		decompose.min_fwhm = self.min		# 1.
		decompose.max_fwhm = 64.		# none
		decompose.snr = self.sig		# 3.0
		decompose.snr_fit = None
		decompose.significance = 3.0	# 5.0
		decompose.snr_negative = None
		decompose.min_pvalue = 0.01
		decompose.max_amp_factor = 1.1
		decompose.refit_neg_res_peak = True
		decompose.refit_broad = True
		decompose.refit_blended = True
		decompose.separation_factor = 0.8493218
		decompose.fwhm_factor = 2.

		decompose.single_prepared_spectrum = prepared_spectrum
		decomposed_test = decompose.decompose()

		self.params = np.concatenate((np.array(decomposed_test["amplitudes_fit"][0]), (np.array(decomposed_test["fwhms_fit"][0])*CDELT3), ((np.array(decomposed_test["means_fit"][0])*CDELT3) + CRVAL3)))
		self.components = int(len(self.params)/3.0)

	# method that creates a list containing each gaussian component
	def gaussian_list(self, x):
		gaussian_list = list()
		num_components = int(len(self.params)/3)
		for i in range(0, num_components):
			Amp = self.params[i]
			fwhm = self.params[i+num_components]
			mu = self.params[i+int(2*num_components)]
			gaussian_list.append(Amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2)))
		return gaussian_list

	# plots the x and y data, and gaussian decomps with sum (if .gpp_parameters has been run)
	def plot_spectra(self):
		plt.plot(self.x_data, self.y_data)
		if type(self.params) == list:
			num_components = int(len(self.params)/3)
			for i in range(0, num_components):
				plt.plot(self.x_data, self.gaussian_list(self.x_data)[i], lw=0.5, ls='--', color='black')
			plt.plot(self.x_data, sum(self.gaussian_list(self.x_data)))
