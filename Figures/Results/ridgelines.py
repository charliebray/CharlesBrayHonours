from classes import *
from scipy.optimize import fmin
import scipy.integrate as integrate

import math

filename_gass = "gass_smallest.fits"

LMC_list = [["0452-6823.1419.fits", 150, 150], ["0502-6625.1419.fits", 150, 150],\
                ["0507-6657.1419.fits", 128, 148], ["0520-7312.1419.fits", 150, 149],\
                ["0507-7144.1419.fits", 148, 149], ["0552-6814.1419.fits", 148, 152],\
                ["0516-7336.1419.fits", 151, 150], ["0450-6940.1419.fits", 150, 150],\
                ["0512-7233.1419.fits", 150, 150], ["0528-6759.1419.fits", 150, 150],\
                ["0525-674.1419.fits", 150, 150], ["0527-6549.1419.fits", 150, 150],\
                ["0518-6617.1419.fits", 150, 150], ["0506-6555.1419.fits", 150, 150],\
                ["0521-6960.1419.fits", 149, 150], ["0459-6658.1419.fits", 150, 150],\
                ["0522-7038.1419.fits", 150, 149], ["0540-6906.1419.fits", 151, 149],\
                ["0547-7206.1419.fits", 150, 151]][11:12]

a_list = list()

fwhm_list = list()

# Significance and min FWHM values
abs_sig = 3.0
abs_min = 3.0

em_sig = 5.0
em_min = 3.0

for source in LMC_list:

	abs_sig = 3.0
	abs_min = 3.0

	filename, x_pixel, y_pixel = source

	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	abs_obj.raw_spectra()
	abs_obj.T_bg(-50,-20)
	abs_obj.optical_depth()

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	em_spec = Plotting_Spectra(filename_gass, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()

	# The following are their respective functions.

	def gauss_func(amp, fwhm, mu, x):
		return amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2))

	def T_B_model(x):
		return sum(em_spec.gaussian_list(x))

	def tau_model(x):
		return sum(tau_spec.gaussian_list(x))

	def EW_func(amp, fwhm, mean, x):
		return (1 - np.exp(-gauss_func(amp, fwhm, mean, x)))

	def N_func(x):
		return ((1.823*(10**18))*T_B_model(x))

	def iso_func(x):
		return ((1.823*(10**18)) * T_B_model(x) * (1 + (tau_model(x)/2) + (tau_model(x)**2)/12) - ((tau_model(x)**4)/720))

	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	tau_check = np.round(tau_spec.params[2*n:])

#	begin with 3 SNR and 3 min width then go to (5,1)
	abs_sig = 5.0
	abs_min = 1.0
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	n = tau_spec.components

#       this code makes sure that the same components aren't being fit twice
	for i in range(0,n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n+i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1

	for i in range(0,1):	# m instead of 1, simply done for figure.
		amp = tau_amp[i]
		fwhm = tau_fwhm[i]
		mean = tau_mean[i]
		if np.abs(mean) > 50.:
			continue
		x = np.linspace(mean - fwhm/2., mean + fwhm/2., 100)
		m, y0 = np.polyfit(EW_func(amp, fwhm, mean, x), T_B_model(x), 1)
		T_ew = y0
		T_c = m + (0.5*T_ew)
		error = 0.25*T_ew

		a_list.append([filename, amp, fwhm, mean, T_ew, m, T_c, error])

cloud_spin_list = list()

for b_list in a_list:
	filename, amp, fwhm, mean, T_ew, m, T_c, error = b_list
	amp = round(amp, 2)
	fwhm = round(fwhm, 2)
	mean = round(mean, 2)
	T_ew = round(T_ew, 2)
	m = round(m, 2)
	T_c = round(T_c, 2)
	error = round(error, 2)

#	print(filename + ": " + str(amp) + ", " + str(fwhm) + ", " + str(mean) + ", " + str(T_ew) + ", " + str(m) + ", " + str(T_c) + ", " + str(error))

	cloud_spin_list.append(T_c)

##

em_T_B = em_obj.T_B
em_vrad = em_obj.vrad
abs_EW = (1 - np.exp(-abs_obj.tau_raw))
abs_vrad = abs_obj.vrad

from scipy.interpolate import interp1d

f_interp = interp1d(em_vrad, em_T_B, kind='quadratic', bounds_error=False)
new_T_B = f_interp(abs_vrad)

ye_EW = abs_EW[151-2:151+2]
ye_T_B = new_T_B[151-2:151+2]

##

print(cloud_spin_list)
#x = np.linspace(int(mean - 5), int(mean + 5), 1000)
plt.plot(EW_func(amp, fwhm, mean, x), T_B_model(x), label=r"Model Comparison")
x = np.linspace(-0.05, 0.6, 100)
plt.plot(x, x*m + T_ew, label=r"Ridge Line ($q=0.5$)", linestyle='--')
plt.scatter(ye_EW, ye_T_B, color="green", marker="x", label="Raw Spectra")
#plt.plot(EW_func(amp, fwhm, mean, x), EW_func(amp, fwhm, mean, x)*m + T_ew, label="Ridge Line ($q=0.5$)", linestyle='--')
plt.xlabel(r"$(1 - e^{-\tau(v)})$")
plt.ylabel(r"$T_{B}(v)$ (K)")
plt.grid()
plt.legend()
plt.show()
#plt.savefig("exampleRidgeLine.pdf")
#print(np.polyfit(EW_func(x), T_B_model(x), 1))
