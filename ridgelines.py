from classes import *
from scipy.optimize import fmin
import scipy.integrate as integrate
import math

# This python script performs the ridge line fit, I've commented out the section that performs the fit for every single absorption line of sight I used
#	and I have just left the code that performs the fit for one spectrum, and shows the plot.

# Our emission .fits file
filename_gass = "gass_smallest.fits"

# All absorption .fits files that show absorption, along with their pixel positions
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

# Lists that will store values later
a_list = list()
fwhm_list = list()

# Fitting criteria for absorption spectra
abs_sig = 3.0
abs_min = 3.0

# Fitting criteria for emission spectra
em_sig = 5.0
em_min = 3.0

for source in LMC_list:

	# First fitting criteria for absorption spectra
	abs_sig = 3.0
	abs_min = 3.0

	filename, x_pixel, y_pixel = source

	# Create absorption spectrum
	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	abs_obj.raw_spectra()
	abs_obj.T_bg(-50,-20)
	abs_obj.optical_depth()

	# Perform Gaussian decomposition on absorption spectrum
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	# Create emission spectrum
	em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	# Perform Gaussian decomposition on emission spectrum
	em_spec = Plotting_Spectra(filename_gass, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()

	# The following are useful functions.

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

	# This stores our current components using our first fitting criteria, and we move onto the next fitting criteria
	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	tau_check = np.round(tau_spec.params[2*n:])

	# Changing fitting criteria, and reperforming Gaussian decomposition
	abs_sig = 5.0
	abs_min = 1.0
	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	n = tau_spec.components

	# this code makes sure that the same components aren't being fitted twice
	for i in range(0,n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n+i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1

	# fit a ridge line to brightness temperature vs optical depth about the full width at half maximum
	# and store the parameters in a list
	for i in range(0,1):	# replace with m (number of absorption components) instead of 1, simply done for figure.
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

# rounding properties to 2 decimal places
for b_list in a_list:
	filename, amp, fwhm, mean, T_ew, m, T_c, error = b_list
	amp = round(amp, 2)
	fwhm = round(fwhm, 2)
	mean = round(mean, 2)
	T_ew = round(T_ew, 2)
	m = round(m, 2)
	T_c = round(T_c, 2)
	error = round(error, 2)
	cloud_spin_list.append(T_c)

# These are arrays from both absorption and emission spectra
em_T_B = em_obj.T_B
em_vrad = em_obj.vrad
abs_EW = (1 - np.exp(-abs_obj.tau_raw))
abs_vrad = abs_obj.vrad

from scipy.interpolate import interp1d

# interpolate such that the absorption and emission spectra have the same number of values
f_interp = interp1d(em_vrad, em_T_B, kind='quadratic', bounds_error=False)
new_T_B = f_interp(abs_vrad)

# roughly the points from the raw emission and absorption spectra that are in the full width at half maximum of detection.
ye_EW = abs_EW[151-2:151+2]
ye_T_B = new_T_B[151-2:151+2]

# we then plot our ridge line fitting
print(cloud_spin_list)
plt.plot(EW_func(amp, fwhm, mean, x), T_B_model(x), label=r"Model Comparison")
x = np.linspace(-0.05, 0.6, 100)
plt.plot(x, x*m + T_ew, label=r"Ridge Line ($q=0.5$)", linestyle='--')
plt.scatter(ye_EW, ye_T_B, color="green", marker="x", label="Raw Spectra")
plt.xlabel(r"$(1 - e^{-\tau(v)})$")
plt.ylabel(r"$T_{B}(v)$ (K)")
plt.grid()
plt.legend()
plt.show()
