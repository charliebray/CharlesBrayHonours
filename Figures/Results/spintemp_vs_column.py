from classes import *
from scipy.optimize import fmin
import scipy.integrate as integrate

filename_gass = "gass_smallest.fits"

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

spin_list = []		# mean spin tempearture for each cloudlet
iso_list = []		# isothermal column density of each absorbing cloud
tau_list = []		# peak optical depth list
tau_noise_list = []	# our uncertainty I suppose?

# Significance and min FWHM values
abs_sig = 3.0
abs_min = 1.0

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
	tau_noise_list.append(float(abs_obj.tau_noise))

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	em_spec = Plotting_Spectra(filename_gass, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()

	# The following returns the peak optical depth, column density (over fhm of tau),
	#	and the avg spin temperature of the sum (over fwhm of tau).

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
		return amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2))

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

	def tau_model(amp_list, fwhm_list, mean_list, x):
		a_sum = 0
		for i in range(0, len(amp_list)):
			amp = amp_list[i]
			fwhm = fwhm_list[i]
			mean = mean_list[i]
			a_sum += gauss_func(amp, fwhm, mean, x)
		return a_sum

	def iso_func(amp, fwhm, mu, x):
		return (N_func(x) * (1 + (tau_model(amp, fwhm, mu, x)/2 + (tau_model(amp, fwhm, mu, x)**2/12) - (tau_model(amp, fwhm, mu, x)**4/720))))

	for i in range(0,m):
		amp = tau_amp[i]		# amplitude of opacity gaussian
		fwhm = tau_fwhm[i]		# fwhm of opacity gaussian
		mu = tau_mean[i]		# mean of opacity gaussian
		if np.abs(mu) > 50.:
			continue
		tau_list.append(amp)		# list of peak optical depths
		N_uncor = integrate.quad(lambda x: N_func(x), -np.inf, np.inf)[0]
		EW = integrate.quad(lambda x: EW_func(tau_amp, tau_fwhm, tau_mean, x), -np.inf, np.inf)[0]
		spin = N_uncor/(1.823*(10**18) * EW)
		spin_list.append(spin)	# list containing the spin temperatures
		iso = integrate.quad(lambda x: iso_func(tau_amp, tau_fwhm, tau_mean, x), -np.inf, np.inf)[0]
		iso_list.append(iso)

iso_array = np.array(iso_list)
spin_array = np.array(spin_list)

spin_array = [336.06, 35.75, 120.49, 61.25, 242.33, 242.33, 298.68, 421.29, 145.74, 145.74, 41.37, 202.47, 179.41, 192.34, 58.96]
spin_unc = [71.17, 2.66, 15.81, 3.7, 29.94, 29.94, 0, 0, 7.77, 7.77, 1.63, 33.11, 41.28, 0, 7.63]

plt.scatter(iso_array, spin_array)
plt.errorbar(iso_array, spin_array, yerr=spin_unc, marker='o', fmt=' ', capsize=4, elinewidth=1.5)
plt.xlabel(r"$N_{H, iso} (cm^{-2})$")
plt.ylabel(r"$<T_{S}> (K)$")
plt.grid()
plt.savefig("spintemp_vs_column.pdf")
