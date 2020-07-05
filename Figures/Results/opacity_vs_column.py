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

tau_list = []		# peak optical depth list
iso_list = []
data_list = []
noise_list = []

mean_list = []	# for putting into table

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

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	em_spec = Plotting_Spectra(filename_gass, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()

	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	# checks for additional thin features
	tau_check = np.round(tau_mean)

	abs_sig = 5.0
	abs_min = 1.0
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

	def gauss_func(amp, fwhm, mu, x):
		return amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2))

	def T_B_model(x):
		return sum(em_spec.gaussian_list(x))

	def tau_model(amp_list, fwhm_list, mean_list, x):
		a_sum = 0
		for i in range(0, len(amp_list)):
			amp = amp_list[i]
			fwhm = fwhm_list[i]
			mean = mean_list[i]
			a_sum += gauss_func(amp, fwhm, mean, x)
		return a_sum

	def iso_col(amp, fwhm, mu, x):
		T_B_model = sum(em_spec.gaussian_list(x))
		return 1.823*(10**18)*T_B_model*(1 + (tau_model(amp, fwhm, mu, x/2)) + (tau_model(amp, fwhm, mu, x)**2/12) - (tau_model(amp, fwhm, mu, x)**4/720))
#		return 1.823*(10**18)*tau_model*T_B_model/(1 - np.exp(-tau_model))

	for i in range(0,m):
		amp = tau_amp[i]		# amplitude of opacity gaussian
		fwhm = tau_fwhm[i]
		mu = tau_mean[i]
		if np.abs(mu) > 50.:
			continue
		mean_list.append([filename, mu])
		tau_list.append(amp)		# list of peak optical depths (with v_rad within +-50km/s)
		iso = integrate.quad(lambda x: iso_col(tau_amp, tau_fwhm, tau_mean, x), -np.inf, np.inf)[0]
#		iso = integrate.quad(lambda x: iso_col(amp, fwhm, mu, x), float(mu - (fwhm/2.)), float(mu + (fwhm/2.)))[0]
		iso_list.append(float(iso))

		noise_list.append(abs_obj.tau_noise)

		data_list.append([filename, float(iso)])


tau_array = np.array(tau_list)
iso_array = np.array(iso_list)

plt.scatter(iso_array, tau_array)
plt.errorbar(iso_array, tau_array, yerr=noise_list, marker='o', fmt=' ', capsize=4, elinewidth=1.5)
plt.xlabel(r"$N_{H, iso} (cm^{-2})$")
plt.ylabel(r"$\tau_{peak}$")
plt.grid()
#plt.savefig("opacity_vs_column.pdf")
plt.show()
