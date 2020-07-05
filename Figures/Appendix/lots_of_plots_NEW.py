from classes import * 

filename_gass = "gass_smallest.fits"

i = 7

LMC_list = [["0452-6823.1419.fits", 150, 150],\
		["0507-6657.1419.fits", 128, 148], ["0520-7312.1419.fits", 150, 149],\
		["0552-6814.1419.fits", 148, 152],\
		["0516-7336.1419.fits", 151, 150], ["0450-6940.1419.fits", 150, 150],\
		["0512-7233.1419.fits", 150, 150],\
		["0527-6549.1419.fits", 150, 150],\
		["0518-6617.1419.fits", 150, 150], ["0506-6555.1419.fits", 150, 150],\
		["0521-6960.1419.fits", 149, 150],\
		["0522-7038.1419.fits", 150, 149],\
		["0547-7206.1419.fits", 150, 151]][i:i+1]

em_min = 3.0
em_sig = 5.0

for source in LMC_list:

	abs_sig = 3.0
	abs_min = 3.0

	filename, x_pixel, y_pixel = source

	abs_obj = Abs_Spectra(filename, x_pixel, y_pixel)
	abs_obj.raw_spectra()
	abs_obj.T_bg(-50, -20)
	abs_obj.optical_depth()

	em_obj = Em_Spectra(filename_gass, abs_obj.ra, abs_obj.dec)
	em_obj.raw_spectra()

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()

	# brightness temperature stuff

	em_spec = Plotting_Spectra(filename, em_obj.vrad, em_obj.T_B, em_min, em_sig)
	em_spec.gpp_parameters()
	n = em_spec.components

	em_amp = em_spec.params[:n]
	em_fwhm = em_spec.params[n:2*n]
	em_mean = em_spec.params[2*n:]

	# define the gaussian function here

	def gauss_func(amp, fwhm, mu, x):
		return amp*np.exp(-4*np.log(2) * (x-mu)**2/(fwhm**2))

	# counting absorption components

	n = tau_spec.components
	m = n
	tau_amp = tau_spec.params[:n]
	tau_fwhm = tau_spec.params[n:2*n]
	tau_mean = tau_spec.params[2*n:]

	tau_check = np.round(tau_mean)

	# checking for thin components

	abs_sig = 5.0
	abs_min = 1.0

	tau_spec = Plotting_Spectra(filename, abs_obj.vrad, abs_obj.tau_raw, abs_min, abs_sig)
	tau_spec.gpp_parameters()
	n = tau_spec.components

	for i in range(0,n):
		if round(tau_spec.params[2*n + i]) in tau_check:
			continue
		tau_amp = np.append(tau_amp, tau_spec.params[i])
		tau_fwhm = np.append(tau_fwhm, tau_spec.params[n+i])
		tau_mean = np.append(tau_mean, tau_spec.params[2*n + i])
		m += 1
# now we have all the parameters for each component of the absorption spectra
#		and the emission spectra.

# we want to play a (2x2) figure with raw spectra on LHS and fitting on the RHS
fig, axes = plt.subplots(2,2, sharex=True, sharey="row", figsize=(8,8))
axes[1,0].set_xlim(-50, 50)
axes[1,1].set_xlim(-50, 50)

# Raw plot of optical depth spectra
axes[0,0].plot(abs_obj.vrad, abs_obj.tau_raw)
axes[0,0].set_ylabel(r"Optical Depth, $\tau(v)$", fontsize=12)

axes[1,0].set_xlabel(r"Radial Velocity, $v$ (km/s)", fontsize=12)
axes[1,1].set_xlabel(r"Radial Velocity, $v$ (km/s)", fontsize=12)

# Raw plot of emission spectra
axes[1,0].plot(em_obj.vrad, em_obj.T_B)
axes[1,0].set_ylabel(r"Brightness Temperature, $T_{B}(v)$ (K)", fontsize=12)

# Plot of raw optical depth spectra and model
axes[0,1].plot(abs_obj.vrad, abs_obj.tau_raw)

sum = 0
for index in range(0, len(tau_amp)):
	amp = tau_amp[index]
	fwhm = tau_fwhm[index]
	mean = tau_mean[index]
	sum += gauss_func(amp, fwhm, mean, abs_obj.vrad)
#	axes[0,1].plot(abs_obj.vrad, gauss_func(amp, fwhm, mean, abs_obj.vrad), color='darkorange')
axes[0,1].plot(abs_obj.vrad, sum)
# plot of raw emission with model
axes[1,1].plot(em_obj.vrad, em_obj.T_B)

sum = 0
for index in range(0, len(em_amp)):
	amp = em_amp[index]
	fwhm = em_fwhm[index]
	mean = em_mean[index]
	axes[1,1].plot(em_obj.vrad, gauss_func(amp, fwhm, mean, em_obj.vrad), color='orange', linestyle='--')
	sum += gauss_func(amp, fwhm, mean, em_obj.vrad)
axes[1,1].plot(em_obj.vrad, sum)

plt.subplots_adjust(wspace=0.1, hspace=0.1)
fig.align_ylabels(axes[:,0])
plt.savefig("appendixFigure.pdf")
