#!/usr/bin/env python3

# Should be all Python3 compatible, and I'd suggest keeping it that way.
# However, it should *also* be all Python2.7 compatible, as well.

from __future__ import print_function, division
import numpy as np
import OutflowCone as oc

# How many clouds to generate in the cone model?
# 1e4 gives a *really, really* rough result, but probably okay for seeing things.
# 1e5 is pretty fine, but still sees slight variation with different random seeds.
# 1e6 is pretty smooth.  Takes a bit longer, though, of course.
n_points = 1e5
inc = 45
PA = 90
theta = 60
r_in = 0.5
r_out = 6.0
vel_vec = np.array([30., 0, 225.])
# Rotation curve paramters
C, p = (0.5, 1.35)
accel = vel_vec[0]

# The code implicitly assumes the line-of-sight is directed along -z. A.K.A. anything
# moving _into_ the LOS is moving in the +z direction. So this next variable is just
# a reminder, and you probably shouldn't try to change this without going into
# the model code.
#LOS = np.array([0, 0, -1])

## LOS grid
# ~plate scale in kpc/pix
step = 0.1
# Xs, aka RA
x_min, x_max = (-1, 1)
# This will represent the lower bin edges, essentially.
sky_xs = np.arange(0, x_max-x_min, step)
nxs = len(sky_xs)
sky_xs -= step * (nxs//2)
# Ys, aka dec
y_min, y_max = (-1, 1)
# This will represent the lower bin edges, essentially.
sky_ys = np.arange(0, y_max-y_min, step)
nys = len(sky_ys)
sky_ys -= step * (nys//2)

cone = oc.Cone(inc=inc, PA=PA, theta=theta, r_in=r_in, r_out=r_out)
# 'flatten' means make it so the cone is flat on the bottom at a disk height of r_in;
# otherwise it'll be a spherical surface of radius r_in. If zero_z is true, the minimum
# z-height of the cone will be 0, otherwise it'll be some non-zero value based on r_in.
cone.GenerateClouds(n_points, flatten=True, zero_z=True)

# Make a rotation curve for each point.
sph_vels = np.repeat(np.asarray([vel_vec]), cone._n_clouds, axis=0)
R_max0 = r_in * np.sin(np.radians(theta))
R_maxs = R_max0 + cone._local_zs * np.tan(np.radians(theta))
Rs = np.sqrt(cone._local_xs**2+cone._local_ys**2)
R0s = R_max0 * Rs / R_maxs
rot_vels = oc.RotCurve(vel_vec[2], R0s, C=C, p=p)
rot_vels *= R0s / cone._local_rs #Ang. mom. cons.
sph_vels[:,2] = rot_vels
R_0 = 1. # kpc normalization for acceleration
sph_vels[:,0] += 1.0 * accel * (cone._local_rs / R_0)

# This function will convert the spherical-coordinate velocities
# into Cartesian, accounting for inclination.
cone.SetLocalVels(sph_vels)

## LOS is ~hard-coded, so these are equivalent.
#proj_vels = -cone._vzs
proj_vels = cone.LOS_vs

# Now we're gonna bin up velocities in the LOS grid and do some convolutions.
max_vels = np.zeros((nys, nxs))
fwhms = np.zeros_like(max_vels)
radii = np.sqrt(cone._xs**2 + cone._ys**2 + cone._zs**2)
weights = 1/radii**2 # 1/r^2 weighting approximates photoionization 

inst_res = 17.0
turb_sigma = 90.0
total_broadening = np.sqrt(inst_res**2 + turb_sigma**2)

for j, y in enumerate(sky_ys):
    for i, x in enumerate(sky_xs):
        masked_vels = cone.GetLOSCloudVels((x, y), step)
        mask = ~masked_vels.mask # foo is a masked array, we want what *isn't* masked, really.
        bin_vels = masked_vels[mask]
        bin_weights = weights[mask]
        bin_radii = radii[mask]

        if len(bin_vels) > 0:
            v_min, v_max = (-500., 500.) # Enough to encompass everything...
            bin_size = 5.                # but probably shouldn't be hard-coded.
            hist_bins = np.linspace(v_min, v_max, (v_max-v_min)/bin_size+1)
            n, edges = np.histogram(bin_vels, bins=hist_bins, weights=bin_weights)

            # Convolve the velocity profile with turb/inst broadening
            gauss_bins = np.linspace(10*v_min,
                                     10*v_max,
                                     (10*v_max-10*v_min)/bin_size+1)
            gauss = np.exp(-gauss_bins**2/(2*total_broadening**2)) / \
                    (total_broadening*np.sqrt(2*np.pi)) * bin_size

            convol = np.convolve(n, gauss, mode='same')

            ## Estimate the FWHM.
            #g_max_idx = np.where(convol == np.max(convol))[0][0]
            #g_max = convol[g_max_idx]
            #g_hmax = g_max / 2.
            #left_hmax_idx = np.abs(convol[:g_max_idx]-g_hmax).argmin()
            #right_hmax_idx = np.abs(convol[g_max_idx:]-g_hmax).argmin() + g_max_idx
            #fwhm_est = gauss_bins[right_hmax_idx] - gauss_bins[left_hmax_idx]
            #fwhms[j,i] = fwhm_est

            max_vels[j,i] = np.average(gauss_bins, weights=convol)
        else:
            max_vels[j,i] = np.nan

## Print some stats:
#print("Outflow velocity stats:")
#print("Minimum:\t\t{:0.1f}".format(np.min(sph_vels[:,0])))
#print("Maximum:\t\t{:0.1f}".format(np.max(sph_vels[:,0])))
#print("Median:\t\t\t{:0.1f}".format(np.median(sph_vels[:,0])))
#print("Average:\t\t{:0.1f}".format(np.average(sph_vels[:,0])))
#print("1/r^2 weighted-average:\t{:0.1f}".format(np.average(sph_vels[:,0],
#                                                weights=1/radii**2)))

###############
## Plotting! ##
###############
import pylab as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

cmap = plt.cm.RdBu_r
cmap_vals = cmap(np.arange(256))
scale = 0.75
new_cmap_vals = np.array([[scale*val[0],
                           scale*val[1],
                           scale*val[2],
                           scale*val[3]] for val in cmap_vals])
new_cmap = mpl.colors.LinearSegmentedColormap.from_list("new_cmap", new_cmap_vals)

params = {'figure.subplot.left':    0.10,
          'figure.subplot.right':   0.93,
          'figure.subplot.bottom':  0.12,
          'figure.subplot.top':     0.97,
          'text.usetex':            True,
          'font.size':              10}
plt.rcParams.update(params)
figsize = (3.35, 2.75)
fig, ax = plt.subplots(figsize=figsize)

## LOS grid
#for y in sky_ys:
#    for x in sky_xs:
#        plt.plot([x, x], [y_min, y_max],
#                'k-', linewidth=0.1, zorder=0.1)
#        plt.plot([x_min, x_max], [y, y],
#                'k-', linewidth=0.1, zorder=0.1)

v_min, v_max = (-150, 150)

# Main image
n_levels = 41
levels = np.linspace(v_min, v_max, n_levels)
extents = (sky_xs[0], sky_xs[-1]+step,
           sky_ys[0], sky_ys[-1]+step)
img = ax.contourf(max_vels, cmap=cmap, vmin=v_min, vmax=v_max,
                origin='lower', zorder=0, levels=levels, extend='both',
                extent=extents)

# Contour overlay
c_levels = 11
c_levels = np.linspace(v_min, v_max, c_levels)
contours = ax.contour(max_vels, cmap=new_cmap, vmin=v_min, vmax=v_max,
                origin='lower', levels=c_levels, extend='both',
                extent=extents)

# Crosshairs
ax.plot([x_min, x_max], [0, 0], 'k:', scalex=False, scaley=False)
ax.plot([0, 0], [y_min, y_max], 'k:', scalex=False, scaley=False)

# Colorbar
ticks = np.arange(v_min, v_max+1, 50)
ticks = c_levels
bar = plt.colorbar(img, ticks=ticks, pad=0.025)
bar.set_label(r'Velocity (km s$^{-1}$)', labelpad=3.0)

plt.xlabel('Distance (kpc)', labelpad=0.)
plt.ylabel('Distance (kpc)', labelpad=-5.)

x_maj_ticks = ticker.MaxNLocator(2)
x_min_ticks = ticker.AutoMinorLocator(10)
y_maj_ticks = ticker.MaxNLocator(2)
y_min_ticks = ticker.AutoMinorLocator(10)
ax.xaxis.set_major_locator(x_maj_ticks)
ax.xaxis.set_minor_locator(x_min_ticks)
ax.yaxis.set_major_locator(y_maj_ticks)
ax.yaxis.set_minor_locator(y_min_ticks)

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.savefig("cone_example.eps")
plt.close()
#plt.show()
