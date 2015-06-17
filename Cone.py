#!/usr/bin/env python3
from OutflowCone.Helpers import *
import numpy as np

class Cone:
    """ Galactic wind outflow cone model.
    """
    def __init__(self, inc=0, PA=0, theta=60, r_in=0.0, r_out=5.0):
        """ Create a new outflow cone.
            
            Keywords:
                inc     --  The inclination of the cone with respect to the
                            line-of-sight.
                PA      --  The position angle of the cone with respect to
                            the line-of-sight.
                theta   --  The opening half-angle of the cone (degrees).
                r_in    --  The inner radius of the cone (kpc).
                r_out   --  The outer radius of the cone (kpc).
        """
        self.inc = inc
        self.PA = PA
        self.theta = theta
        self.r_in = r_in
        self.r_out = r_out

        self.positions = None

    def GenerateClouds(self, n, bicone=False, falloff=1):
        """ Generate 'n' model clouds within the cone bounds.

            Arguments:
                n   --

            Keywords:
                bicone  --  Create a single cone or a bi-cone. Default is False.
                falloff --  Radial density distribution exponent. Default is 1 for
                            a mass-conserving outflow (density goes as r^-2).
                            A value of 1/3 creates a constant-density profile.

            Returns:
                None. Creates "positions" member variable, containing Cartesian
                position vectors for 'n' clouds, and "velocities" member variable,
                containing zero-velocity vectors for all clouds (ostensibly these
                are Cartesian as well).
        """
        self._n = n

        # Even spread in cos(theta) to avoid clustering at poles.
        theta_rad = np.radians(self.theta)
        if bicone:
            vs1 = np.random.random(self._n/2.) * \
                   (1-np.cos(theta_rad)) + np.cos(theta_rad)
            vs2 = -(np.random.random(self._n/2.) * \
                    (1-np.cos(theta_rad)) + np.cos(theta_rad))
            vs = np.concatenate((vs1, vs2))
        else:
            vs = np.random.random(self._n) * (1-np.cos(theta_rad)) + np.cos(theta_rad)
        thetas = np.arccos(vs)

        us = np.random.random(self._n)
        phis = us * np.pi * 2.0

        #falloff = 1 # 1/3 ~ constant density, 1/2 ~ 1/r fall-off, 1 ~ r^-2 fall-off
        rs = np.random.random(self._n)**falloff * (self.r_out - self.r_in) + self.r_in

        # Still need to rotate them!
        self._sph_pts = np.vstack((rs, thetas, phis)).transpose()

        # Convert to Cartesian so we can rotate things around.
        self._cart_pts = np.array([SphPosToCart(sph_pt, radians=True) for
                    sph_pt in self._sph_pts])

        # Coord system will be:
        #   -Z-axis is LOS, making X go right and Y go up (+Z out of the page)
        # Rot in -X for inc (or rotate -inc in +X) and in Z for PA
        self.positions = np.asarray([Rot(pt, x=-self.inc, z=self.PA) \
                            for pt in self._cart_pts])
        self.velocities = np.zeros_like(self.positions)

    # Properties for nicely slicing the 2D array of positions and velocities
    # into lists for each coordinate.
    @property
    def _rs(self):
        return self._sph_pts[:,0]

    @property
    def _thetas(self):
        return self._sph_pts[:,1]

    @property
    def _phis(self):
        return self._sph_pts[:,2]

    @property
    def _xs(self):
        return self.positions[:,0]

    @property
    def _ys(self):
        return self.positions[:,1]

    @property
    def _zs(self):
        return self.positions[:,2]

    @property
    def _vxs(self):
        return self.velocities[:,0]
    @_vxs.setter
    def _vxs(self, vxs):
        if np.shape(vxs) == np.shape(self._vxs):
            self.velocities[:,0] = vxs
        else:
            raise Exception("Array is not the same length as self._vxs.")

    @property
    def _vys(self):
        return self.velocities[:,1]
    @_vys.setter
    def _vys(self, vys):
        if np.shape(vys) == np.shape(self._vys):
            self.velocities[:,1] = vys
        else:
            raise Exception("Array is not the same length as self._vys.")

    @property
    def _vzs(self):
        return self.velocities[:,2]
    @_vzs.setter
    def _vzs(self, vzs):
        if np.shape(vzs) == np.shape(self._vzs):
            self.velocities[:,2] = vzs
        else:
            raise Exception("Array is not the same length as self._vzs.")

if __name__ == "__main__":
    pass
