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
        """

        # Even spread in cos(theta) to avoid clustering at poles.
        theta_rad = np.radians(self.theta)
        if bicone:
            vs_1 = np.random.random(n/2.) * \
                   (1-np.cos(theta_rad)) + np.cos(theta_rad)
            vs_2 = -(np.random.random(n/2.) * \
                    (1-np.cos(theta_rad)) + np.cos(theta_rad))
            vs = np.concatenate((vs1, vs2))
        else:
            vs = np.random.random(n) * (1-np.cos(theta_rad)) + np.cos(theta_rad)
        thetas = np.arccos(vs)

        us = np.random.random(n)
        phis = us * np.pi * 2.0

        #falloff = 1 # 1/3 ~ constant density, 1/2 ~ 1/r fall-off, 1 ~ r^-2 fall-off
        rs = np.random.random(n)**falloff * (self.r_out - self.r_in) + self.r_in

        # Still need to rotate them!
        self._sph_pts = np.vstack((rs, thetas, phis)).transpose()

        # Convert to Cartesian so we can rotate things around.
        cart_pts = np.array([SphPosToCart(sph_pt, radians=True) for
                    sph_pt in self._sph_pts])

    # Properties for nicely slicing the 2D array of coordinates into lists of each
    # of your normal spherical coordinates.
    @property
    def _rs(self):
        return self._sph_pts[:,0]

    @property
    def _thetas(self):
        return self._sph_pts[:,1]

    @property
    def _phis(self):
        return self._sph_pts[:,2]

if __name__ == "__main__":
    pass
