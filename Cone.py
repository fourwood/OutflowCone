#!/usr/bin/env python3
import OutflowCone as oc
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

    def GenerateClouds(self, n_clouds, bicone=False, falloff=1, zero_z=False):
        """ Generate 'n' model clouds within the cone bounds.

            Arguments:
                n_clouds--

            Keywords:
                bicone  --  Create a single cone or a bi-cone. Default is False.
                falloff --  Radial density distribution exponent. Default is 1 for
                            a mass-conserving outflow (density goes as r^-2).
                            A value of 1/3 creates a constant-density profile.
                zero_z  --  r_in makes the z-height of the base of the cone non-zero.
                            Should the clouds all get translated down? (e.g. z -= r_in)

            Returns:
                None. Creates "positions" member variable, containing Cartesian
                position vectors for 'n' clouds, and "velocities" member variable,
                containing zero-velocity vectors for all clouds (ostensibly these
                are Cartesian as well).
        """
        self._n_clouds = n_clouds

        # Even spread in cos(theta) to avoid clustering at poles.
        theta_rad = np.radians(self.theta)
        if bicone:
            vs1 = np.random.random(self._n_clouds/2.) * \
                   (1-np.cos(theta_rad)) + np.cos(theta_rad)
            vs2 = -(np.random.random(self._n_clouds/2.) * \
                    (1-np.cos(theta_rad)) + np.cos(theta_rad))
            vs = np.concatenate((vs1, vs2))
        else:
            vs = np.random.random(self._n_clouds) * \
                    (1-np.cos(theta_rad)) + np.cos(theta_rad)
        thetas = np.arccos(vs)

        us = np.random.random(self._n_clouds)
        phis = us * np.pi * 2.0

        #falloff = 1 # 1/3 ~ constant density, 1/2 ~ 1/r fall-off, 1 ~ r^-2 fall-off
        rs = np.random.random(self._n_clouds)**falloff * \
                (self.r_out - self.r_in) + self.r_in

        # Still need to rotate them!
        self._sph_pts = np.vstack((rs, thetas, phis)).transpose()

        # Convert to Cartesian so we can rotate things around.
        self._cart_pts = np.array([oc.SphPosToCart(sph_pt, radians=True) for
                    sph_pt in self._sph_pts])

        if zero_z:
            self._cart_pts[:,2] -= self.r_in * np.cos(theta_rad)

        # Coord system will be:
        #   -Z-axis is LOS, making X go right and Y go up (+Z out of the page)
        # Rot in -X for inc (or rotate -inc in +X) and in Z for PA
        self.positions = oc.Rot(self._cart_pts, x=-self.inc, z=self.PA)
        self.velocities = np.zeros_like(self.positions)

    def SetLocalVels(self, vels, radians=True):#, coordinates='spherical'):
        """ Sets the velocities of the clouds in the galaxy/cone's frame,
            meaning the input velocities should be in r/theta/phi of the galaxy,
            and *without* inclination/PA taken into account. This function
            applies the inclination effects automatically.

            Arguments:
                vels    --  List/array of velocities to apply to the clouds.
                            Given velocities should be in (r, theta, phi) format.
                            List/array length must be equal to --- and assumed to
                            be in the same order as --- that of self.positions.

            Keywords:
                radians --  Are the passed theta/phi values in radians?

            Returns:
                None. self.velocities is set to the equivalent Cartesian
                velocities with inclination accounted for.
        """
        cart_vels = np.zeros_like(self._sph_pts)
        for i, pt in enumerate(self._sph_pts):
            cart_vels[i] = oc.SphVecToCart(pt, vels[i], radians=radians)

        cart_vels_rot = oc.Rot(cart_vels, x=-self.inc)
        self.velocities = cart_vels_rot

    def ProjectVels(self, LOS):
        """ Projects the 3D Cartesian velocities into the given line-of-sight.
            NOTE:   UNTESTED!  So far I do a lot of assuming that -z is the LOS,
                    but maybe this still works okay.
            NOTE2:  If you just want it projected into a Cartesian axis,
                    use self._vxs/_vys/_vzs to avoid all the dot products.

            Arguments:
                LOS --  3-element NumPy array representing the LOS vector.
                        NOTE: UNTESTED!  So far I do a lot of assuming that
                        -z is the LOS, but this still probably works okay.

            Returns:
                NumPy array of length == len(self.velocities), where each value
                is the velocity vector dotted into the given LOS.
        """
        return np.asarray([v.dot(LOS) for v in self.velocities])

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
