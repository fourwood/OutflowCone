# OutflowCone
Galactic wind outflow cone model.

`Helpers.py` functions are imported automatically when `import OutflowCone` is executed.

See the `examples` folder for how you might use this.

## Cone.py
    Help on class Cone in OutflowCone:

    OutflowCone.Cone = class Cone
        Galactic wind outflow cone model.
        
        Methods defined here:
        
        GenerateClouds(self, n_clouds, bicone=False, falloff=1, zero_z=False, flatten=False)
            Generate 'n' model clouds within the cone bounds.
            
            Arguments:
                n_clouds--
            
            Keywords:
                bicone  --  Create a single cone or a bi-cone. Default is False.
                falloff --  Radial density distribution exponent. Default is 1 for
                            a mass-conserving outflow (density goes as r^-2).
                            A value of 1/3 creates a constant-density profile.
                zero_z  --  r_in makes the z-height of the base of the cone non-zero.
                            Should the clouds all get translated down? (e.g. z -= r_in)
                flatten --  Keep the inner radius spherical? Or flatten it?
            
            Returns:
                None. Creates "positions" member variable, containing Cartesian
                position vectors for 'n' clouds, and "velocities" member variable,
                containing zero-velocity vectors for all clouds (ostensibly these
                are Cartesian as well).
        
        GetLOSCloudVels(self, coord, dx, dy=None)
            Returns an array of all the projected velocities along a line of sight.
            
            Arguments:
                coord   --  x/y (~RA/dec) coordinate pair, in projected kpc,
                            of the requested line-of-sight.
                dx      --  Full width of the x-direction spatial bin, in kpc.
            
            Keywords:
                dy      --  Optional. Full width of hte y-direction spatial bin,
                            in kpc. If None/omitted, then dy = dx is assumed.
            
            Returns:
                NumPy masked array containing all LOS velocities, with velocities for
                clouds located inside the requested LOS bin *not* masked (i.e. set
                to *False* in return.mask).
        
        ProjectVels(self, LOS)
            Projects the 3D Cartesian velocities into the given line-of-sight.
            NOTE:   UNTESTED!  So far I do a lot of assuming that -z is the LOS,
                    but maybe this still works okay.
            NOTE2:  If you just want it projected into a Cartesian axis,
                    use self._vxs, ._vys, and ._vzs to avoid all the dot products.
            
            Arguments:
                LOS --  3-element NumPy array representing the LOS vector.
                        NOTE: UNTESTED!  So far I do a lot of assuming that
                        -z is the LOS, but this still probably works okay.
            
            Returns:
                NumPy array of length == len(self.velocities), where each value
                is the velocity vector dotted into the given LOS.
        
        SetLocalVels(self, vels, radians=True)
            Sets the velocities of the clouds in the galaxy/cone's frame,
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
        
        __init__(self, inc=0, PA=0, theta=60, r_in=0.0, r_out=5.0)
            Create a new outflow cone.
            
            Keywords:
                inc     --  The inclination of the cone with respect to the
                            line-of-sight.
                PA      --  The position angle of the cone with respect to
                            the line-of-sight.
                theta   --  The opening half-angle of the cone (degrees).
                r_in    --  The inner radius of the cone (kpc).
                r_out   --  The outer radius of the cone (kpc).
        
        ----------------------------------------------------------------------
        Data descriptors defined here:
        
        LOS_vs
            Returns the line-of-sight velocities of all clouds.
            NOTE: Line-of-sight is assumed to be along the -z axis.

## Helpers.py
    Help on module OutflowCone.Helpers in OutflowCone:

    NAME
        OutflowCone.Helpers

    FUNCTIONS
        CartPosToCyl(vector)
            Convert a Cartesian position vector into cylindrical coordinates.
            NOTE: Haven't really used this, so it might not be great.
            
            Arguments:
                vector -- A 3-element NumPy array representing, in order,
                          Cartesian x, y, and z position coordinates.
            
            Returns a 3-element NumPy array representing, in order, the
            cylindrical r, theta, and z position coordinates of the input vector.
        
        CartPosToSph(vectors, radians=False)
            Convert a Cartesian position vector into spherical coordinate space.
            NOTE: Largely untested...
            
            Arguments:
                vectors -- A 3-element NumPy array representing, in order,
                          Cartesian x, y, and z position coordinates.
            
            Returns a 3-element NumPy array representing, in order, the
            spherical r, theta, and phi position coordinates of the input vector.
        
        CylPosToCart(vector)
            Convert a cylindrical position vector into Cartesian position.
            NOTE: Haven't really used this, so it might not be great.
            
            Arguments:
                vector -- A 3-element NumPy array representing, in order,
                          the cylindrical r, theta, and z position coordinates.
            
            Returns a 3-element NumPy array representing, in order, the
            representative Cartesian x, y, and z coordinates of the input vector.
        
        CylVecToCart(position, vector)
            Convert a cylindrical vector into Cartesian vector space.
            
            N.B.: Not optimized!  See SphVecToCart() for a better way to do this.
            
            Takes a cylindrical-space vector and its corresponding cylindrical-
            space position and returns the magnitude of the vector in x, y,
            and z Cartesian directions.
            
            Arguments:
                position -- A 3-element NumPy array, representing the position
                            of the vector in cylindrical space.
                vector -- A 3-element NumPy array, representing the vector to
                          be converted, also in cylindrical space.
            
            Returns a 3-element NumPy array representing the magnitude of
            the input vector in Cartesian vector space.
        
        Rot(vectors, x=0.0, y=0.0, z=0.0, radians=False)
            Rotate Cartesian vectors.
            
            This function rotates input vectors about the Cartesian axes.
            The rotation is in a right-handed sense.
            
            Arguments:
                vector  --  A NumPy array of vectors to be rotated.
            
            Keywords:
                x       --  The angle to rotate about the x-axis.
                y       --  The angle to rotate about the y-axis.
                z       --  The angle to rotate about the z-axis.
                radians --  Whether the above angles are given in radians (True)
                            or degrees (False; default).
            
            Returns a NumPy array representing the rotated vectors.
        
        RotCurve(vel, radius, C=0.3, p=1.35)
            Create an analytic disk galaxy rotation curve.
            
            Arguments:
                vel    -- The approximate maximum circular velocity.
                radius -- The radius (or radii) at which to calculate the
                          rotation curve.
            
            Keywords:
                C      -- Controls the radius at which the curve turns over,
                          in the same units as 'radius'.
                p      -- Controls the fall-off of the curve after the turn-over;
                          values expected to be between 1 and 1.5 for disks.
            
            Returns the value of the rotation curve at the given radius.
            See Bertola et al. 1991, ApJ, 373, 369 for more information.
        
        RotX(vector, angle, radians=False)
            Rotate a Cartesian vector by a given angle about the +x axis.
            NOTE: Probably needs to be re-written (to be like Rot()) for accepting
            arrays of vectors.
            
            This function rotates a given vector about the Cartesian +x axis.
            The rotation is in a right-handed sense; positive angles rotate
            from the +y axis toward the +z axis.
            
            Arguments:
                vector -- A 3-element NumPy array to be rotated.
                angle -- The angle by which the input vector will be rotated.
            
            Returns a 3-element NumPy array representing the rotated vector.
        
        RotY(vector, angle, radians=False)
            Rotate a Cartesian vector by a given angle about the +y axis.
            NOTE: Probably needs to be re-written (to be like Rot()) for accepting
            arrays of vectors.
            
            This function rotates a given vector about the Cartesian +y axis.
            The rotation is in a right-handed sense; positive angles rotate
            from the +z axis toward the +x axis.
            
            Arguments:
                vector -- A 3-element NumPy array to be rotated.
                angle -- The angle by which the input vector will be rotated.
            
            Keywords:
                radians -- Whether 'angle' is in radians (True) or degrees (False; default).
            
            Returns a 3-element NumPy array representing the rotated vector.
        
        RotZ(vector, angle, radians=False)
            Rotate a Cartesian vector by a given angle about the +z axis.
            NOTE: Probably needs to be re-written (to be like Rot()) for accepting
            arrays of vectors.
            
            This function rotates a given vector about the Cartesian +z axis.
            The rotation is in a right-handed sense; positive angles rotate
            from the +x axis toward the +y axis.
            
            Arguments:
                vector -- A 3-element NumPy array to be rotated.
                angle -- The angle by which the input vector will be rotated.
            
            Keywords:
                radians -- Whether 'angle' is in radians (True) or degrees (False; default).
            
            Returns a 3-element NumPy array representing the rotated vector.
        
        SphPosToCart(vectors, radians=False)
            Convert a spherical position vector into Cartesian position.
            
            Arguments:
                vector -- A 3-element NumPy array representing, in order,
                          the spherical r, theta, and phi position coordinates.
            
            Returns a 3-element NumPy array representing, in order, the
            representative Cartesian x, y, and z coordinates of the input vector.
        
        SphVecToCart(position, vector, radians=False)
            Convert spherical vectors into Cartesian vector space.
            
            Takes spherical-space vectors and their corresponding spherical-
            space positions and returns the magnitude of the vectors in x, y,
            and z Cartesian directions.
            
            Arguments:
                position -- A 3-element NumPy array, or array of arrays, representing
                            the position of the vector(s) in spherical space.
                vector   -- A 3-element NumPy array, or array of arrays, representing
                            the vector(s) to be converted, also in spherical space.
            
            Returns a 3-element NumPy array representing the magnitude of
            the input vector in Cartesian vector space.
