#!/usr/bin/env python3
import numpy as np
from numpy.core.umath_tests import matrix_multiply

# TODO: Write unit tests for all of these helper functions.

def SphPosToCart(vectors, radians=False):
    """Convert a spherical position vector into Cartesian position.

    Arguments:
        vector -- A 3-element NumPy array representing, in order,
                  the spherical r, theta, and phi position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    representative Cartesian x, y, and z coordinates of the input vector.
    """
    if vectors.ndim == 1:
        if len(vectors) != 3:
            print("ERROR - SphPosToCart(): Vector not a 3-dimensional vector! Aborting.")
            return
    elif vectors.ndim > 2:
        print("ERROR - SphPosToCart(): Only handles a list of 3D vectors \
               (2-dimensional array). Aborting.")
        return

    if vectors.ndim == 1:
        r, theta, phi = vectors
    elif vectors.ndim == 2:
        r = vectors[:,0]
        theta = vectors[:,1]
        phi = vectors[:,2]

    if not radians:
        theta = np.radians(theta % 360)
        phi = np.radians(phi % 360)

    result = np.array([r * np.sin(theta) * np.cos(phi),
                       r * np.sin(theta) * np.sin(phi),
                       r * np.cos(theta)])

    # Transpose only has an effect for arrays of vectors, but puts vectors into
    # rows not columns, the way it should be.
    return result.T

def CartPosToSph(vectors, radians=False):
    """Convert a Cartesian position vector into spherical coordinate space.

    Arguments:
        vectors -- A 3-element NumPy array representing, in order,
                  Cartesian x, y, and z position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    spherical r, theta, and phi position coordinates of the input vector.
    """
    if vectors.ndim == 1:
        if len(vectors) != 3:
            print("ERROR - CartPosToSph(): Vector not a 3-dimensional vector! Aborting.")
            return
    elif vectors.ndim > 2:
        print("ERROR - CartPosToSph(): Only handles a list of 3D vectors \
               (2-dimensional array). Aborting.")
        return

    if vectors.ndim == 1:
        x, y, z = vectors
    elif vectors.ndim == 2:
        x = vectors[:,0]
        y = vectors[:,1]
        z = vectors[:,2]

    r = np.sqrt(x**2+y**2+z**2)
    theta = np.arctan2(np.sqrt(x**2+y**2), z)
    phi = np.arctan2(y, x)

    if not radians:
        theta = np.degrees(theta % (2*np.pi))
        phi = np.degrees(phi % (2*np.pi))

    result = np.array([r, theta, phi])

    # Transpose only has an effect for arrays of vectors, but puts vectors into
    # rows not columns, the way it should be.
    return result.T

def SphVecToCart(position, vector, radians=False):
    """Convert spherical vectors into Cartesian vector space.

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
    """
    if len(position) != len(vector):
        print("ERROR - SphVecToCart(): \
                Vector and position arrays must have the same length! Aborting.")
        return

    if position.ndim == 1 and vector.ndim == 1:
        if len(position) == 3:
            r, theta, phi = position
        else:
            print("ERROR - SphVecToCart(): \
                    Vectors and positions must each have three elements! Aborting.")
            return

    elif position.ndim == 2 and vector.ndim == 2:
        # Maybe an error-checking thing for 3-element vectors like above?
        r, theta, phi = position[:,[0,1,2]].T
    else:
        print("ERROR - SphVecToCart(): \
                Vector and position arrays must have the same dimensions, or must \
                be either 1D or 2D arrays! Aborting.")
        return

    if not radians:
        theta = np.radians(theta % 360)
        phi = np.radians(phi % 360)

    # Calculating x-hat, y-hat, and z-hat from r-hat, theta-hat, and phi-hat
    transform_matrix = np.array([
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),-np.sin(theta)],
        [-np.sin(phi), np.cos(phi), np.zeros_like(theta)]
        ])

    # Do the dot products!
    return np.squeeze(matrix_multiply(transform_matrix.T, vector[...,None]))

def RotX(vector, angle, radians=False):
    """Rotate a Cartesian vector by a given angle about the +x axis.

    This function rotates a given vector about the Cartesian +x axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +y axis toward the +z axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Returns a 3-element NumPy array representing the rotated vector.
    """
    if not radians:
        angle = np.radians(angle % 360)

    R_X = np.array([[1,             0,              0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle),  np.cos(angle)]])
    return R_X.dot(vector)

def RotY(vector, angle, radians=False):
    """Rotate a Cartesian vector by a given angle about the +y axis.

    This function rotates a given vector about the Cartesian +y axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +z axis toward the +x axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Keywords:
        radians -- Whether 'angle' is in radians (True) or degrees (False; default).

    Returns a 3-element NumPy array representing the rotated vector.
    """
    if not radians:
        angle = np.radians(angle % 360)

    R_Y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                    [             0, 1,             0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    return R_Y.dot(vector)

def RotZ(vector, angle, radians=False):
    """Rotate a Cartesian vector by a given angle about the +z axis.

    This function rotates a given vector about the Cartesian +z axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +x axis toward the +y axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Keywords:
        radians -- Whether 'angle' is in radians (True) or degrees (False; default).

    Returns a 3-element NumPy array representing the rotated vector.
    """
    if not radians:
        angle = np.radians(angle % 360)

    R_Z = np.array([[ np.cos(angle), -np.sin(angle), 0],
                    [ np.sin(angle),  np.cos(angle), 0],
                    [             0,              0, 1]])
    return R_Z.dot(vector)

def Rot(vectors, x=0., y=0., z=0., radians=False):
    """Rotate a Cartesian vector.

    This function rotates a given vector about the Cartesian axes.
    The rotation is in a right-handed sense; positive angles rotate
    from the +x axis toward the +y axis.

    Arguments:
        vector -- A NumPy array of vectors to be rotated.
        angle -- The angle by which the input vectors will be rotated.

    Keywords:
        radians -- Whether 'angle' is in radians (True) or degrees (False; default).

    Returns a NumPy array representing the rotated vectors.
    """
    if vectors.ndim == 1:
        if len(vectors) != 3:
            print("ERROR - Rot(): Vector not a 3-dimensional vector! Aborting.")
            return
    elif vectors.ndim > 2:
        print("ERROR - Rot(): Only handles a list of 3D vectors (2-dimensional array). \
               Aborting.")
        return

    if not radians:
        x = np.radians(x % 360)
        y = np.radians(y % 360)
        z = np.radians(z % 360)

    R_X = np.matrix([[         1,         0,           0],
                     [         0,  np.cos(x), -np.sin(x)],
                     [         0,  np.sin(x),  np.cos(x)]])

    R_Y = np.matrix([[ np.cos(y),          0,  np.sin(y)],
                     [         0,          1,          0],
                     [-np.sin(y),          0,  np.cos(y)]])

    R_Z = np.matrix([[ np.cos(z), -np.sin(z),          0],
                     [ np.sin(z),  np.cos(z),          0],
                     [         0,          0,          1]])

    R = R_Z * R_Y * R_X

    if vectors.ndim == 1: # A single vector
        result = R.dot(vectors).A1 # Return result as flattened array.
    elif vectors.ndim == 2: # A list of vectors
        result = R.dot(vectors.T).T.A

    return result

def CylPosToCart(vector):
    """Convert a cylindrical position vector into Cartesian position.

    Arguments:
        vector -- A 3-element NumPy array representing, in order,
                  the cylindrical r, theta, and z position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    representative Cartesian x, y, and z coordinates of the input vector.
    """
    if len(vector) != 3:
        print("WARNING - CylPosToCart(): Not a 3-dimensional vector!")
    r, phi, z = vector
    phi = np.radians(phi)
    return np.array([r * np.cos(phi),
                     r * np.sin(phi),
                     z])

def CartPosToCyl(vector):
    """Convert a Cartesian position vector into cylindrical coordinates.

    Arguments:
        vector -- A 3-element NumPy array representing, in order,
                  Cartesian x, y, and z position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    cylindrical r, theta, and z position coordinates of the input vector.
    """
    if len(vector) != 3:
        print("WARNING - CartPosToCyl(): Not a 3-dimensional vector!")
    x, y, z = vector[:3]

    r = np.sqrt(x**2+y**2)
    phi = np.degrees(np.arctan2(y, x))

    return np.array([r, phi, z])

def CylVecToCart(position, vector):
    """Convert a cylindrical vector into Cartesian vector space.

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
    """
    if len(position) != 3:
        print("WARNING - CylVecToCart(): Position not a 3-dimensional vector!")
    if len(vector) != 3:
        print("WARNING - CylVecToCart(): Vector not a 3-dimensional vector!")

    r, phi, z = position
    phi = np.radians(phi % 360)

    r_hat = np.array([np.cos(phi), #x_hat
                      np.sin(phi), #y_hat
                      0])              #z_hat

    phi_hat = np.array([-np.sin(phi), #x_hat
                         np.cos(phi), #y_hat
                         0])             #z_hat

    z_hat = np.array([0,  #x_hat
                      0,  #y_hat
                      1]) #z_hat
    transform_matrix = np.array([r_hat, phi_hat, z_hat])
    return vector.dot(transform_matrix)

def RotCurve(vel, radius, C=0.3, p=1.35):
    """Create an analytic disk galaxy rotation curve.

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
    """
    C_ = C # kpc
    p_ = p

    return vel * radius / ((radius**2 + C_**2)**(p_/2.))
