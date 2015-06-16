#!/usr/bin/env python3
import numpy as np

def SphPosToCart(vector, radians=False):
    """Convert a spherical position vector into Cartesian position.

    Arguments:
        vector -- A 3-element NumPy array representing, in order,
                  the spherical r, theta, and phi position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    representative Cartesian x, y, and z coordinates of the input vector.
    """
    if len(vector) != 3:
        print("WARNING - SphPosToCart(): Not a 3-dimensional vector!")
    r, theta, phi = vector
    if not radians:
        theta = np.radians(theta % 360)
        phi = np.radians(phi % 360)
    return np.array([r * np.sin(theta) * np.cos(phi),
                     r * np.sin(theta) * np.sin(phi),
                     r * np.cos(theta)])

def CartPosToSph(vector):
    """Convert a Cartesian position vector into spherical coordinate space.

    Arguments:
        vector -- A 3-element NumPy array representing, in order,
                  Cartesian x, y, and z position coordinates.

    Returns a 3-element NumPy array representing, in order, the
    spherical r, theta, and phi position coordinates of the input vector.
    """
    if len(vector) != 3:
        print("WARNING: Not a 3-dimensional vector!")
    x, y, z = vector[:3]

    return np.array([np.sqrt(x**2+y**2+z**2),
                     np.degrees(np.arctan2(np.sqrt(x**2+y**2), z)),
                     np.degrees(np.arctan2(y, x))])

def SphVecToCart(position, vector, radians=False):
    """Convert a spherical vector into Cartesian vector space.

    Takes a spherical-space vector and its corresponding spherical-
    space position and returns the magnitude of the vector in x, y,
    and z Cartesian directions.

    Arguments:
        position -- A 3-element NumPy array, representing the position
                    of the vector in spherical space.
        vector -- A 3-element NumPy array, representing the vector to
                  be converted, also in spherical space.

    Returns a 3-element NumPy array representing the magnitude of
    the input vector in Cartesian vector space.
    """
    if len(position) != 3:
        print("WARNING - SphVecToCart(): Position not a 3-dimensional vector!")
    if len(vector) != 3:
        print("WARNING - SphVecToCart(): Vector not a 3-dimensional vector!")

    r, theta, phi = position
    if not radians:
        theta = np.radians(theta % 360)
        phi = np.radians(phi % 360)

    r_hat = np.array([np.sin(theta) * np.cos(phi), #x_hat
                      np.sin(theta) * np.sin(phi), #y_hat
                      np.cos(theta)])              #z_hat

    theta_hat = np.array([np.cos(theta) * np.cos(phi), #x_hat
                          np.cos(theta) * np.sin(phi), #y_hat
                          -np.sin(theta)])             #z_hat

    phi_hat = np.array([-np.sin(phi),   #x_hat
                          np.cos(phi),  #y_hat
                                    0]) #z_hat
    transform_matrix = np.array([r_hat, theta_hat, phi_hat])
    return vector.dot(transform_matrix)

def RotX(vector, angle):
    """Rotate a Cartesian vector by a given angle about the +x axis.

    This function rotates a given vector about the Cartesian +x axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +y axis toward the +z axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Returns a 3-element NumPy array representing the rotated vector.
    """
    angle = np.radians(angle)
    R_X = np.array([[1,             0,              0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle),  np.cos(angle)]])
    return R_X.dot(vector)

def RotY(vector, angle):
    """Rotate a Cartesian vector by a given angle about the +y axis.

    This function rotates a given vector about the Cartesian +y axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +z axis toward the +x axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Returns a 3-element NumPy array representing the rotated vector.
    """
    angle = np.radians(angle)
    R_Y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                    [             0, 1,             0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    return R_Y.dot(vector)

def RotZ(vector, angle):
    """Rotate a Cartesian vector by a given angle about the +z axis.

    This function rotates a given vector about the Cartesian +z axis.
    The rotation is in a right-handed sense; positive angles rotate
    from the +x axis toward the +y axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Returns a 3-element NumPy array representing the rotated vector.
    """
    angle = np.radians(angle)
    R_Z = np.array([[ np.cos(angle), -np.sin(angle), 0],
                    [ np.sin(angle),  np.cos(angle), 0],
                    [             0,              0, 1]])
    return R_Z.dot(vector)

def Rot(vector, x=0., y=0., z=0., radians=True):
    """Rotate a Cartesian vector.

    This function rotates a given vector about the Cartesian axes.
    The rotation is in a right-handed sense; positive angles rotate
    from the +x axis toward the +y axis.

    Arguments:
        vector -- A 3-element NumPy array to be rotated.
        angle -- The angle by which the input vector will be rotated.

    Returns a 3-element NumPy array representing the rotated vector.
    """
    R_X = np.array([[1,             0,              0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle),  np.cos(angle)]])
    R_Y = np.array([[ np.cos(angle), 0, np.sin(angle)],
                    [             0, 1,             0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    R_Z = np.array([[ np.cos(angle), -np.sin(angle), 0],
                    [ np.sin(angle),  np.cos(angle), 0],
                    [             0,              0, 1]])
    return vector

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
