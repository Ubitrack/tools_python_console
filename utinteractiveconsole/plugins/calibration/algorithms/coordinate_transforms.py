from __future__ import division
from math import sin, cos, acos, atan2, sqrt
import numpy as np


def rad_norm(angle):
    if angle > np.pi:
        angle -= 2*np.pi
    elif angle <= -np.pi:
        angle += 2*np.pi
    return angle


def cartesian_to_spherical(vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    r = np.linalg.norm(vector)

    # The polar angle.
    phi = np.arctan2(np.sqrt(vector[0]**2 + vector[1]**2), vector[2])

    # The azimuth.
    theta = np.arctan2(vector[1], vector[0])

    # Return the spherical coordinate vector.
    return np.array([r, theta, phi], np.float64)


def cartesian_to_latlon(vector):
    """Convert the Cartesian vector [x, y, z] to lattitude, longitude [lat, lon].


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [lat, lon].
    @rtype:         numpy rank-1, 2D array
    """
    r = np.linalg.norm(vector)

    lat = np.arcsin(vector[2]/r)
    lon = np.arctan2(vector[1], vector[0])

    # Return the spherical coordinate vector.
    return np.array([lon, lat], np.float64)


def spherical_to_cartesian(spherical_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """

    # Trig alias.
    sin_theta = sin(spherical_vect[1])
    sin_phi = sin(spherical_vect[2])

    # The vector.
    return np.array([spherical_vect[0] * cos(spherical_vect[1]) * sin_phi,
                  spherical_vect[0] * sin_phi * sin_theta,
                  spherical_vect[0] * cos(spherical_vect[2]),], np.float64)