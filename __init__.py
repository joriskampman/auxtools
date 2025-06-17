"""
auxtools module

this module contains all kinds of helpfull scripts that are stripped of any relations with projects
or other thales-based links
"""

# import files
# pylint: disable=too-many-lines
# pylint: disable=wrong-import-order
# pylint: disable=useless-return
# basics
# from typing import assert_never
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import tkinter as tk
from tkinter import filedialog
import time

# more exotic ones
import re
import datetime as dtm
from copy import deepcopy
import numexpr as ne
import glob
import inspect
from itertools import cycle

# sub-modules for scipyt
from scipy.interpolate import interp1d
from scipy.special import factorial
import scipy.signal.windows as spsw
from scipy.fftpack import fftshift
from scipy.signal import find_peaks, convolve2d

# matplotlib sub-modules
from matplotlib.legend import Legend
from matplotlib.colors import to_rgb, to_rgba
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Circle, Rectangle, Wedge, CirclePolygon, Ellipse, \
                               FancyArrow, RegularPolygon
import matplotlib.transforms as mtrans
import matplotlib.dates as mdates

# used sporadically
# pylint: disable-next=E0611
from skimage.color import rgb2gray
import pdb  # noqa
import sys # noqa

# import all subfunctions
from .cmaps import * # noqa

# constants
LIGHTSPEED_RADAR = 299706720.0
R_EARTH = 6371.0088e3
T0 = 290

# use_levels:
# 0: use always
# 1: use for most cases (kg, meters,...)
# 2: specific cases (hectopascal comes to mind)
# 3: don't know any case for now
si_prefixes = {-30: dict(sf=None, name="one quintillionth", prefix="quecto", sym='q', use_level=0),
               -27: dict(sf=None, name="one quadrilliardth", prefix="ronto", sym='r', use_level=0),
               -24: dict(sf=None, name="one quadrillionth", prefix="yocto", sym='y', use_level=0),
               -21: dict(sf=None, name="one trilliardth", prefix="zepto", sym='z', use_level=0),
               -18: dict(sf=None, name="one trillionth", prefix="atto", sym='a', use_level=0),
               -15: dict(sf=None, name="one billiardth", prefix="femto", sym='f', use_level=0),
               -12: dict(sf=None, name="one billionth", prefix="pico", sym='p', use_level=0),
               -9: dict(sf=None, name="one milliardth", prefix="nano", sym='n', use_level=0),
               -6: dict(sf=None, name="one millionth", prefix="micro", sym='u', use_level=0),
               -3: dict(sf=None, name="one thousandth", prefix="milli", sym='m', use_level=0),
               -2: dict(sf=None, name="one hundreth", prefix="centi", sym='c', use_level=1),
               -1: dict(sf=None, name="one tenth", prefix="deci", sym='d', use_level=1),
               0: dict(sf=None, name="one", prefix="", sym='', use_level=0),
               +1: dict(sf=None, name="ten", prefix="deca", sym='da', use_level=3),
               +2: dict(sf=None, name="hundred", prefix="hecto", sym='h', use_level=2),
               +3: dict(sf=None, name="thousand", prefix="kilo", sym='k', use_level=0),
               +6: dict(sf=None, name="million", prefix="mega", sym='M', use_level=0),
               +9: dict(sf=None, name="milliard", prefix="giga", sym='G', use_level=0),
               +12: dict(sf=None, name="billion", prefix="tera", sym='T', use_level=0),
               +15: dict(sf=None, name="billiard", prefix="peta", sym='P', use_level=0),
               +18: dict(sf=None, name="trillion", prefix="exa", sym='E', use_level=0),
               +21: dict(sf=None, name="trilliard", prefix="zetta", sym='Z', use_level=0),
               +24: dict(sf=None, name="quadrillion", prefix="yotta", sym='Y', use_level=0),
               +27: dict(sf=None, name="quadrilliard", prefix="ronna", sym='R', use_level=0),
               +30: dict(sf=None, name="quintillion", prefix="quetta", sym='Q', use_level=0)}

confidence_table = pd.Series(
    index=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95,
           0.975, 0.99, 0.995],
    data=[0.010, 0.020, 0.051, 0.103, 0.211, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
          np.nan, np.nan, np.nan, 4.605, 5.991, 7.378, 9.210, 10.597])

# interpolate the nan's
confidence_table.interpolate(method='cubic', inplace=True)


# CLASSES
class IncorrectNumberOfFilesFoundError(Exception):
  """ Incorrect number of files is found """


class NotInvertibleDictError(Exception):
  """ The dictionary to invert is not invertible (non-unique values most likely) """


class DimensionError(Exception):
  """ and exception for when some reshaping cannot work due to incorrect dimensions """


def multiply_quaternions(*quats):
  """ multiply many quaternions

  """
  # pick the first one as a starting point
  q1 = (1., 0., 0., 0.)
  w1 = q1[0]
  v1 = np.asarray(q1[1:])

  for quat in quats:
    if isinstance(quat, dict):
      quat = quaternion_dict_to_array(quat)

    w2 = quat[0]
    v2 = np.asarray(quat[1:])

    w_mult = w1*w2 - v1.dot(v2)
    v_mult = w1*v2 + w2*v1 + np.cross(v1, v2)

    w1 = w_mult
    v1 = v_mult

  return (w1, *v1)


def convert_normal_to_quaternion(normal, as_dict=False):
  """ convert a rotation given by a normal to a quaternion

  arguments:
  ----------
  normal : 3-array-like of floats

  returns:
  --------
  quat: 4-tuple of floats
        The quaternion according to the format q=(w, x, y, z)
  """
  normal = np.asarray(normal)
  zvec = np.array([0., 0., 1.])
  if np.all(np.isclose(zvec, normal)):
    quat = np.asarray([1., 0., 0., 0.])
  elif (np.all(np.isclose(zvec, -1*normal))):
    quat = np.asarray([0., 1., 0., 0.])
  else:
    rotvec = np.cross(zvec, normal)
    rotvec_n = rotvec/np.linalg.norm(rotvec)

    theta = np.arccos(normal.dot(zvec.T)).item()
    # build quaternion
    quat_final_part = (rotvec_n*np.sin(theta/2)).ravel().tolist()
    quat = [np.cos(theta/2)] + quat_final_part


  if as_dict:
    qdict = quaternion_as_dict(quat)
    return qdict

  return np.asarray(quat)


def transform_points_via_pose(points, pose):
  """ transform point(s) via a pose


  """
  # some shaping of points
  points = np.asarray(points)
  # if 1-dim vector, add a dimension
  if points.ndim == 1:
    points = points.reshape(3, 1)

  # must be shaped to 3xN
  if points.shape[0] != 3:
    points = points.T

  # convert to pos(ition) and quat(ernion) arrays first
  pos, quat = pose_dict_to_arrays(pose)

  # ----------------- translation first -----------------------------------------
  points_trans = points + pos.reshape(3, 1)

  # ----------------- rotate ---------------------------------------------------
  # use quaternion multiplication
  points_trans_rot = rotate_via_quaternion(points_trans, quat)

  return points_trans_rot


def combine_poses(*poses):
  """ combine poses in succession

  """
  p1, q1 = pose_dict_to_arrays(poses[0])
  for pose in poses[1:]:

    p2, q2 = pose_dict_to_arrays(pose)

    q3 = multiply_quaternions(q2, q1)  # note the order.. very counter-intuitive!

    # first rotate all back, and then translate
    # TODO: find out why this is differently ordered that what chatgpt/deepseek mentioned!!!
    R2 = convert_quaternion_to_rotation_matrix(q2)

    p3 = p2 + R2.dot(p1)

    # prepare next iteration
    p1 = p3
    q1 = q3

  pose = build_pose_dict(p3, q3)

  return pose


def quaternion_as_dict(qvec, scalar_first=True):
  """ pack a quaternion vector in a dictionary

  arguments:
  ----------
  quat : 4-array-like of floats
         contains the 4 elements of a quaternion.
  scalar_first: bool, default=True
                whether the first (True) or last (False) element of the qvec is the scalar

  returns:
  --------
  qdict : dict
          contains the quaternion formatted as a dictionary
  """
  qkeys = ('w', 'x', 'y', 'z')
  if not scalar_first:
    qkeys = ('x', 'y', 'z', 'w')

  # build the dictionary
  qdict = dict(zip(qkeys, qvec))

  return qdict


def print_pose_dict(pose):
  """ print a pose dictionary

  """
  print("position:")
  print(f"  x: {pose['position']['x']:0.2f}")
  print(f"  y: {pose['position']['y']:0.2f}")
  print(f"  z: {pose['position']['z']:0.2f}")

  rotax, rotang = convert_quaternion_to_axis_angle(pose['orientation'])

  print("orientation")
  print("  axis:")
  print(f"    x: {rotax[0]:0.2f}")
  print(f"    y: {rotax[1]:0.2f}")
  print(f"    z: {rotax[2]:0.2f}")
  print(f"  angle: {np.rad2deg(rotang):0.2f} [deg]")


def convert_axis_angle_to_quaternion(rotax, ang, angle_units='rad', as_dict=False):
  """ convert a rotation via axis and angle pair to a quaternion

  arguments:
  ----------
  rotax : 3-array-like of floats
          The axis of rotation. Normalization is done in this function
  ang : float
        The angle of rotation

  returns:
  --------
  quat : 4-array of floats
         normalized quaternion
  """
  # convert degrees to radians
  if angle_units.startswith('deg'):
    ang = np.deg2rad(ang)

  # normalize the axis
  rotax = np.asarray(rotax).astype(float)
  rotax /= np.linalg.norm(rotax)

  w = np.cos(ang/2)
  sf = np.sin(ang/2)  # scale vector for x, y and z components
  x, y, z = sf*rotax

  quat = np.array([w, x, y, z])

  if as_dict:
    qdict = quaternion_as_dict(quat)
    return qdict

  return quat


def quaternion_dict_to_array(qdict):
  """ convert a quaternion dict to an array

  """
  if isinstance(qdict, dict):
    return np.asarray([qdict['w'], qdict['x'], qdict['y'], qdict['z']])
  return qdict


def position_dict_to_array(pdict):
  """ convert a position dict to an array

  """
  if isinstance(pdict, dict):
    return np.asarray([pdict['x'], pdict['y'], pdict['z']])
  return pdict


def pose_dict_to_arrays(pose_dict):
  """ conver the pose dictionary to 2 arrays


  """
  position = position_dict_to_array(pose_dict['position'])
  quat = quaternion_dict_to_array(pose_dict['orientation'])

  return position, quat


def convert_quaternion_to_axis_angle(quat):
  """ convert a quaternion to a normal vector"""
  if isinstance(quat, dict):
    # it is a pose dict
    quat = quaternion_dict_to_array(quat)

  w, x, y, z = quat
  vec = np.asarray([x, y, z])
  nvec = np.linalg.norm(vec)

  rotvec = vec/nvec
  theta = 2*np.arctan2(nvec, w)

  return rotvec, theta


def invert_quaternion(quat):
  """ calculate the inverse of a quaternion

  q_inv = (w, -x, -y, -z)
  """
  as_dict = isinstance(quat, dict)

  quat = quaternion_dict_to_array(quat)

  iquat = np.asarray((quat[0], -quat[1], -quat[2], -quat[3]))
  if as_dict:
    iquat = quaternion_as_dict(iquat)

  return iquat


def invert_pose(pose):
  """ invert a pose

  """
  pos, quat = pose_dict_to_arrays(pose)

  # invert the quaternion
  invquat = invert_quaternion(quat)

  invR = convert_quaternion_to_rotation_matrix(invquat)
  invpos = -1*invR.dot(pos)

  invpose = build_pose_dict(invpos, invquat)

  return invpose


def convert_quaternion_to_rotation_matrix(quat):
  """ convert a quaternion to a rotation matrix

  arguments:
  ----------
  quat : [4-array-like | dict ]
         The quaternion numbers as a pose dict or a 4 array. In case of an array, it is assumed
         to be 'scalar-first'

  returns:
  --------
  rotmat : 3x3 array of floats
           the rotation matrix belonging to the rotation as defined by the given quaternion
  """
  if isinstance(quat, dict):
    w = quat['w']
    x = quat['x']
    y = quat['y']
    z = quat['z']
  else:
    w, x, y, z = quat

  r11 = 1 - 2*(y**2 + z**2)
  r12 = 2*(x*y - w*z)
  r13 = 2*(x*z + w*y)

  r21 = 2*(x*y + w*z)
  r22 = 1 - 2*(x**2 + z**2)
  r23 = 2*(y*z - w*x)

  r31 = 2*(x*z - w*y)
  r32 = 2*(y*z + w*x)
  r33 = 1 - 2*(x**2 + y**2)

  rotmat = np.array([[r11, r12, r13],
                     [r21, r22, r23],
                     [r31, r32, r33]])

  return rotmat


def rotate_via_quaternion(points, quat):
  """ rotate an array of points via a quaternion
  """
  points = np.asarray(points)
  if points.ndim == 1:
    points = points.reshape(3, 1)

  if points.shape[0] != 3:
    points = points.T

  # convert quaternion to rotation matrix
  rotmat = convert_quaternion_to_rotation_matrix(quat)

  rpoints = rotmat@points

  return rpoints


def rotate_point_via_quaternion(point, quat, rtype='passive'):
  """ rotate a point via a quaternion

  'passive' type means the point is decribed in a different CS
  'active' type means the point is rotated

  they are each other's inverse
  """
  # make point to be a pseudo-quaternion
  qpoint = np.asarray([0., *point])

  print(qpoint)
  if rtype == 'passive':
    qpoint_rot = multiply_quaternions(quat, qpoint, invert_quaternion(quat))

  elif rtype == 'active':
    qpoint_rot = multiply_quaternions(invert_quaternion(quat), qpoint, quat)
  else:
    raise ValueError(f"The rtype must be either 'active' or 'passive'. '{rtype}' is not allowed")

  return qpoint_rot[1:]


def build_pose_dict(position, orientation):
  """ build a pose dictionary

  arguments:
  ----------
  position : 3-array-like of floats
              the position in (x, y, z)
  normal: 3-array of floats
          The normal vector of the plane of the pose

  returns:
  --------
  pose : dict
          resulting pose dictionary. Keys 'position' and orientation'. sub-keys are:
          - position: x, y and z
          - orientation: w, x, y and z
  """
  posekeys = ['x', 'y', 'z']

  pose_pos = dict(zip(posekeys, position))

  if isinstance(orientation, dict):
    qvec = np.array([orientation['w'], orientation['x'], orientation['y'], orientation['z']])
  elif isinstance(orientation, (list, tuple, np.ndarray)):
    if len(orientation) == 2:  # axis-angle
      qvec = convert_axis_angle_to_quaternion(*orientation)
    elif len(orientation) == 3:  # it is a normal to a plane
      qvec = convert_normal_to_quaternion(orientation)
    else:
      qvec = np.asarray(orientation)
  else:
    raise TypeError(f"The type of the orientation argument ({type(orientation)}) is not known")

  pose = {'position': pose_pos,
          'orientation': quaternion_as_dict(qvec)}  # pylint: disable=E0606

  return pose


def dev_from_vector(vecs, refvec, angle_units='rad'):
  """ calculate the angular deviation from a reference vector

  """
  # check 1: sort by verticalness
  vecs = np.asarray(vecs)
  if vecs.ndim == 1:
    vecs = np.reshape(vecs, (1, -1))
  else:
    shp = vecs.shape
    if shp[0] == 3:
      if shp[1] != 3:
        vecs = vecs.T

  # transpose for correct dimensions
  vvec = vecs.T
  refvec = np.asarray(refvec).reshape(1, -1)
  refvec_2d = refvec.reshape(1, -1)

  dev_from_ref = np.arccos(refvec_2d.dot(vvec)).ravel()

  if angle_units == 'deg':
    dev_from_ref = np.rad2deg(dev_from_ref)

  return dev_from_ref


def dev_from_vertical(vecs, angle_units='rad'):
  """ calculate the deviation from vertical for a certain vector
  """
  dvert_pos = dev_from_vector(vecs, [0., 0., 1.])
  dvert_neg = dev_from_vector(vecs, [0., 0., -1.])

  dvert = np.fmin(dvert_pos, dvert_neg)

  if angle_units == 'deg':
    dvert = np.rad2deg(dvert)

  return dvert


def dist_to_plane(points, planecoefs):
  """ Calculate the distance from (a) point(s) to a plane

  The plane coefficients are in the form ax + by + cz + d = 0; thus a 4-array-like

  arguments:
  ----------
  points : array-like of floats
           An array(-like) of point(s). At least 1 dimension must be 3 long. In case of a 3x3 array
           the points are assumed to be stacked vertically..
  planecoefs : 4-array-like of floats
               The plane coefficients given as a 4-element array-like.

  returns:
  --------
  dists : array-like of floats
          The distance as a float
  """
  points = np.asarray(points)
  if points.ndim == 1:
    points.reshape(1, -1)
  else:
    shp = points.shape
    if shp[0] == 3:
      if shp[1] != 3:
        # transpose if the coefficients are the vertical direction
        points = points.T

  # append 1. to the end (for the d-coefficient)
  points_ext = np.hstack((points, np.ones((points.shape[0], 1), dtype=float)))

  # make intok a 4x1 vector
  planecoefs = np.asarray(planecoefs).reshape(4, 1)

  dists = np.abs(points_ext@planecoefs).ravel()

  return dists


def colorvec(color):
  """ create a color vector

  """
  if isinstance(color, str):
    if len(color) == 1:
      cvec = mcolors.BASE_COLORS[color]
    else:
      if color.startswith('tab:'):
        chex = mcolors.TABLEAU_COLORS[color]
      elif color.startswith('xkcd:'):
        chex = mcolors.XKCD_COLORS[color]
      else:
        chex = mcolors.CSS4_COLORS[color]
      cvec = np.array([int(chex[1:3], 16), int(chex[3:5], 16), int(chex[5:7], 16)])/255
  else:
    cvec = color

  return np.asarray(cvec)


def _set_file_or_dir(fodstr, not_exist_response, file_or_dir):
  """ superfunction which is overloaded by setfile and setdir

  options for 'not_exist_response' are:
  - error/exception  (default)
  - warning
  - nothing/silence
  """
  if not os.path.exists(fodstr):
    if not_exist_response.lower().startswith('ex'):
      raise FileNotFoundError(f"The {file_or_dir} '{fodstr}' does not exist")
    elif not_exist_response.lower().startswith('warn'):
      warnings.warn(f"The {file_or_dir} '{fodstr}' does not exist")
    elif not_exist_response.lower().startswith('nothing'):
      pass  # do nothing

  # add filesep if not present
  if file_or_dir == 'directory':
    if not fodstr.endswith(os.path.sep):
      fodstr += os.path.sep

  return fodstr


def setdir(dirstr, not_exist_response='exception'):
  """
  overloaded function of _set_file_or_dir() for directories

  options for 'not_exist_response' are:
  - error/exception  (default)
  - warning
  - nothing/silence
  """
  return _set_file_or_dir(dirstr, not_exist_response, 'directory')


def setfile(filestr, not_exist_response='exception'):
  """
  overloaded function of _set_file_or_dir() for directories

  options for 'not_exist_response' are:
  - error/exception  (default)
  - warning
  - nothing/silence
  """
  return _set_file_or_dir(filestr, not_exist_response, 'file')


def get_background_separation_value(imint, nof_bins='doane'):
  """estimate the value below which can be considered background. Uses the histogram

  This function uses 'scale_for_objects' and 'numpy.histogram_bin_edges' functions

  arguments:
  ----------
  imint : ndarray
          2D intensity image for which the background must be estimated
  nof_bins : [ float | str], default='doane'
             The number of bins for the histogram to calculate. In case of a string, the list as
             given by numpy.histogram_bin_edges() is valid

  Returns:
  --------
  float: the value which separates the background optimally from the foreground
  """
  vsep, _ = scale_for_objects(imint, percentile=99., nof_bins=nof_bins)  # percentile is dont care

  return vsep


def scale_for_objects(imint, percentile, nof_bins='doane'):
  """ scale for objects

  scale a vmin and vmax for the objects in an image

  arguments:
  ----------
  imint : ndarray of ints
          An image as an integer NxM array
  percentile : float
               The percentile between which the valid samples lie
  nof_bins : [str | int], default='doane'
             The number of bins for the histogram. See np.histogram for all the options

  returns:
  --------
  vmin : float
         The minimum value for value scaling
  vmax : float
         The maximum value for value scaling
  """
  binedges = np.histogram_bin_edges(imint, bins=nof_bins)
  bincenters = binedges[:-1] + np.diff(binedges)/2
  hcounts, _ = np.histogram(imint.ravel(), bins=binedges)
  imax = np.argmax(hcounts)
  idips = find_peaks(-hcounts)[0]  # note the minus sign!
  iidip = np.argwhere(idips - imax > 0).item(0)
  vmin = bincenters[idips[iidip]]
  vmax = np.percentile(imint, percentile)  # intensity cut-off

  return (vmin, vmax)


def rgb2hsv(rgbs, makeplot=False, split=True):
  """Convert an RGB image to a Hue, Saturation, Value (HSV) image

  Args:
      rgbs (3D ndarray): The 3D RGB image
      makeplot (bool, optional): if True, some debugging plots are generated. Defaults to False.
      split (bool, optional): if True, the h, s and v values are returned separately.
                              Defaults to True.

  Returns:
      if 'split' is True:
        (ndarray, ndarray, ndarray): the h, s, v values in separate arrays
      else
        (3D ndarray): the hsv image in a 3D ndarray
  """

  # check if range is correct
  if rgbs.max() > 1.:
    warnings.warn("The values are exceeding 1. Scaled back!", category=UserWarning)
    rgbs /= rgbs.max()

  # auxiliaries
  rs = rgbs[..., 0]
  gs = rgbs[..., 1]
  bs = rgbs[..., 2]

  cmaxs = np.max(rgbs, axis=-1)
  cmins = np.min(rgbs, axis=-1)
  deltas = cmaxs - cmins

  # initialize to NaN
  hues = np.nan*np.ones_like(cmaxs, dtype=float)
  sats = np.nan*np.ones_like(cmaxs, dtype=float)
  vals = np.nan*np.ones_like(cmaxs, dtype=float)

  # HUE
  hues[np.isclose(deltas, 0.)] = np.nan
  tf_max_is_red = np.isclose(cmaxs - rs, 0.)
  tf_max_is_green = np.isclose(cmaxs - gs, 0.)
  tf_max_is_blue = np.isclose(cmaxs - bs, 0.)
  hues[tf_max_is_red] = ((gs - bs)/deltas)[tf_max_is_red]
  hues[tf_max_is_green] = ((bs - rs)/deltas)[tf_max_is_green] + 2
  hues[tf_max_is_blue] = ((rs - gs)/deltas)[tf_max_is_blue] + 4
  hues_in_deg = angled(np.exp(1j*np.deg2rad(hues*60)))
  hues_in_deg[hues_in_deg < 0.] = hues_in_deg[hues_in_deg < 0.] + 360

  hues = hues_in_deg/360

  # brightness/value
  vals = cmaxs.copy()

  # saturation
  sats = deltas/vals
  sats[np.isclose(vals, 0.)] = 0.

  if makeplot:
    _, axs = plt.subplots(2, 2, num=figname("RGB to HSV"))
    ax = axs[0, 0]
    ax.imshow(rgbs)
    ax.set_title("RGB")

    ax = axs[0, 1]
    ax.imshow(hues_in_deg)
    ax.set_title("Hue")
    ax = axs[1, 0]
    ax.imshow(sats)
    ax.set_title("Saturation")
    ax = axs[1, 1]
    ax.imshow(vals)
    ax.set_title("Value/Brightness")
    # add_colorbar()

  if split:
    return hues, sats, vals

  return np.stack((hues, sats, vals), axis=-1)


def hsv2rgb(hsvs, makeplot=False, split=False):
  """Convert a HSV image to a RGB image

  Args:
      hsvs (3D ndarray): The 3D HSV image
      makeplot (bool, optional): if True, some debugging plots are generated. Defaults to False.
      split (bool, optional): if True, the r, g and b values are returned separately.
                              Defaults to True.

  Returns:
      if 'split' is True:
        (ndarray, ndarray, ndarray): the h, s, v values in separate arrays
      else
        (3D ndarray): the hsv image in a 3D ndarray
  """

  # corner case: values exceed 1
  if hsvs.max() > 1.:
    warnings.warn("The values are exceeding 1. Scaled back!", category=UserWarning)
  # hsvs /= np.nanmax(hsvs)

  # auxiliaries
  hues = hsvs[..., 0]
  sats = hsvs[..., 1]
  vals = hsvs[..., 2]

  hues *= 6  # make the hues vary from 0 to 6
  chromas = sats*vals
  Xs = chromas*(1. - np.abs((hues%2) - 1.))

  rs1 = np.zeros_like(hues, dtype=float)
  gs1 = np.zeros_like(hues, dtype=float)
  bs1 = np.zeros_like(hues, dtype=float)

  tf = hues <= 6.
  rs1[tf] = chromas[tf]
  gs1[tf] = 0.
  bs1[tf] = Xs[tf]
  tf = hues <= 5.
  rs1[tf] = Xs[tf]
  gs1[tf] = 0.
  bs1[tf] = chromas[tf]
  tf = hues <= 4.
  rs1[tf] = 0.
  gs1[tf] = Xs[tf]
  bs1[tf] = chromas[tf]
  tf = hues <= 3.
  rs1[tf] = 0.
  gs1[tf] = chromas[tf]
  bs1[tf] = Xs[tf]
  tf = hues <= 2.
  rs1[tf] = Xs[tf]
  gs1[tf] = chromas[tf]
  bs1[tf] = 0.
  tf = hues <= 1.
  rs1[tf] = chromas[tf]
  gs1[tf] = Xs[tf]
  bs1[tf] = 0.

  # match lightness
  m = vals - chromas

  # final result
  rs = rs1 + m
  gs = gs1 + m
  bs = bs1 + m

  if makeplot:
    _, axs = plt.subplots(2, 2, num=figname("HSV to RGB"))
    ax = axs[0, 0]

    rgbs = np.stack((rs, gs, bs), axis=-1)
    ax.imshow(rgbs)
    ax.set_title("RGB")

    ax = axs[0, 1]
    ax.imshow(hues*360)
    ax.set_title("Hue")
    ax = axs[1, 0]
    ax.imshow(sats)
    ax.set_title("Saturation")
    ax = axs[1, 1]
    ax.imshow(vals)
    ax.set_title("Value/Brightness")

    plt.show(block=False)
    plt.draw()
    plt.pause(0.1)

  if split:
    return rs, gs, bs

  return np.stack((rs, gs, bs), axis=-1)


def brighten_rgb(rgbin, perc=99, makeplot=False, ax=None):
  """
  brighten an RGB image
  """
  hues, sats, vals = rgb2hsv(rgbin, makeplot=False, split=True)

  # modify the lightness
  satmax = np.percentile(sats, perc)
  valmax = np.percentile(vals, perc)

  satsc = sats/satmax
  valsc = vals/valmax
  satsc[satsc > 1.] = 1.
  valsc[valsc > 1.] = 1.

  hsvc = np.stack((hues, satsc, valsc), axis=-1)
  rgbout = hsv2rgb(hsvc, makeplot=False, split=False)

  if makeplot:
    if ax is None:
      _, ax = plt.subplots(1, 1, num=figname("brightened RGB image"))
      ax.imshow(rgbout, aspect='equal')

  return rgbout


def check_sockets(iprange, port=5025, timeout=1., sufrange=np.r_[2:255]):
    """
    check all sockets in a certain IP range

    arguments:
    ----------
    iprange : str
              an IP-range in the form of a string, such as '192.168.1'
    port : int, default=5025
           port number, default is the TCP/IP port of 5025
    timeout : float, default=1.
              timeout in seconds to wait for a connection to be created
    sufrange : array-like, default=np.r_[2:255]
               an array-like containing the suffix range to check. the full range is the default;
               from 2 upto and including 254. Note that 1 and 255 are not valid component addresses

    returns:
    --------
    valid_addresses_list : list
                           a list containing the addresses with which connection could be
                           established
    """
    # pylint: disable=C0415
    import pyvisa
    import socket

    ipparts = iprange.split('.')
    ipprefix = '.'.join(ipparts[:3])

    # open socket
    rm = pyvisa.ResourceManager()
    valid_addresses_list = []
    for ipsuf in sufrange:
        ip2chk = ipprefix + f'.{ipsuf}'.format(ipsuf)
        sock_ = socket.socket()
        sock_.settimeout(timeout)
        conncode = sock_.connect_ex((ip2chk, port))
        if conncode == 0:
            print(f" - {ip2chk} --> connected", end='')
            valid_addresses_list.append(ip2chk)
            dev = rm.open_resource(f'TCPIP0::{ip2chk}::{port}::SOCKET')
            dev.read_termination = '\n'
            dev.write_termination = '\n'
            try:
              idn = dev.query("*idn?")
              print(f" ({idn})")
            except pyvisa.VisaIOError as err:
              print(f" (no identification because of VisaIOError: '{err.description}')")
        else:
            print(f" - {ip2chk} -- FAIL")

    return valid_addresses_list


def print_fraction(floatval, dotreplacement, ndec=1):
  """
  print a fraction without using a dot. For instance: 3.3V -> 3V3
  """
  first_int = int(floatval)
  frac = floatval - first_int
  second_int = int(0.5 + np.power(10, ndec)*frac)

  output = f"{first_int}{dotreplacement}{second_int}"

  return output


def popup(message, title="Next up", add_title_to_message=True, silence=False, yesno=False):
  """
  display a pop-up
  """
  root = tk.Tk()
  root.iconify()
  if add_title_to_message:
    message = f"{title}: {message}"

  answer = None
  if not silence:
    if yesno:
      answer = tk.messagebox.askyesno(title, message)
    else:
      answer = tk.messagebox.showinfo(title=title, message=message, parent=root)
  else:
    if yesno:
      answer = True

  root.destroy()

  return answer


def isnumber(strs):
  """ check if strings represent a number. Floating point number or other representations count! """
  allowed_chars = ['.', 'e', '-', '+', 'E']
  is_scalar = np.isscalar(strs)

  strs = listify(strs)
  tfs = []
  for str_ in strs:
    # remove all other objects
    cleanstr = remove_chars_from_str(str_, allowed_chars)
    tfs.append(cleanstr.isnumeric())

  ret = tfs
  if is_scalar:
    ret = ret[0]

  return ret


def invert_dict(mydict):
  """ invert the dict, keys <--> values """
  # check if the dict is invertible
  values = list(mydict.values())
  unq_values = np.unique(values)
  if len(values) > len(unq_values):
    raise NotInvertibleDictError("The dict given is not invertible since double values occur")

  invdict = dict.fromkeys(mydict.values())
  for key, value in mydict.items():
    invdict[value] = key

  return invdict


def plot_interval_patch(xdata, ydata, axis=0, stat='minmax', ax=None, use_median=True,
                        **plotkwargs):
  """
  plot an interval patch
  """
  # check data
  if ydata.ndim != 2:
    raise ValueError("The 'ydata' argument must be a ndarray of 2 dimensions")

  if stat.endswith('minmax'):
    mindata = np.min(ydata, axis=axis)
    maxdata = np.max(ydata, axis=axis)

  elif stat.endswith('std'):
    nof_chars = len('std')
    if len(stat) == nof_chars:
      factor = 1.
    else:
      facstr = stat[:-(nof_chars)]
      if facstr.endswith('*'):
        facstr = facstr[:-1]

      factor = float(facstr)

    if use_median:
      meandata = np.median(ydata, axis=axis)
    else:
      meandata = np.mean(ydata, axis=axis)
    stddata = np.std(ydata, axis=axis)

    mindata = meandata - factor*stddata
    maxdata = meandata + factor*stddata
  elif stat.endswith('percentile'):
    nof_chars = len('percentile')
    if len(stat) == nof_chars:
      pct = 100.
    else:
      pct = float(stat[:-(nof_chars)])

    mindata = np.percentile(ydata, 100 - pct, axis=axis)
    maxdata = np.percentile(ydata, pct, axis=axis)

  else:
    raise ValueError(f"The 'stat' keyword argument value given ({stat}) is not valid!")

  # find the upper and lower lines
  verts = [*zip(xdata, maxdata), *zip(xdata[-1::-1], mindata[-1::-1])]
  poly = Polygon(verts, **plotkwargs)

  if ax is None:
    ax = plt.gca()
    plt.show(block=False)
    plt.draw()

  ax.add_patch(poly)
  plt.draw()

  return poly


def pick_from_interval(minval, maxval, nof_samples=1):
  """
  pick a sample from the interval defined by min and max
  """
  pick_rel = np.random.random_sample(nof_samples)

  drange = datarange([minval, maxval])
  minval = min([minval, maxval])
  pick = drange*pick_rel + minval

  if nof_samples == 1:
    pick = pick.item()

  return pick


def angled(cvals, axis=-1, unwrap=False, icenter=None, wrap_range=(-180, 180)):
  """
  return an angle in degrees

  arguments:
  ----------
  cvals : [ array-like | float | int ]
          A complex value or an array-like of complex values for which the angle in degrees must
          be calculated
  axis : int, default=-1
         The axis along which the phase must be unwrapped. Is moot when unwrap=False
  unwrap : bool, default=False
           Whether or not to unwrap the phase. Works with 'icenter' and 'axis'
  icenter : [ int | None], default=None
            if not None, the index given on the axis given by 'axis' is set to 0 degrees. the rest
            is all relative to this index

  Returns:
  --------
  angvals : [ array-like | float ]
            The angle (whether unwrapped or not) in degrees. Can be a single float or an
            array-like of floats
  """
  cvals = np.squeeze(cvals)
  if unwrap:
    angvals = np.rad2deg(np.unwrap(np.angle(cvals), axis=axis))
  else:
    angvals = np.angle(cvals, deg=True)
    # apply wrap_range
    if not np.isscalar(angvals):
        tf_below = angvals < wrap_range[0]
        tf_above = angvals > wrap_range[1]
        angvals[tf_below] = angvals[tf_below] + 360.
        angvals[tf_above] = angvals[tf_above] - 360.

  if icenter is not None:
    angvals_center = np.expand_dims(angvals.take(icenter, axis=axis), axis=axis)

    # do the correction
    angvals -= angvals_center

  return angvals


def timestamp(dt=None, short=False, fmt=None):
  """ get the timestamp """
  if dt is None:
    dt = dtm.datetime.now()
  # first is for sorting, maximually short
  if fmt is None:
    if short:
      tstr = dtm.datetime.strftime(dt, "%Y%m%d_%H%M%S")

    # slightly longer and more readable
    else:
      tstr = dtm.datetime.strftime(dt, "%H:%M:%S, %d %b %Y")
  else:
    tstr = dtm.datetime.strftime(dt, fmt)

  return tstr


def add_colorbar(cdata=None, fig=None, nof_ticks=None, axs=None, fmt="{:0.2f}", cbarlabel='',
                 **kwargs):
  """
  wrapper around the colorbar
  """

  if nof_ticks is None:
    cbticks = None
  else:
    # not implemented, but must find the full range (pmin to pmax) first and step make the steps
    # cbticks = np.linspace(pmin, pmax, 11)
    raise NotImplementedError("Providing the *nof_ticks* parameter is not implemented yet")

  if axs is None:
    if fig is None:
      fig = plt.gcf()
    axs = np.array(fig.axes)
  elif isinstance(axs, plt.Axes):
    axs = np.array([axs])

  if cdata is None:
    im = axs[0].get_images()[0]
  elif isinstance(cdata, plt.Axes):
    im = cdata.get_images()[0]
  else:
    im = cdata

  cb = plt.colorbar(im, ax=axs.ravel().tolist(), ticks=cbticks, **kwargs)

  # get the ticks back
  ticks = cb.get_ticks()
  # add the min/max to it
  cmin, cmax = im.get_clim()
  ticks_new = np.linspace(cmin, cmax, ticks.size+2)
  print(ticks_new)
  ticklabels = print_list(ticks_new, floatfmt=fmt).split(', ')
  cb.set_ticks(ticks_new, update_ticks=True)
  plt.draw()
  plt.pause(1e-4)
  cb.set_ticklabels(ticklabels, update_ticks=True)
  cb.ax.set_ylabel(cbarlabel, rotation=-90, va='bottom', fontsize=8, fontweight='normal')
  print(ticklabels)
  plt.draw()

  return cb


def remove_chars_from_str(string, skipped_chars=(' ', ',', '.', ':', ';', '/', '_', '-', '+')):
  """
  remove certain characters from a string
  """
  for char in skipped_chars:
    string = string.replace(char, '')

  return string


def find_between_indices(datavec, values, notfound=None):
  """
  find the indices between which the value is found
  """
  datavec = arrayify(datavec)

  # get the values
  isscalar = np.isscalar(values)
  if isscalar:
    values = listify(values)

  indices = []
  for val in values:
    indices_ = [notfound]*2
    # below
    ibelows = np.argwhere(datavec <= val)
    if ibelows.size > 0:
      indices_[0] = ibelows.item(-1)

    # above
    iaboves = np.argwhere(datavec >= val)
    if iaboves.size > 0:
      indices_[1] = iaboves.item(0)

    # append to indices
    indices.append(indices_)

  if isscalar:
    retval = indices[0]
  else:
    retval = indices

  return retval


def check_if_even(vals):
  """
  check if a float is even
  """
  vals = listify(vals)

  # check type --> assume all have the same type
  reslist = []
  for val in vals:
    if isinstance(val, (float, np.floating)):
      if val.is_integer():
        val = int(0.5 + val)
      else:
        reslist.append(False)
        continue

    # now the value is an integer
    if val%2 == 0:
      reslist.append(True)
    else:
      reslist.append(False)

  if len(reslist) == 1:
    ret = reslist[0]
  else:
    ret = reslist

  return ret


def check_if_odd(vals):
  """
  check if a number is odd
  """
  vals = listify(vals)

  # check type --> assume all have the same type
  reslist = []
  for val in vals:
    if isinstance(val, (float, np.floating)):
      if val.is_integer():
        val = int(0.5 + val)
      else:
        reslist.append(False)
        continue

    # now the value is an integer
    if val%2 == 0:
      reslist.append(False)
    else:
      reslist.append(True)

  if len(reslist) == 1:
    ret = reslist[0]
  else:
    ret = reslist

  return ret


def format_as_si(value, nsig=3, ndec=None, sep=" ", fmt='auto', max_use_level=0,
                 check_if_int=True, force_int=False):
  """
  print a value as an SI string value
  """
  val, dct = scale_by_si_prefix(value, max_use_level=max_use_level)

  if force_int:
    val = int(0.5 + val)

  if check_if_int:
    if isinstance(val, (np.floating, float)) and val.is_integer():
      val = int(val)
      fmt = "{:d}"

  if fmt.startswith('auto'):
    if isinstance(val, (np.floating, float)):
      n_before_dec_point = np.log10(val).astype(int)
      if ndec is not None:
        n_after_dec_point = ndec
      else:
        n_after_dec_point = nsig - 1 - n_before_dec_point

      fmt = f"{{:{n_before_dec_point}.{n_after_dec_point}f}}"
    elif isinstance(val, (np.integer, int)):
      fmt = "{:d}"

  fmt += sep
  fmt += "{:s}"

  string = fmt.format(val, dct['sym'])

  # remove possible trailing spaces
  string = string.strip()

  return string


def scale_by_si_prefix(values, base_pref_on_what='rms', max_use_level=0):
  """
  return the scaled values and return the prefix
  """
  return_scalar = False
  if np.isscalar(values):
    return_scalar = True

  # make an array
  values = arrayify(values)

  # check which value to analyze
  if base_pref_on_what.startswith("mean") or base_pref_on_what.startswith("av"):
    val2check = values.mean()
  elif base_pref_on_what.startswith("med"):
    val2check = np.median(values)
  elif base_pref_on_what.startswith("min"):
    val2check = values.min()
  elif base_pref_on_what.startswith("max"):
    val2check = values.max()
  elif base_pref_on_what.startswith("absmax"):
    val2check = np.abs(values).max()
  elif base_pref_on_what.startswith("rms"):
    val2check = rms(values)
  else:
    raise ValueError(f"The given value for *base_pref_on_what* ({base_pref_on_what}) is not valid")

  # split the dictionary si_prefixes

  # get all the actual scaling values (powers of 10)
  valid_powvals = []
  for powval, prefdict in si_prefixes.items():
    if prefdict['use_level'] <= max_use_level:
      valid_powvals.append(powval)
  valid_powvals = arrayify(valid_powvals)

  # extract the order
  pow2check = np.log10(np.abs(val2check))

  powdiffs = pow2check - arrayify(valid_powvals)
  powdiffs[powdiffs < 0] = np.inf
  ifnd = np.argmin(powdiffs)
  fnd_powval = valid_powvals[ifnd]

  # get the SI prefix dictionary
  si_prefix_dict = si_prefixes[fnd_powval]
  si_prefix_dict['key'] = fnd_powval
  si_prefix_dict['sf'] = np.power(10., fnd_powval)

  scale_factor = np.power(10., fnd_powval)
  scaled_values = values/scale_factor
  if return_scalar:
    scaled_values = scaled_values.item(0)

  # add to output_list
  output_list = [scaled_values, si_prefix_dict]

  return output_list


def dms2angle(dms):
  """
  convert the degree/minute/second format to degrees
  """
  if isinstance(dms, (tuple, list, np.ndarray)):
    # check if the first component is a string or another array-like -> it is a list
    if isinstance(dms[0], (list, tuple, np.ndarray, str)):
      nof_elms = len(dms)
    else:
      nof_elms = 1
      dms = [dms]
  else:
    raise TypeError(f"The type of 'dms' ('{type(dms)}') is not recognized")

  # loop all elements
  ang_list = []
  for dms_ in dms:
    dms_elm_list_ = [float(elm) for elm in dms_.split()] if isinstance(dms_, str) else dms_

    ang_val = dms_elm_list_[0] + dms_elm_list_[1]/60 + dms_elm_list_[2]/3600
    ang_list.append(ang_val)

  if nof_elms == 1:
    ang = ang_list[0]
  else:
    ang = ang_list

  return ang


def break_text(text, maxlen, glue=None, silence_warning=False, print_=False):
  """
  break a long text line into a list or a glued string (glue=\n)

  if glue is not None but a str or char, this str or char will be the glue
  """
  nof_too_long_parts = 0
  str_list = []
  nof_chars = len(text)
  istart = 0
  stop_after_this_iter = False
  while True:
    iend0 = istart + maxlen
    if iend0 <= nof_chars:
      iend = iend0
    else:
      iend = nof_chars
      stop_after_this_iter = True
    textpart = text[istart:iend]
    words = textpart.split(' ')
    if len(words) > 1:
      if not stop_after_this_iter:
        len_last_part = len(words[0])
        if len_last_part > 0:
          words = words[:-1]
      textpart_ok = ' '.join(words)
    else:
      nof_too_long_parts += 1
      word = text[istart:].split(' ')[0]
      textpart_ok = word
    str_list.append(textpart_ok.strip())

    # prepare for next iteration
    istart += len(textpart_ok) + 1
    if istart >= nof_chars:
      break

  if not silence_warning:
    if nof_too_long_parts > 0:
      warnings.warn("The broken text contains {nof_too_long_parts} parts that exceed "
                    + f"{nof_chars} characters", category=UserWarning)

  output = str_list
  if glue is not None:
    output = glue.join(str_list)

  if print_:
    print(output)
    return None
  return output


def multiplot(nof_subs, name=None, nof_sub_stacks=1, ratio=5, subs_loc='right',
              orientation='landscape', sharex=True, sharey=True):
  """
  create a figure containing of one main plot plus a set of subplots.

  The subplots can be located to either of the 4 sides of the main axes and can span 1 or multiple
  rows and or columns; depending on the relative location with respect to the main axes

  arguments:
  ----------
  nof_subs : int
             The number of subfigures to be created
  name : [None, str], default=None
         If not None, this is the figure name
  nof_sub_stacks : int, default=1
                   The number of subplot rows or columns (depending on *subs_loc*). The default of
                   1 means that it is a single row/column.
  ratio : [int | float], default=5
          The ratio between the size of the large central plot axes and the sub axes. A value of
          5 is very nice for only a few (5 or less) subplots
  subs_loc : ['bottom', 'top', 'left', 'right'], defalt='bottom'
             The location of the subplots stack relative to the main axes
  orientation : ['portrait' | 'landscape'], default='landscape
                The orientation of the figure. Should be changed only in consideration with
                *subs_loc*.
  sharex : bool, default=True
           If on zoom, the axes should be zoomed together in the horizontal direction
  sharey : bool, default=True
          same as *sharex* but for the verical diretion

  returns:
  --------
  fig : the figure object
  ax0 : the object of the main axes
  axs_sub : the array-like of the subplots in the stack
  """
  fig = plt.figure(num=figname(name))

  resize_figure(orientation=orientation)

  axs_sub = []
  nof_subs_per_stack = int(0.5 + nof_subs/nof_sub_stacks)
  if subs_loc in ('top', 'bottom'):
    gs = GridSpec(ratio+nof_sub_stacks, int(0.5 + nof_subs/nof_sub_stacks))
    # add the subplots to the figure
    if subs_loc == 'bottom':
      ax0 = fig.add_subplot(gs[:-nof_sub_stacks, :])
      if sharex:
        sharex = ax0
      else:
        sharex = None
      if sharey:
        sharey = ax0
      else:
        sharey = None
      for isub in range(nof_subs):
        i_in_stack = isub%nof_subs_per_stack
        istack = int(isub/nof_subs_per_stack)
        irow_this_stack = -(nof_sub_stacks - istack)
        ax = fig.add_subplot(gs[irow_this_stack, i_in_stack], sharex=sharex, sharey=sharey)
        axs_sub.append(ax)
    elif subs_loc == 'top':
      ax0 = fig.add_subplot(gs[nof_sub_stacks:, :])
      if sharex:
        sharex = ax0
      else:
        sharex = None
      if sharey:
        sharey = ax0
      else:
        sharey = None
      for isub in range(nof_subs):
        i_in_stack = isub%nof_subs_per_stack
        istack = int(isub/nof_subs_per_stack)
        irow_this_stack = istack
        ax = fig.add_subplot(gs[irow_this_stack, i_in_stack], sharex=sharex, sharey=sharey)
        axs_sub.append(ax)
  elif subs_loc in ('left', 'right'):
    gs = GridSpec(int(0.5 + nof_subs/nof_sub_stacks), ratio+nof_sub_stacks)
    if subs_loc == 'left':
      ax0 = fig.add_subplot(gs[:, nof_sub_stacks:])
      if sharex:
        sharex = ax0
      else:
        sharex = None
      if sharey:
        sharey = ax0
      else:
        sharey = None
      for isub in range(nof_subs):
        i_in_stack = isub%nof_subs_per_stack
        istack = int(isub/nof_subs_per_stack)
        icol_this_stack = istack
        ax = fig.add_subplot(gs[i_in_stack, icol_this_stack], sharex=sharex, sharey=sharey)
        axs_sub.append(ax)
    elif subs_loc == 'right':
      ax0 = fig.add_subplot(gs[:, :-nof_sub_stacks])
      if sharex:
        sharex = ax0
      else:
        sharex = None
      if sharey:
        sharey = ax0
      else:
        sharey = None
      for isub in range(nof_subs):
        i_in_stack = isub%nof_subs_per_stack
        istack = int(isub/nof_subs_per_stack)
        icol_this_stack = -(nof_sub_stacks - istack)
        ax = fig.add_subplot(gs[i_in_stack, icol_this_stack], sharex=sharex, sharey=sharey)
        axs_sub.append(ax)

  return fig, ax0, axs_sub


def interpret_linespec(fmt):
  """
  convert a linespec string (e.g., 'r.-') to a set of 3 keyword arguments in a dict
  """
  markers = ['.', 'o', 'x', '+', '^', 'v', 'd', 's']
  linestyles = ['-', '--', ':', '-.']
  colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'k']

  # order is [color][marker][linestyle].. all are optional
  output = dict()
  if len(fmt) == 0:
    return output

  if fmt[0] in colors:
    output['color'] = fmt[0]
    fmt = fmt[1:]

  if len(fmt) == 0:
    return output

  # check if marker is given
  if fmt[0] in markers:
    output['marker'] = fmt[0]
    fmt = fmt[1:]

  if len(fmt) == 0:
    return output

  if fmt in linestyles:
    output['linestyle'] = fmt

  return output


def add_figlegend(legdata=None, labels=None, fig=None, dy_inch=None, clearup=False,
                  clearup_pars=None, buffer_pix=(4, 8), nrows=1, ncols=None,
                  remove_axs_legends=True):
  """
  add a legend to a figure
  """
  if clearup is None:
    clearup = {}
  clearup_dict={'lw': 2,
                'marker': None,
                'markersize': 5,
                'alpha': 1.}
  if clearup_pars is not None:
    clearup_dict.update(clearup_pars)

  if np.isscalar(buffer_pix):
    buffer_pix = [buffer_pix]*2

  # get the figure instance
  if fig is None:
    fig = plt.gcf()

  dpi = fig.get_dpi()
  hinch = fig.get_size_inches()[1]

  # remove existing figure legend instance (normally there is 0 or 1, not more)
  legends = [kid for kid in fig.get_children() if isinstance(kid, Legend)]
  for legend in legends:
    legend.remove()

  # get the legend data
  if legdata is None:
    legdata = plt.gca()
  elif isinstance(legdata, np.ndarray):
    legdata = legdata.ravel().tolist()
  legdata = listify(legdata)

  # get the data based on legdata
  if isinstance(legdata[0], plt.Axes):
    legdata_list = []
    for legdata_ in legdata:
      kids = legdata_.get_children()
      for kid in kids:
        if isinstance(kid, (plt.Line2D, Polygon, Circle, Rectangle, Wedge, RegularPolygon,
                            FancyArrow, CirclePolygon, Ellipse)):
          if kid.axes is not None:
            legdata_list.append(kid)
        else:
          pass  # do nothing, probably some kid that is part of the axes or ticks or whatever
      # legdata_list += legdata_.get_lines()
    legdata = legdata_list

  if labels is None:
    labels = [legobj.get_label() for legobj in legdata]
  elif isinstance(labels, str):
    labels = [labels]

  # check for valid labels
  tf_labels = [False if lab.startswith('_') else True for lab in labels]
  labels = np.array(labels)[tf_labels].tolist()
  legdata = np.array(legdata)[tf_labels].tolist()

  # check if the legdata is a string
  for ileg, legdata_this in enumerate(legdata):
    if isinstance(legdata_this, str):
      legdict_this_line = interpret_linespec(legdata_this)
      legdata_obj = plt.Line2D([], [], **legdict_this_line)
      legdata[ileg] = legdata_obj

  # check if title is present
  axs = fig.get_axes()
  has_title = [len(ax.get_title()) > 0 for ax in fig.get_axes()]
  if np.any(has_title):
    iax_w_title = np.argwhere(has_title).ravel()[0]
    fontsize_pts = axs[iax_w_title].title.get_fontsize()
    fontsize_inch = fontsize_pts/72
    padsize_pix = plt.rcParams['axes.titlepad']
    padsize_inch = padsize_pix/dpi
    title_os_inch = fontsize_inch + padsize_inch + buffer_pix[0]/dpi
  else:
    title_os_inch = 0.

  if dy_inch is None:
    top_fig = fig.subplotpars.top
    top_inch = top_fig*hinch
    dy_inch = hinch - top_inch - title_os_inch

  # get offset and make a transform
  offset_sub = mtrans.ScaledTranslation(0., -1*dy_inch - buffer_pix[0]/dpi, fig.dpi_scale_trans)
  trans_sub = fig.transFigure + offset_sub

  # add the legend
  nof_legends = len(legdata)
  if nof_legends == 0:
    warnings.warn("There are no legends to create!")
    return None

  if ncols is None:
    ncols = int(0.5 + nof_legends/nrows)

  leg = fig.legend(legdata, labels, loc="upper center", bbox_to_anchor=[0.5, 1.],
                   bbox_transform=trans_sub, borderaxespad=0.2, ncol=ncols,
                   columnspacing=0.4, fontsize=8, edgecolor='k',
                   prop={'size': 7, 'weight': 'bold'})

  # modify the legdata and markers
  if clearup:
    for legobj in leg.legend_handles:
      if clearup_dict['marker'] is not None:
        legobj.set_marker(clearup_dict['marker'])
        print("something is not fully correct yet. Please check this when applicable")
      legobj.set_markersize(clearup_dict['markersize'])
      legobj.set_linewidth(clearup_dict['lw'])
      legobj.set_alpha(clearup_dict['alpha'])

  hleg_dis = leg.get_window_extent().height
  hleg_inch = hleg_dis/dpi

  # adjust top
  buffer_inch = buffer_pix[1]/dpi
  dy_inch += buffer_inch + hleg_inch  # add the legend size and buffer
  ytop_rel = 1 - (dy_inch + title_os_inch)/hinch  # take axes title size too

  fig.subplots_adjust(top=ytop_rel)

  # remove the existing legends
  if remove_axs_legends:
    # loop all axes
    for ax in fig.get_axes():
      # get the legend object
      legobj = ax.get_legend()
      # remove if it actually exists
      if legobj is not None:
        legobj.remove()
    plt.draw()

  # activate the settings
  plt.show(block=False)
  plt.draw()
  plt.pause(1e-3)

  return leg


def add_figtitles(texts, fig=None, xpos_rel=0.5, ypos_rel=1., fontsize_top=12, fontsize_sub=8,
                  fontweight_top='bold', fontweight_sub='bold', buffer_pix=4,
                  return_handles=False):
  """
  add a title to a figure
  """
  htxt_list = []
  if fig is None:
    fig = plt.gcf()

  texts = listify(texts)

  # get height
  _, hinch = fig.get_size_inches()
  dpi = fig.get_dpi()
  ppi = 72

  # =================================
  # TOP line
  # =================================
  # top buffer of 2 pixels
  os_inch = buffer_pix/dpi
  os_rel = os_inch/hinch
  ypos_rel = 1. - os_rel
  if isinstance(xpos_rel, plt.Axes):
    axlims = xpos_rel.get_position().extents
    xpos_rel = (axlims[0] + axlims[2])/2

  # add the top text and fix it to the figure!!
  htxt_ = fig.text(xpos_rel, ypos_rel, texts[0], ha='center', va='top', fontsize=fontsize_top,
                   fontweight=fontweight_top, transform=fig.transFigure)
  htxt_list.append(htxt_)

  # now the subs must be relative to the top line
  dy_inch = fontsize_top/ppi + buffer_pix/dpi
  for txt in texts[1:]:
    offset_sub = mtrans.ScaledTranslation(0., -dy_inch, fig.dpi_scale_trans)
    trans_sub = fig.transFigure + offset_sub

    htxt_ = fig.text(xpos_rel, ypos_rel, txt, ha='center', va='top', fontsize=fontsize_sub,
                     fontweight=fontweight_sub, transform=trans_sub)
    htxt_list.append(htxt_)

    # new delta y
    dy_inch += fontsize_sub/ppi + buffer_pix/dpi

  # adjust top
  ytop_rel = 1 - dy_inch/hinch

  # check if there is a title present
  axs = fig.get_axes()
  has_title = [len(ax.get_title()) > 0 for ax in fig.get_axes()]
  if np.any(has_title):
    iax_w_title = np.argwhere(has_title).ravel()[0]
    fontsize_pts = axs[iax_w_title].title.get_fontsize()
    fontsize_inch = fontsize_pts/72
    padsize_pix = plt.rcParams['axes.titlepad']
    padsize_inch = padsize_pix/dpi
    offset_inch = fontsize_inch + padsize_inch + buffer_pix/dpi
    ytop_rel -= offset_inch/hinch

  fig.subplots_adjust(top=ytop_rel)

  # activate the settings
  plt.show(block=False)
  plt.draw()

  output_list = dy_inch
  if return_handles:
    output_list = [dy_inch, htxt_list]

  return output_list


def get_screen_dims(units='inches'):
  """
  get the dimensions of the screen
  """
  root = tk.Tk()
  w_pix = root.winfo_screenwidth()
  h_pix = root.winfo_screenheight()

  # destroy the Tcl
  root.destroy()

  # output units switching
  if units.startswith('inch'):
    dpi = plt.rcParams['figure.dpi']
    w = w_pix/dpi
    h = h_pix/dpi
  elif units == 'pix' or units == 'dots':
    w = w_pix
    h = h_pix
  elif units.startswith('mm'):
    w = 25.4*w_pix/dpi
    h = 25.4*h_pix/dpi
  elif units.startswith('cm'):
    w = 2.54*w_pix/dpi
    h = 2.54*h_pix/dpi

  else:
    raise ValueError("The units value given ({units}) is not valid. "
                     + "Only 'inch' and 'pix'/'dots' are")

  return w, h


def resize_figure(size='optimal', fig=None, sf_a=0.9, orientation='landscape', dy_inch='auto',
                  tighten=True, shortest_dim_mm=100., pos=(50, 50)):
  '''
  resize_figure sets the figure size such that the ratio for the A-format is kept, while maximizing
  the display on the screen.None

  positional arguments:
  ---------------------
  <none>

  keyword arguments:
  ------------------
  fig         [handle] figure handle. If None is given, the current figure handle will be taken
  size        [None/'maximize'/list(float)] either None or a list of 2 elements containing the
                                 width and
                                 height in inches. In case None is given (the default), the figure
                                 is maximized to the screen while maintaining the a-format ratio of
                                 sqrt(2) to 1
  orientation [str] either 'portrait' or 'landscape', this value is only used in combination with
                    size=None. Otherwise this value is don't care
  tight_layout [bool] is boolean indicating to use the function 'jktools.tighten' to
                      minimize whitespace around the axes while still preserving room for all
                      possible titles. Works well when size=None

  resize_figure returns None, but immediately updates the figure size via the function
  fig.set_size_inches(..., forward=True)

  '''
  if fig is None:
    fig = plt.gcf()

  mng = plt.get_current_fig_manager()

  # move figure to top-left
  if pos is not None:
    _, _, width, height = mng.window.geometry().getRect()
    mng.window.setGeometry(*pos, width, height)

  shortest_dim_inch = shortest_dim_mm/25.4

  if isinstance(dy_inch, str) and dy_inch == 'auto':
    top_fig = fig.subplotpars.top
    hinch = fig.get_size_inches()[1]
    top_inch = top_fig*hinch
    dy_inch = hinch - top_inch

  # if; maximize
  wmax, hmax = get_max_a_size_for_display(units='inches', orientation=orientation)
  width = 1.1*shortest_dim_inch
  height = shortest_dim_inch
  if isinstance(size, str):
    if size == "maximize":
      # set figure manager
      # first: maximize
      mng.window.showMaximized()

    # else: A paper dimensions (a/b=sqrt(2))
    elif size.startswith('a'):
      if size == 'amax':
        # Height can always be maximized
        width = wmax
        height = hmax
      else:
        width, height = paper_A_dimensions(int(size[1:]), units='inches',
                                           orientation=orientation)

      # use width and height to set the figure size
      width = sf_a*width
      height = sf_a*height

    elif size.startswith('optimal'):
      # find the number of axes
      axs = fig.get_axes()
      extents = [ax.get_position().extents for ax in axs]
      x0s, y0s, x1s, y1s = np.array(extents).T
      nof_rows = int(0.5 + (np.unique(y0s).size + np.unique(y1s).size)/2)
      nof_cols = int(0.5 + (np.unique(x0s).size + np.unique(x1s).size)/2)
      width *= nof_cols
      height *= nof_rows
      if len(size[7:]) > 0:
        if size[8:].startswith('w'):
          width *= np.sqrt(2)
        elif size[8:].startswith('xw'):
          width *= 2
        elif size[8:].startswith('xxw'):
          width *= 2*np.sqrt(2)
        else:
          print("unknown size '{size}'. Simple 'optimal' taken")

    # else: it will be shorthand from now
    else:
      # width = 1.1*shortest_dim_inch
      # height = shortest_dim_inch
      # the next is shorthand:
      # s: 'smaller'
      # l: 'larger'
      # w: 'wider'
      # t: 'taller'
      # e.g.: 'wwtt' is the same as 'll'. Or 'wl'
      nof_s = np.sum([elm == 's' for elm in list(size)])
      nof_l = np.sum([elm == 'l' for elm in list(size)])
      nof_w = np.sum([elm == 'w' for elm in list(size)])
      nof_t = np.sum([elm == 't' for elm in list(size)])
      wfactor = np.power(np.sqrt(2), nof_w + nof_l - nof_s)
      hfactor = np.power(np.sqrt(2), nof_t + nof_l - nof_s)
      width *= wfactor
      height *= hfactor

    # # compensate for the dy_inch offset
    # height += dy_inch

  # else: witdth and height are given
  else:
    if np.isscalar(size):
      size = [size]
    else:
      width = 1.1*shortest_dim_inch
      height = shortest_dim_inch

    width, height = size

  # change the size
  wh_ratio = width/height
  if width > wmax:
    warnings.warn(f"The width ({width}) exceeds the screen width ({wmax})! "
                  + "Clipped with same aspect!")
    width = wmax
    height = width/wh_ratio

  if height > hmax:
    height = hmax
    width = height*wh_ratio
    warnings.warn(f"The height ({height}) exceeds the screen height ({wmax})! "
                  + "Clipped with same aspect!")

  fig.set_size_inches(width, height, forward=True)
  plt.draw()
  plt.pause(1e-3)

  if tighten:
    fig.tight_layout()
    plt.draw()
    plt.pause(1e-3)

  # set the top position
  ytop = 1 - dy_inch/fig.get_size_inches()[1]

  fig.tight_layout()
  plt.draw()
  plt.pause(1e-3)

  fig.subplots_adjust(top=ytop)
  plt.draw()
  plt.pause(1e-3)

  return fig


def strip_common_parts(list_of_strings):
  """
  strip the common parts of the strings in a list
  """
  # strip the beginnings
  (common_start,
   stripped_from_start) = get_common_part(list_of_strings, return_uncommon=True, from_end=False)
  (common_end,
   stripped) = get_common_part(stripped_from_start, return_uncommon=True, from_end=True)

  return stripped, common_start, common_end


def get_common_part(list_of_strings, return_uncommon=False, from_end=False):
  """
  what is the common part in all strings in the list
  """
  list_of_strings = listify(list_of_strings)
  # corner case: single element
  if len(list_of_strings) == 1:
    return list_of_strings

  # how many
  nof_strs = len(list_of_strings)

  # find the maximum length
  sizes = np.array([len(str_) for str_ in list_of_strings])
  size_min = sizes.min()

  # build the character array
  chararr = np.empty((nof_strs, size_min), dtype="U1")
  for istr, str_ in enumerate(list_of_strings):
    if from_end:
      str_ = str_[-1::-1]
    chararr[istr, :] = list(str_[:size_min])

  is_equal = np.zeros((size_min), dtype=np.bool_)
  for ichar in range(size_min):
    is_equal[ichar] = True if np.unique(chararr[:, ichar]).size == 1 else False

  ifirst_uncommon = np.argwhere(~is_equal).ravel()[0]

  if from_end:
    common_part = list_of_strings[0][-ifirst_uncommon:]
  else:
    common_part = list_of_strings[0][:ifirst_uncommon]

  if return_uncommon:
    stripped_list_of_strings = []
    for str_ in list_of_strings:
      if from_end:
        str__ = str_[:-ifirst_uncommon]
      else:
        str__ = str_[ifirst_uncommon:]
      stripped_list_of_strings.append(str__)

      # create output
      output = (common_part, stripped_list_of_strings)
  else:
    output = common_part

  return output


def savefig(fig=None, ask=False, name=None, dirname=None, ext=".png", force=False,
            close=False, treat_minus_as_hyphen=False, signs_in_words=False,
            set_lowercase=True, throw_exception=False, **savefig_kwargs):
  """
  save the figure
  """
  # ------------- replaceables ----------------------------
  replacedict = {" ": "_",
                 ",": "",
                 ")": "",
                 "(": "__",
                 "[": "__",
                 "]": "",
                 "{": "__",
                 "}": "",
                 "|": "",
                 "~": "",
                 "%": ""}
  if treat_minus_as_hyphen:
    replacedict["-"] = "_"
  if signs_in_words:
    replacedict["+"] = "plus"
    replacedict["-"] = "minus"

  # get the figures
  if fig is None:
    fig = plt.gcf()

  kwargs = dict(format=ext[1:])
  kwargs.update(savefig_kwargs)

  # ----------- get the name and tidy up --------------
  if name is None:
    name = fig.get_label()
    # remove [x] part
    istop = name.find("[")
    if istop > 0:
      name = name[:istop]
  # tidy up the name
  for key, value in replacedict.items():
    name = name.replace(key, value)

  if set_lowercase:
    name = name.lower()

  # -------------- SAVE DIRECTORY ----------------------
  if dirname is None:
    dirname = os.curdir

  if ask:
    ffilename = select_savefile(title="select the filename",
                                initialdir=dirname,
                                initialfile=name + ext,
                                filetypes=[("PNG files", ".png"),
                                           ("JPEG files", ".jpg"),
                                           ("GIF files", ".gif"),
                                           ("all files", ".txt")])
  else:
    if name.endswith(ext):
      ffilename = os.path.join(dirname, name)
    else:
      ffilename = os.path.join(dirname, name + ext)

  if not force:
    if os.path.exists(ffilename):
      if throw_exception:
        raise FileExistsError(f"The file '{ffilename}' already exists")

      print(f"File '{ffilename}' already exists. Not overwritten!")
      return None

  print(f"Saving figure '{ffilename}' .. ", end='')

  ext = os.path.splitext(ffilename)[-1]
  kwargs.update(format=ext[1:])
  fig.savefig(ffilename, **kwargs)
  print("done")

  if close:
    plt.close(fig)

  return None


def timer(seconds, minutes=0, hours=0, days=0, only_seconds=False, ndec=1):
  """ running time either forwards (stopwatch) or backwards (timer) """
  length = os.get_terminal_size().columns
  empty_line = ' '*(length - 10)

  tstop = ((days*24 + hours)*60 + minutes)*60 + seconds
  tic = time.time()
  # loop if stop time is not passed yet
  while True:
    telapsed = time.time() - tic
    ttogo = tstop - telapsed
    if only_seconds:
      print(f"\r{ttogo:0.{ndec}f} seconds", end="\r", flush=True)
    else:
      days = int(ttogo/86400)
      hours = int((ttogo - days*86400)/3600)
      minutes = int((ttogo - days*86400 - hours*3600)/60)
      seconds = ttogo - days*86400 - hours*3600 - minutes*60

      # build string
      daystr = f"{days} days, "
      hourstr = f"{hours} hours, "
      minstr = f"{minutes} minutes, "
      secstr = f"{seconds:0.{ndec}f} seconds"
      if days > 0:
        telstr = daystr + hourstr + minstr + secstr
      else:
        if hours > 0:
          telstr = hourstr + minstr + secstr
        else:
          if minutes > 0:
            telstr = minstr + secstr
          else:
            telstr = secstr
      print("\r" + empty_line, end="\r", flush=True)
      print(f"\r{telstr}", end="\r", flush=True)

    # check if loop must be broken
    if telapsed >= tstop:
      print("\r" + empty_line, end="\r", flush=True)
      print("\r{fimish_msg}", end="\r", flush=True)
      break
    time.sleep(0.025)

  return None


def sleep(sleeptime, msg='default', polling_time=0.1, nof_blinks=1, loopback=False,
          wakemsg='awake!'):
  """
  a sleep function that shows a wait message
  """
  if msg is None:
    time.sleep(sleeptime)
    return None

  if msg == 'default':
    wm = create_wait_message(nof_blinks=nof_blinks, loopback=loopback)
  else:
    wm = create_wait_message(msg=msg, nof_blinks=nof_blinks, loopback=loopback)

  tic = time.time()
  while time.time() - tic < sleeptime:
    print("\r" + next(wm), end='\r', flush=True)
    time.sleep(polling_time)

  print(f"\r{wakemsg:{len(msg)}s}")

  return None


def create_wait_message(msg="sssstt! I'm asleep!", nof_blinks=1,
                        loopback=True):
  """
  display a wait message
  """
  nof_chars = len(msg)
  parts = []

  # single full message
  for ichar in range(nof_chars):
    parts.append(f"{msg[:ichar]:{nof_chars}s}")
  parts.append(msg)

  # blinking
  if nof_blinks > 0:
    to_append = [" "*nof_chars]*nof_blinks
    parts.append(*to_append)

  # loopback
  if loopback:
    for ichar in range(nof_chars, 0, -1):
      parts.append(f"{msg[:ichar]:{nof_chars}s}")

  # make an iterable out of it
  rotor = cycle(parts)

  return rotor


def remove_empty_axes(fig):
  """
  remove empty axes
  """
  axs = listify(fig.axes)

  artists_to_check = ['lines', 'collections', 'images']
  for ax in axs:
    nof_artists_in_ax = 0
    for art in artists_to_check:
      nof_artists_in_ax += len(getattr(ax, art))

    if nof_artists_in_ax == 0:
      ax.remove()

  plt.draw()

  return None


def inspector(obj2insp, searchfor=None, maxlen=None, name=None):
  """
  inspect an object
  """
  if isinstance(obj2insp, dict):
    inspect_dict(obj2insp, searchfor=searchfor, maxlen=maxlen, name=name)
  elif isinstance(obj2insp, list):
    if isinstance(obj2insp[0], dict):
      inspect_list_of_dicts(obj2insp, fields=searchfor, print_=True)
    else:
      inspect_object(obj2insp, searchfor=searchfor, maxlen=maxlen, name=name)

  else:
    inspect_object(obj2insp, searchfor=searchfor, maxlen=maxlen, name=name)

  return None


def inspect_dict(dict_, searchfor=None, maxlen=None, name=None):
  """
  show all key value pairs in a dictionary
  """
  keys = list(dict_.keys())

  if searchfor is not None:
    keys = [key for key in keys if key.find(searchfor) > -1]

  linetext = " Dictionary inspector "
  if name is not None:
    linetext += f"- '{name}' "
  markerline("=", text=linetext)
  list_to_print = [['NAME', 'TYPE(#)', 'VALUES/CONTENT']]
  for key in keys:
    value = dict_[key]
    # if np.isscalar(value):
    if isinstance(value, (int, np.integer)):
      item_for_list = [key, 'integer', f'{value}']
    elif isinstance(value, (complex, np.complexfloating)):
      item_for_list = [key, 'complex', f'{value:3g}']
    elif isinstance(value, str):
      item_for_list = [key, 'string', f"'{value.strip()}'"]
    elif isinstance(value, (float, np.floating)):
      item_for_list = [key, 'float', f'{value:3g}']
    elif isinstance(value, bytes):
        item_for_list = [key, 'string', value.decode('utf-8')]
    elif isinstance(value, dict):
      item_for_list = [key, f'{len(value.keys())}-dict',
                       f'{value}']
    elif isinstance(value, np.ndarray):
      item_for_list = [key, f'{value.shape}-array ({value.dtype})',
                       print_list(value.ravel(), maxlen=maxlen, max_num_elms=3)]
    elif isinstance(value, list):
      item_for_list = [key, f'{len(value)}-list ({np.array(value).dtype})',
                       print_list(value, maxlen=maxlen, max_num_elms=3)]
    elif isinstance(value, tuple):
      item_for_list = [key, f'{len(value)}-tuple ({np.array(value).dtype})',
                       print_list(listify(value), maxlen=maxlen, max_num_elms=3)]
    elif value is None:
      item_for_list = [key, 'None', 'None']
    elif isinstance(value, object):
      item_for_list = [key, 'object', value.__class__.__name__]
    else:
      raise TypeError(f"The type of the value ({type(value)}) is not implemented or known")

    list_to_print.append(item_for_list)

  print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.',
                   shorten_last_col=True, maxlen=maxlen)

  return None


def inspect_object(obj, searchfor=None, show_methods=True, show_props=True, show_unders=False,
                   show_dunders=False, maxlen='auto', name=None):
  """
  show all class properties
  """
  attrs = dir(obj)

  if not show_dunders:
    attrs = [attr for attr in attrs if not attr.startswith("__")]

  if not show_unders:
    attrs = [attr for attr in attrs if not attr.startswith("_")]

  if searchfor is not None:
    attrs = [attr for attr in attrs if attr.find(searchfor) > -1]

  # split into properties and methods
  props = [attr for attr in attrs if not callable(getattr(obj, attr))]
  meths = [attr for attr in attrs if callable(getattr(obj, attr))]

  linetext = " Object inspector "
  if name is not None:
    linetext += f"- '{name}' "
  markerline("=", text=linetext)
  print(f"\nClass: '{obj.__class__.__name__}' ")
  if obj.__doc__ is not None:
    print(f"Docstring: {obj.__doc__.strip()}")
  if searchfor is not None:
    print(f"\nATTENTION: Searching for string: '{searchfor}' in attribute name")

  if show_methods:
    print("\n")
    markerline("-", text=' METHODS ')
    print("")
    list_to_print = [['NAME', 'SIGNATURE']]
    for methname in meths:
      meth = getattr(obj, methname)
      try:
        list_to_print.append([methname, str(inspect.signature(meth))[1:-1]])
      except ValueError:
        list_to_print.append([methname, '<no signature found>'])

    print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.',
                     shorten_last_col=True, maxlen=maxlen)

  if show_props:
    print("\n")
    markerline('-', text=' PROPERTIES ')
    print("")
    list_to_print = [['NAME', 'TYPE(#)', 'VALUES/CONTENT']]
    for propname in props:
      prop = getattr(obj, propname)
      if np.isscalar(prop):
        if isinstance(prop, (int, np.integer)):
          item_for_list = [propname, 'integer', f'{prop}']
        elif isinstance(prop, str):
          prop = prop.replace('\n', '\\n').replace('\r', '\\r').replace('\r', '\\t')
          item_for_list = [propname, 'string', f"'{prop.strip()}'"]
        elif isinstance(prop, (float, np.floating)):
          item_for_list = [propname, 'float', f'{prop:3g}']
        elif isinstance(prop, (complex, np.complexfloating)):
          item_for_list = [propname, 'complex', f'{prop:3g}']
      elif isinstance(prop, dict):
        item_for_list = [propname, f'{len(prop.keys())}-dict',
                         f'{print_list(list(prop.keys()))}']
      elif isinstance(prop, np.ndarray):
        item_for_list = [propname, f'{prop.shape}-array ({prop.dtype})',
                         print_list(prop.ravel())]
      elif isinstance(prop, list):
        item_for_list = [propname, f'{len(prop)}-list ({np.array(prop).dtype})',
                         print_list(prop)]
      elif isinstance(prop, tuple):
        item_for_list = [propname, f'{len(prop)}-tuple ({np.array(prop).dtype})',
                         print_list(listify(prop))]
      elif prop is None:
        item_for_list = [propname, 'None', 'None']
      elif isinstance(prop, object):
        item_for_list = [propname, 'object', prop.__class__.__name__]
      else:
        raise TypeError(f"the property type ({type(prop)}) is not valid")

      list_to_print.append(item_for_list)  # pylint: disable=E0606
    print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.',
                     maxlen=maxlen, shorten_last_col=True)
  markerline("=", text=" End of class content ")

  return None


def inspect_list_of_dicts(list_of_dicts, fields=None, print_=False, skip_none_type=True):
  """
  list the stream measurements from meas_hist
  """
  if fields is None:
    fields = set()
    for dict_ in list_of_dicts:
      if dict_ is not None:
        fields = fields.union(set(dict_.keys()))

    fields = [*fields]

  fields = listify(fields)

  datadict = dict.fromkeys(fields)
  for key in datadict:
    datadict[key] = []

  strlist = [['index', *deepcopy(fields)]]
  for ielm, dataelm in enumerate(list_of_dicts):
    strlist_ = [f'{ielm}']
    if dataelm is None and skip_none_type:
      strlist_ += ['None'] + ['']*(len(fields) - 1)

    else:
      for field in fields:
        val = dataelm[field]
        if isinstance(val, (list, tuple, np.ndarray)):
          typestr = f"{type(val)}".format(type(val))
          valstr = f"({len(val)}x) {typestr[8:-2]}"
        else:
          if isinstance(val, (float, complex, np.complexfloating, np.floating)):
            valstr = f"{val:3g}"
          else:
            valstr = f"{val}"
        strlist_.append(valstr)
        datadict[field].append(val)

    # add the new 'row' to the list of listsc
    strlist.append(strlist_)
    # print(strlist)

  if print_:
    if not isinstance(strlist[0], list):
      strlist = [strlist]

    print_in_columns(strlist, maxlen=100, hline_at_index=1, what2keep='begin')

  # check if there are multiple files
  output = datadict
  if len(datadict.keys()) == 1:
    output = datadict[fields[0]]

  return output


def dictify(list_of_dicts, skip_none_type=True, make_array=True):
  """
  make a dictionary of a list of dicts
  """
  # get the keys
  for dict_ in list_of_dicts:
    # check if this dict is not None
    if dict_ is not None:
      keys = dict_.keys()
      dict_of_lists = dict.fromkeys(keys)
      for key in keys:
        dict_of_lists[key] = []
      break

  # loop all dicts in the list
  for dict_ in list_of_dicts:
    # corner case: no dictionary but a single value
    if dict_ is None:
      if skip_none_type:
        continue
      else:
        for key in keys:
          dict_of_lists[key].append(None)
    # else: a normal dict
    else:
      for key in keys:
        dict_of_lists[key].append(dict_[key])

  if make_array:
    for key in keys:
      dict_of_lists[key] = arrayify(dict_of_lists[key])

  return dict_of_lists


def format_matdata_as_dataframe(matdata, fields_to_keep=None):
  """
  format the data read from a matfile to a dataframe
  """

  datadict = dict()
  keys = [key for key in matdata.keys() if not key.startswith("__")]

  if fields_to_keep is None:
    fields_to_keep = keys

  for key in fields_to_keep:
    if isinstance(matdata[key], np.ndarray):
      if matdata[key].ndim > 1:
        value_list = [np.array(row) for row in matdata[key]]
        datadict[key] = value_list
      else:
        datadict[key] = matdata[key]
    else:
      datadict[key] = matdata[key]

  # make into data frame
  datadf = pd.DataFrame(data=datadict)

  return datadf


def interpret_sequence_string(seqstr, lsep=",", rsep=':', typefcn=float, check_if_int=True,
                              is_iq=False, nof_vals_per_meas_point=1):
  """
  interpret a sequence string like '0, 10, 3' or '10.4' or 10:0.1:20'

  arguments:
  ----------
  seqstr: str
          Constains an interpretable string. E.g. piet: '1, 2, 3, 5'
  lsep: str, default=','
        The list separator. that is the separator which occurs often and separates the values
  rsep: str, default=':'
         The separator which splits the remainder from the list string
  typefcn: <function>, default=float
            The function for the values. Default I assume they are all floats
  check_if_int: bool, default=True
                 Check if the floats are actually ALL integers. In that case the output is converted
  is_iq: bool, default=False
          if the data is a SIMPLE stream of alternating i, q samples, this will make them complex.
          The number of samples is thus halved using this set to True. can only be used in
          combination with `nof_vals_per_meas_point=1`. Otherwise, the latter takes precedence.
  nof_vals_per_meas_point: int, default=1
                           the number of values associated with every measurement point. For IQ
                           samples this is forced to 2. However, it can have any value.
                           If this is set to anything other than 1 or 2 AND `is_iq` is set, an
                           exception will be thrown.

  return:
  -------
  output : ndarray
           All the interpreted values, may be reshaped to a 2D array in case
           `nof_vals_per_meas_point` is not 1
           In
  """
  # split the string
  pos_parts_list = seqstr.split(rsep)
  if len(pos_parts_list) == 1:
    # see if they are separate values
    seq_values = np.array([typefcn(pos.strip()) for pos in seqstr.strip().split(lsep)])
  elif len(pos_parts_list) == 2:
    begin_, end_ = [typefcn(val) for val in pos_parts_list]
    incr_angle = 1.
    seq_values = np.arange(begin_, end_+0.001, incr_angle)
  elif len(pos_parts_list) == 3:
    begin_, incr_, end_ = [typefcn(val) for val in pos_parts_list]
    seq_values = np.arange(begin_, end_+0.001, incr_)
  else:
    raise ValueError(f"The values in the sequence '{seqstr}' cannot be determined")

  # check if they are integers
  if typefcn in (np.floating, float):
    if check_if_int:
      is_int_seq = np.alltrue([elm.is_integer() for elm in seq_values])
      if is_int_seq:
        seq_values = np.int_(seq_values)

  # make output variable
  output = seq_values

  # check if measurement points have more than 1 value (i, q, i, q or freq, val, freq, val...)
  if nof_vals_per_meas_point > 1:
    # check dims
    output = output.reshape(-1, nof_vals_per_meas_point).T

  # else: check if they are IQ values
  else:
    if is_iq:
      output = output[::2] + 1j*output[1::2]

  return output


def find_outliers(data, sf_iqr=1.5, axis=None):
  """
  find the outliers according to the 1.5 iqr method
  """
  if data.ndim > 1:
    if axis is None:
      output = find_outliers(data.ravel(), sf_iqr=sf_iqr, axis=None)
      output = np.reshape(output, data.shape)
      return output
    else:
      # move the iteration axis to the first index
      datamod = np.moveaxis(data, axis, 0)
      datamod2 = datamod.reshape(datamod.shape[0], np.prod(datamod.shape[1:])).T
      # datamod2 = np.moveaxis(datamod, 0, 1)
      resultveclist = []
      for datavec in datamod2:
        resultvec = find_outliers(datavec, sf_iqr=sf_iqr, axis=0)
        resultveclist.append(resultvec)

      # reshape
      results = np.array(resultveclist)

      results = np.moveaxis(results.T, 0, axis)

      return results

  if np.iscomplex(data).sum() >= 1:
    data_ = np.abs(data)
  else:
    data_ = data.copy()

  q1 = np.percentile(data_, 25)
  q3 = np.percentile(data_, 75)
  iqr = q3 - q1

  # determine low and high thresholds
  thres_low = q1 - sf_iqr*iqr
  thres_high = q3 + sf_iqr*iqr

  # test against thresholds
  tf_inliers = (data_ >= thres_low)*(data_ <= thres_high)
  tf_outliers = ~tf_inliers

  return tf_outliers


def dec2hex(decvals, nof_bytes=None):
  """
  convert a decimal value tot hexadecimal representation also for a array-like
  """
  decvals_arr = arrayify(decvals)

  shape = decvals_arr.shape

  decvals_vec = decvals_arr.ravel()

  # make hexadecimal values
  hexvals_list = [hex(decval) for decval in decvals_vec]

  # make them equally long
  nof_bytes_min = max([len(hexval) - 2 for hexval in hexvals_list])

  if nof_bytes is None:
    nof_bytes = nof_bytes_min
  else:
    nof_bytes = max(nof_bytes_min, nof_bytes)

  nof_zeros_to_pad = np.array([nof_bytes*2 - (len(hexval) - 2) for hexval in hexvals_list])

  hexvals_pad_list = ['0x' + n0*'0' + hexval[2:] for (n0, hexval)
                      in zip(nof_zeros_to_pad, hexvals_list)]

  hexvals = np.array(hexvals_pad_list).reshape(shape)

  if isinstance(decvals, list):
    hexvals = listify(hexvals)
  elif isinstance(decvals, tuple):
    hexvals = tuplify(hexvals)
  elif np.isscalar(decvals):
    hexvals = hexvals[0]

  return hexvals


def plot_grid(data, *args, ax=None, aspect='equal', center=False, tf_valid=None, **kwargs):
  """
  plot a grid from a 2D set of complex data
  """
  kwargs_ = dict(color='k', lw=2, ls='-', marker='')
  kwargs_.update(kwargs)

  if np.iscomplex(data.item(0)):
    data_cplx = data
  else:
    data_cplx = data + 1j*args[0]
    args = args[1:]

  # if len(args) == 0:
  #   args = ('',)

  xmeas = np.real(data_cplx)
  ymeas = np.imag(data_cplx)

  nr, nc = data_cplx.shape

  if tf_valid is None:
    tf_valid = np.ones((nr, nc), dtype=np.bool_)
  else:
    if tf_valid.shape != (nr, nc):
      raise ValueError("The dimensions for *tf_valid* are not equal to *data*.")

  # plot a grid of lines
  if ax is None:
    _, ax = plt.subplots(1, 1)
  elif isinstance(ax, str) and ax.startswith('h'):
    ax = plt.gca()

  for irow in range(nr):
    if tf_valid[irow, :].sum() > 0:
      # plot horizontal line
      qplot(ax, xmeas[irow, tf_valid[irow, :]], ymeas[irow, tf_valid[irow, :]], *args,
            plotnow=False, **kwargs_)
  for icol in range(nc):
    if tf_valid[:, icol].sum() > 0:
      # plot vertical line
      qplot(ax, xmeas[tf_valid[:, icol], icol], ymeas[tf_valid[:, icol], icol], *args,
            plotnow=False, **kwargs_)

  # set the aspect ratio
  ax.set_aspect(aspect)

  # center if necessary
  if center:
    center_plot_around_origin(ax)

  plt.show(block=False)

  return ax


def add_zoom_inset(zoombox, ax=None, loc='top left', padding=0.1, buffer=0.08, fraction=0.4,
                   alpha=0.9, grid=True, indicate_zoombox=True, facecolor=(0.9)*3, **axkwargs):
  """
  add an inset for zooming
  """
  if np.isscalar(fraction):
    fraction = [fraction]*2

  if np.isscalar(buffer):
    buffer = [buffer]*2

  # get axees
  if ax is None:
    ax = plt.gca()

  if isinstance(loc, str):
    if loc.find("bottom") > -1 or loc.find("lower") > -1:
      yb = buffer[1]
    elif loc.find("top") > -1 or loc.find("upper") > -1:
      yb = 1. - buffer[1] - fraction[1]
    else:
      yb = 0.5 - fraction[1]/2

    if loc.find("left") > -1:
      xl = buffer[0]
    elif loc.find("right") > -1:
      xl = 1. - buffer[0] - fraction[0]
    else:
      xl = 0.5 - fraction[0]/2

  # else: it is a fraction of the window where the center is located
  else:
    if np.isscalar(loc):
      loc = [loc]*2

    if np.isclose(0., loc[0]):
      xl = buffer[0]
    elif np.isclose(1., loc[0]):
      xl = 1. - fraction[0] - buffer[0]
    else:
      xl = min(1. - fraction[0] - buffer[0], max(buffer[0], loc[0] - fraction[0]/2))

    if np.isclose(0., loc[1]):
      yb = buffer[1]
    elif np.isclose(1., loc[1]):
      yb = 1. - fraction[1] - buffer[1]
    else:
      yb = min(1. - fraction[1] - buffer[1], max(buffer[1], loc[1] - fraction[1]/2))

  # convert to figure coordinates
  blin_disp = ax.transAxes.transform((xl, yb))
  trin_disp = ax.transAxes.transform((xl+fraction[0], yb+fraction[1]))

  blin_ax = ax.transAxes.inverted().transform(blin_disp)
  trin_ax = ax.transAxes.inverted().transform(trin_disp)

  w_ax, h_ax = trin_ax - blin_ax

  ax_inset = ax.inset_axes([*blin_ax, w_ax, h_ax], facecolor=facecolor, **axkwargs)
  # ax_inset = fig.add_axes([*blin_fig, w_fig, h_fig], **axkwargs)
  ax_inset.patch.set_alpha(alpha)
  # ax_inset.margins(0.0)
  ax_inset.grid(grid)
  ax_inset.set_transform("axes")
  # ax_inset.set_title("zoom window", fontsize=8, fontweight='bold', backgroundcolor='w')

  props_to_copy = ['color',
                   'linestyle',
                   'linewidth',
                   'marker',
                   'markersize',
                   'mfc',
                   'mec',
                   'alpha',
                   'zorder']
  linelist = ax.get_lines()
  for ln in linelist:
    ln_ = ax_inset.plot(ln.get_xdata(), ln.get_ydata())[0]
    for prop in props_to_copy:
      propval = plt.getp(ln, prop)
      if not np.isscalar(propval):
        propval = listify(propval)
      plt.setp(ln_, prop, propval)

  # add data if required
  if len(zoombox) == 2:
    xdmin, xdmax = zoombox
    # determine ydmin and ydmax automatically
    linelist = ax.get_lines()
    if len(linelist) == 0:
      ydmin, ydmax = ax.get_ylim()
    else:
      ydmin = np.inf
      ydmax = -np.inf
      for ln in linelist:
        xdata = ln.get_xdata()
        tf_valid = (xdata >= xdmin)*(xdata <= xdmax)
        ydata = ln.get_ydata()
        ydmin_, ydmax_ = bracket(ydata[tf_valid])
        ydmin = min(ydmin, ydmin_)
        ydmax = max(ydmax, ydmax_)
  else:
    xdmin, xdmax, ydmin, ydmax = zoombox

  # add 10% buffer
  xdmid = (xdmax + xdmin)/2
  xdrange = (1 + padding)*(xdmax - xdmin)
  xdmin = xdmid - xdrange/2
  xdmax = xdmid + xdrange/2

  # add 10% buffer
  ydmid = (ydmax + ydmin)/2
  ydrange = (1 + padding)*(ydmax - ydmin)
  ydmin = ydmid - ydrange/2
  ydmax = ydmid + ydrange/2

  ax_inset.set_xlim(left=xdmin, right=xdmax)
  ax_inset.set_ylim(top=ydmax, bottom=ydmin)

  if indicate_zoombox:
    ax.indicate_inset_zoom(ax_inset, edgecolor='black')
  plt.draw()
  plt.pause(1e-3)

  return ax, ax_inset


def add_text_inset(text_inset_strs_list, x=None, y=None, loc='upper right', axfig=None,
                   ha='right', va='top', left_align_lines=True, boxcolor=(0.8, 0.8, 0.8),
                   boxalpha=1., fontweight='normal', fontsize=8, fontname='monospace',
                   fontcolor='k'):
  """
  add text inset
  """
  # get the positions
  xpos = x
  ypos = y
  if xpos is None:
    if loc.lower().find("right") > -1:
      xpos = 0.98
    elif loc.lower().find("left") > -1:
      xpos = 0.0
    elif loc.lower().find("center") > -1:
      xpos = 0.5

  if ypos is None:
    if loc.lower().find("upper") > -1 or loc.lower().find("top") > -1:
      ypos = 0.98
    elif loc.lower().find("lower") > -1 or loc.lower().find("bottom") > -1:
      ypos = 0.02
    elif loc.lower().find("center") > -1:
      ypos = 0.5

  print(xpos, ypos)

  # get axees
  if axfig is None:
    ax = plt.gca()
    fig = ax.figure
    axfig = fig

  if isinstance(axfig, plt.Figure):
    transform = axfig.transFigure
  else:
    transform = axfig.transAxes

  # info is on the right side, calculate the offset
  if ha == 'right' and left_align_lines:
    # determine the size of the box
    nof_chars_right_box = max([len(str_) for str_ in text_inset_strs_list])
    text_inset_strs_list = [f"{str_:<{nof_chars_right_box}s}" for str_ in text_inset_strs_list]

  # glue the lines
  text_inset_text = '\n'.join(text_inset_strs_list)

  # add the text to the axes
  txtobj = axfig.text(xpos, ypos, text_inset_text, fontsize=fontsize, fontweight=fontweight,
                      fontname=fontname, ha=ha, va=va,
                      bbox=dict(boxstyle="Round, pad=0.2", ec='k', fc=boxcolor, alpha=boxalpha),
                      transform=transform, color=fontcolor)

  plt.draw()

  return txtobj


def cov_from_image(im2d, remove_outliers=False):
  """ calculate the covariance from an image

  """
  # the value is the weight or frequency
  aweights = im2d[np.nonzero(im2d)]
  # convert to data --> not the transpose

  data = np.argwhere(im2d).T

  if remove_outliers:
    tf_valid_i = ~find_outliers(data[0])[0]
    tf_valid_q = ~find_outliers(data[1])[0]
    tf_valid = tf_valid_i*tf_valid_q
    data[0] = data[0][tf_valid]
    data[1] = data[1][tf_valid]
  cov = np.cov(data, aweights=aweights)

  # determine the center if not given
  center = tuplify(np.mean(data, axis=1))

  return cov, center

def plot_cov(data_or_cov, plotspec='k-', ax=None, center=None, geo='ellipse', nof_pts=101,
             fill=False, conf=0.67, remove_outliers=True, **kwargs):
  """
  plot the covariance matrix
  sf = 5.99 corresponds to the 95% confidence interval
  """
  cov_to_calc = False
  cov = None
  data = None
  if isinstance(data_or_cov, np.ndarray) and data_or_cov.shape == (2, 2):
    cov = data_or_cov
  elif isinstance(data_or_cov, (list, tuple, np.ndarray)):
    cov_to_calc = True
    if len(data_or_cov) == 2:
      data = data_or_cov
    else:  # assume complex numbers
      data = [np.real(data_or_cov), np.imag(data_or_cov)]
  else:
    raise ValueError("The value for argument *data_or_cov* is not valid. Please check!")

  # check if the covariance is still to be plotted
  if cov_to_calc:
    if remove_outliers:
      tf_valid_i = ~find_outliers(data[0])[0]
      tf_valid_q = ~find_outliers(data[1])[0]
      tf_valid = tf_valid_i*tf_valid_q
      data[0] = data[0][tf_valid]
      data[1] = data[1][tf_valid]
    cov = np.cov(data)
    if center is None:
      center = tuplify(np.mean(data, axis=1))

  if center is None:
    center = (0., 0.)

  # parametric representation
  if ax is None:
    ax = plt.gca()
  elif ax == 'new':
    _, ax = plt.subplots(1, 1)

  t = np.linspace(0, 2*np.pi, nof_pts, endpoint=True)
  eigvals, eigvecs = np.linalg.eig(cov)

  # create non-skewed ellipse
  # make scale factor
  sf = np.interp(conf, confidence_table.index, confidence_table.values)
  if geo.startswith('ellipse'):
    xy_unit_circle = np.array([np.cos(t), np.sin(t)])

    xy_straight_ellipse = np.sqrt(sf*eigvals).reshape(2, 1)*xy_unit_circle
    xy_ellipse = eigvecs@xy_straight_ellipse  # noqa
    xt_, yt_ = xy_ellipse

    # add the center point
    xt = xt_ + center[0]
    yt = yt_ + center[1]
    if fill:
      ax.fill(xt, yt, plotspec, **kwargs)

    else:
      ax.plot(xt, yt, plotspec, **kwargs)
  elif geo.startswith('rect'):
    isort = np.argsort(eigvals)
    evals_sort = eigvals[isort]
    evecs_sort = eigvecs[:, isort]

    lmaj = np.sqrt(sf*evals_sort[1])
    lmin = np.sqrt(sf*evals_sort[0])

    rotation_of_major_axis = np.arctan2(evecs_sort[1, 1], evecs_sort[0, 1])

    # shift anchor by half the
    xanch = center[0] - lmaj
    yanch = center[1] - lmin
    rect = Rectangle((xanch, yanch), lmaj*2, lmin*2,
                      angle=np.rad2deg(rotation_of_major_axis),
                      rotation_point='center',
                      fc='none',
                      ec='y',
                      zorder=10)
    ax.add_patch(rect)

  plt.show(block=False)
  plt.draw()

  return ax, center, cov


def print_list(list2glue, sep=', ', pfx='', sfx='', floatfmt='{:f}', intfmt='{:d}',
               strfmt='{:s}', cplxfmt='{:f}', compress=False, maxlen=None,
               spiffy=False, max_num_elms=None, check_for_ints=False, **short_kws):
  """
  glue a list of elements to a string
  """
  def empty(arg):
    return arg

  if len(list2glue) == 0:
      return '[]'

  types_conv_dict = {str: (strfmt, empty),
                     (int, np.integer): (intfmt, empty),
                     (float, np.floating): (floatfmt, empty),
                     (complex, np.complexfloating): (cplxfmt, empty),
                     (bool, np.bool_): ('{}', empty),
                     dict: ("<{:d}-key dict>", len),
                     list: ("<{:d}-list>", len),
                     type(None): ("None", empty),
                     np.ndarray: ("<{:d}-array>", len),
                     np.void: ("np.void", empty),
                     dtm.datetime: ('{}', timestamp)}

  # make inputs a list (in case they are not)
  list2glue = listify(list2glue)
  nof_in_list = len(list2glue)

  # corner case: single element
  if nof_in_list == 1:
    value = list2glue[0]
    if isinstance(value, (np.floating, float)):
      string = floatfmt.format(value)
    elif isinstance(value, (np.integer, int)):
      string = intfmt.format(value)
    elif isinstance(value, str):
      string = strfmt.format(value)
    elif isinstance(value, dict):
      string = f"<{len(value)}-key dict>"
    elif isinstance(value, object):
      string = f"name: {value.name}"
    else:
      raise TypeError(f"The value type given ({type(value)}) is not recognized")

    return pfx + string + sfx

  if spiffy and len(pfx) > 0:
    pfx_sep_opts = [':', '=', ';', '-']
    ifnds_sep_from_back = [pfx[-1::-1].find(pfx_sep) for pfx_sep in pfx_sep_opts]
    ifnds_sep_from_back_valid = [ifnd for ifnd in ifnds_sep_from_back
                                 if ifnd > -1]

    if len(ifnds_sep_from_back_valid) > 0:
      # get the position of the separator (nb. the finds where from the back)
      isep = len(pfx) - (1 + min(ifnds_sep_from_back_valid))

      pfx_sep = pfx[isep]

      replace_with = f" ({len(list2glue)}x){pfx_sep}"
      pfx = pfx[:isep] + replace_with + pfx[(isep + 1):]

  # check the three types
  if compress and nof_in_list > 1:
    # check if monospaced
    arr2glue = arrayify(list2glue)
    stepvals = np.unique(np.diff(arr2glue))
    if stepvals.size == 1:
      step = float(stepvals.item())
      minval, maxval = bracket(arr2glue)
      minval = float(minval)
      maxval = float(maxval)
      if minval.is_integer() and maxval.is_integer() and step.is_integer():
        fmt = intfmt
        step = int(step)
        minval = int(minval)
        maxval = int(maxval)
      else:
        fmt = floatfmt
      if np.isclose(1, step):
        output_string = pfx + fmt.format(minval) + ":" + fmt.format(maxval) + sfx
      else:
        output_string = (pfx + fmt.format(minval) + ":" + fmt.format(step) +
                        ":" + fmt.format(maxval) + sfx)
      return output_string

    warnings.warn("This list cannot be compressed in min:step:max, "
                  + "since there is not a single step", category=UserWarning)

  # if not compressed or compressible (after a warning)
  output_parts = []

  # check if integer
  if check_for_ints:
    list2glue = [int(np.sign(elm)*0.5 + float(elm)) if np.abs(float(elm)).is_integer() else elm
                 for elm in list2glue]

  for elm in list2glue:
    if check_for_ints:
      if isinstance(elm, (np.floating, float)):
        elm = int(np.sign(elm)*0.5 + elm) if elm.is_integer() else elm
    for type_, (fmt_, fcn_) in types_conv_dict.items():
      if isinstance(elm, type_):
        output_part = fmt_.format(fcn_(elm))
        output_parts.append(output_part)
        break  # if found, then break out of for loop

  # check if a maximum is specified and strip accordingly
  if max_num_elms is None:
    max_num_elms = np.inf
  num_elms_to_pick = min(max_num_elms, len(output_parts))
  output_string = pfx + sep.join(output_parts[:(num_elms_to_pick-1)])
  if spiffy and nof_in_list > 1:
    output_string += ' and '
  else:
    output_string += sep

  # finalize
  output_string += output_parts[-1] + sfx

  if maxlen is not None:
    output_string = short_string(output_string, maxlength=maxlen, **short_kws)

  return output_string


def print_dict(dict2glue, sep=": ", pfx='', sfx='', glue_list=False, glue="\n", floatfmt='{:f}',
               intfmt='{:d}', strfmt='{:s}', maxlen=None, **short_kws):
  """
  print a dict
  """
  types_conv_dict = {str: strfmt,
                     int: intfmt,
                     np.integer: intfmt,
                     float: floatfmt,
                     np.floating: floatfmt,
                     np.ndarray: '{}'}

  # check the three types
  output_list = []
  for key, value in dict2glue.items():
    if isinstance(value, (list, tuple, np.ndarray)):
      value = listify(value)
      if maxlen is not None:
        maxlen_list = maxlen - len(key) - len(sep)
      else:
        maxlen_list = None
      output_str_ = print_list(value, pfx='{', sfx='}', floatfmt=floatfmt, intfmt=intfmt,
                               strfmt=strfmt, maxlen=maxlen_list, **short_kws)
    else:
      if type(value) in types_conv_dict:
        fmt = types_conv_dict[type(value)]
      else:
        fmt = '{}'
      output_str_ = fmt.format(value)

    # combine to full output string
    output_string = pfx + key + sep + output_str_ + sfx

    if maxlen is not None:
      output_string = short_string(output_string, maxlen, **short_kws)

    output_list.append(output_string)

  if glue_list:
    output = glue.join(output_list)
  else:
    output = output_list

  return output


def print_matrix(mat, pfx=None, ndigits=7, ndec=-1, force_sign=False, as_single=False, sep=', ',
                 check_for_ints=True):
  """
  print a 2D matrix as either a matrix or a list of lists on a single row

  arguments:
  ----------
  mat : np.array 2D
        A 2D np.ndarray containing (complex) floats, integers or booleans
  ndigits : int, default=7
            The number of digits in total per element of the matrix
  ndec : int, default=-1
         The number of decimal numbers in case of floats. A value of -1 implies it being calculated
         by the actual values and the set `ndigits` argument
  as_single : bool, default=False
              (to be implemented) whether to return it as a single row of the form
              [[...], [...],...]
  spiffy : bool, default=False
           (to be implemented yet)

  returns:
  --------
  None
  """
  if pfx is None:
    pfx = ''

  # ==== check the dimensions and make array-like into array ==========
  mat = arrayify(mat)
  if mat.ndim < 2:
    mat = mat.reshape(1, -1)
    # raise DimensionError(f"The dimension is {mat.ndim}. Only 2D is accepted")

  # ============ set the formatting ===========================================
  # should allow nice rectangular shape
  floatfmt : str = ''
  intfmt: str = ''
  if isinstance(mat.item(0), (np.floating, float, np.integer, int)):
    # find the maximum integer
    with np.errstate(divide='ignore'):
      logfloats_max = np.log10(np.abs(mat.ravel()).max())
      # mat_ = mat[mat < 1] = 1.0
      # logfloats = np.log10(np.abs(mat_))

    # # cap the bottom values
    # logfloats[logfloats < 1.] = 1.
    # logfloats_max = np.nanmax(logfloats)
    # get the power of 10 (is the number of digits before the .)
    ndigits_int_part = np.int_(np.ceil(logfloats_max))

    # get the number of digits to be used for the sign (0 or 1)
    signstr = '+' if force_sign else ''
    ndigits_sign_in_data = 1 if np.nanmin(mat) < 0 else 0

    ndigits_sign = np.fmax(ndigits_sign_in_data, force_sign)

    if ndec == -1:
      ndec = ndigits - ndigits_int_part - 1 - ndigits_sign
    else:
      ndigits = ndec + ndigits_int_part + 1 + ndigits_sign

    # the integer formatting is only required in case of 2D
    if mat.squeeze().ndim == 2 and not as_single:
      # determine the float and integer formats
      floatfmt_content = f":{signstr}{ndigits}.{ndec}f"
      floatfmt = f"{{{floatfmt_content}}}"
      intfmt_content = f":{signstr}{ndigits}d"
      intfmt = f"{{{intfmt_content}}}"
    else:
      floatfmt_content = f":{signstr}0.{ndec}f"
      floatfmt = f"{{{floatfmt_content}}}"
      intfmt = "{:d}"

    # print outs
  str2print_list = []
  nr = mat.shape[0]
  for ir in range(nr):
    str2print = print_list(mat[ir, :].tolist(), floatfmt=floatfmt, intfmt=intfmt, pfx="[", sfx="]",
                           sep=sep, check_for_ints=check_for_ints)
    str2print_list.append(str2print)

  # choose how to display
  if as_single:
    print(f"{pfx}[", end="")
    for str2print in str2print_list:
      print(f"{str2print}, ", end="")
    print("\b\b]")
  else:
    for str2print in str2print_list:
      print(f"{pfx}{str2print}")
      pfx = ' '*len(pfx)

  return None


def extract_value_from_strings(input2check, pattern2match, output_fcn=None, output_type=None,
                               notfoundvalue=None, check_if_int=False, replacements=False):
  """
  extract values from a list of strings
  """
  if replacements is None:
    replacements = dict(v=1.,
                        V=1.,
                        uv=1e-6,
                        uV=1e-6,
                        mv=1e-3,
                        mV=1e-3,
                        p=1e-12,
                        n=1e-9,
                        u=1e-6,
                        m=1e-3,
                        k=1e3,
                        M=1e6,
                        g=1e9,
                        G=1e9)
  elif replacements is False:
    replacements = dict()
  elif isinstance(replacements, dict):
    pass
  else:
    raise ValueError("The allowd values for keyword 'replacements' are: 'None', 'False' or 'dict'")

  list_of_strings = listify(input2check)
  fmt = conv_fmt(pattern2match)

  # do the search
  search_results = [re.search(fmt, str_) for str_ in list_of_strings]

  # get the strings
  value_strings = [res.group() if res is not None else notfoundvalue for res in search_results]

  if output_fcn is not None:
    values = [output_fcn(valstr) if valstr is not None else notfoundvalue
              for valstr in value_strings]
  else:
    values = []
    for valstr in value_strings:
      if valstr is None:
        continue
      if valstr.startswith('-') or valstr.startswith('+'):
        valstr_ = valstr[1:]
      else:
        valstr_ = valstr
      if valstr_.isnumeric():
        value = float(valstr)
      else:
        # check against known decimal replacements
        value = valstr
        # check if there is anything to replace
        for key, mult in replacements.items():
          if valstr.find(key) > -1:
            value = float(valstr.replace(key, '.'))*mult
            break

      if check_if_int:
        if value.is_integer():
          value = int(value)

      # append to the list
      values.append(value)

  if output_type is not None:
    values = [output_type(value) for value in values]

  if np.isscalar(input2check):
    output = values[0]
  else:
    output = values

  return output


def conv_fmt(pyfmt):
  """
  convert the python string formatting to re formatting
  note that square brackets enclosing means optional
  """
  # asdfpylint: disable=asdasdfanomalous-backslash-in-string
  pyfmt = pyfmt.replace('+', '\\+')
  py2re = {'*': r'\S+',
           '%c': r'.',
           '%nc': r'.{n}',
           '%d': r'[-+]?\d+',
           '%e': r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%E': r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%f': r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%g': r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '$i': r'[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)',
           '%o': r'[-+]?[0-7]+',
           '%n': r'[0-9_]*',
           '%w': r'[a-zA-z]*',
           '%u': r'\d+',
           '%s': r'\S+',
           '%x': r'[-+]?(0[xX])?[\dA-Fa-f]+',
           '%X': r'[-+]?(0[xX])?[\dA-Fa-f]+'}
  # pylint: enable=anomalous-backslash-in-string

  # process possible square brackets ([...] becomes (...)?)

  # modify the lookback and lookforward parts
  # lookback part (everything before the first %[df...])
  iprefix = pyfmt.find('%')
  if iprefix > 0:
    # check if there is an optional part (format [...]?)
    if pyfmt[iprefix-2:iprefix] == ']?':
      iprefix = pyfmt[:(iprefix-2)].find('[')
    lookback_str = pyfmt[:iprefix]
    lookback_str_fmt = '(?<=' + lookback_str + ')'
    pyfmt = lookback_str_fmt + pyfmt[iprefix:]

  # lookforward part
  isuffix = pyfmt.rfind('%')+2
  if isuffix < len(pyfmt):
    # check if there is a [...]? part -> optional
    if pyfmt[isuffix] == '[':
      isuffix += pyfmt[isuffix:].find(']?') + 2
    lookfw_str = pyfmt[isuffix:]
    lookfw_str_fmt = '(?=' + lookfw_str + ')'
    pyfmt = pyfmt[:isuffix] + lookfw_str_fmt

  # initialize the regexp string
  refmt = pyfmt
  for key, value in py2re.items():
    refmt = refmt.replace(key, value)

  return refmt


def find_pattern(pattern, list_of_strings, nreq=None, nreq_mode='exact',
                 squeeze=True):
  """
  find a (python) formatted file name in a directory
  """

  # convert the formatted file to the regexpr formatted file
  pattern_regexp = conv_fmt(pattern)

  # search all elements in the contents list
  re_results_list = [re.search(pattern_regexp, fname) for fname in list_of_strings]

  # get the strings that match
  valid_filenames = [fname.string for fname in re_results_list if fname is not None]

  # throw nof found in variable, necessary for possible post processing
  nof_found = len(valid_filenames)

  if nreq is None:
    pass

  elif isinstance(nreq, (int, np.integer)):
    errstr = (f"The expected number of found files is ({nreq}), but only ({nof_found}) are found. "
              + f"This is incompatible with *nreq_mode*=`{nreq_mode}`")
    if nof_found == nreq:
      pass
    elif nof_found > nreq:
      if nreq_mode == 'min':
        pass
      elif nreq_mode in ('max', 'exact'):
        raise IncorrectNumberOfFilesFoundError(errstr)
    elif nof_found < nreq:
      if nreq_mode == 'max':
        pass
      elif nreq_mode in ('min', 'exact'):
        raise IncorrectNumberOfFilesFoundError(errstr)
    else:
      raise FileNotFoundError(errstr)

  # check if I have to squeeze
  if squeeze:
    if nof_found == 1:
      valid_filenames = valid_filenames[0]

  # return the stuff
  return valid_filenames


def find_filename(filepattern, dirname, nreq=1, nreq_mode='exact',
                  squeeze=True):
  """
  find a (python) formatted file name in a directory
  """
  # get the contents of the folder
  contents = os.listdir(dirname)

  found_strings = find_pattern(filepattern, contents, nreq=nreq,
                               nreq_mode=nreq_mode, squeeze=squeeze)

  return found_strings


def pconv(dirname):
  """
  convert the path in windows format to linux
  """

  if sys.platform.startswith('win'):
    conversion_table = {'/work': 's:',
                        '/home/dq968/': 'm:',
                        '/': os.sep}

  elif sys.platform.startswith('linux'):
    conversion_table = {'s:': '/work',
                        'm:': '/home/dq968/',
                        '\\': os.sep}
  else:
    raise ValueError(f"The platform found ({sys.platform}) is not known")

  # first ensure all \ are doubled to \\
  # replace all backslashes with forward slashes
  for wkey, sub in conversion_table.items():
    dirname = dirname.replace(wkey, sub)

  # end with file separator
  if not dirname.endswith(os.sep):
    dirname += os.sep

  return dirname


def nof_bits_needed(count):
  """
  calculate how many bits are needed to encode a list of *value* items
  """
  return (np.ceil(np.log2(count)) + 0.5).astype(int)


def show_object_property_values(obj):
  """
  show the properties with values of an object
  """
  props = dir(obj)
  max_prop_chars = max([len(prop) for prop in props])
  hline = markerline('=', text=" OBJECT CONTENT ", doprint=True)
  max_line_chars = len(hline)
  print_in_columns(['attribute', 'value'], sep=' | ',
                   colwidths=np.array([max_prop_chars, max_line_chars - max_prop_chars]))
  # for prop in props:
  #   value = getattr(obj, prop)
  #   print_in_columns([prop, value], sep=' | ')

  markerline('=', text=' END ', doprint=True)


def unique_nonint(values, precision=None):
  """
  give the unique numbers upto a certain precision
  """
  # make into an array
  values = arrayify(values)

  # set precision
  if precision is None:
    precision = np.spacing(values.min())

  # pick unique values via up/down method
  outvals = np.unique(np.round(values/precision))*precision

  return outvals


def xovery(x, y):
  """
  combinatorial (x;y)
  """
  result = factorial(x)/(factorial(x-y)*factorial(y))

  return result


def nanplot(*_args, **kwargs):
  """
  initiate a nan plot for updating/animations.
  This plot will show up as empty, however it contains NaN values, so it is not empty

  arguments:
  ----------
  *_args : array-like
           List of arguments to are passed to plt.plot function
  **kwargs : dict
             keyword arguments dictionary to be passed to the plt.plot() function

  returns:
  --------
  ln : line oject
  """
  if 'ax' in kwargs:
    ax = kwargs.pop('ax')
  else:
    ax = plt.gca()

  if isinstance(_args[0], (np.integer, int)):
    xs = np.r_[:_args[0]]
  else:
    xs = _args[0]

  ln, = ax.plot(xs, np.nan*np.ones_like(xs, dtype=float), *_args[1:], **kwargs)

  return ln


def split_complex(cplx, sfx=1, sfy=1):
  """
  split a complex point in real and imaginary parts with a scale factor

  arguments:
  ----------
  cplx : (ndarray of) complex value(s)
         The complex value or array of to be split
  sfx: numerical value, default=1
       scale factor for the real part
  sfy: number, default=1
       scale factor for the imaginary part

  returns:
  --------
  (re, im): 2-tuple of floats
            The scaled real and imaginary parts of the complext value
  """
  return sfx*np.real(cplx), sfy*np.imag(cplx)


def monospace(array, delta=None):
  """
  straighten an array which has rounding errors

  arguments:
  ----------
  array: ndarray of floats
         the almost monospaced array
  delta: [None | float], default=None
         The delta value for the monospaced grid. If *None* it is calculated as the median
         difference value

  returns:
  --------
  array : ndarray of floats
          The resulting monospaced grid
  """
  if delta is None:
    delta = np.median(np.diff(array))

  array = np.r_[:array.size]*delta

  return array


def get_closest_index(value_wanted, values, suppress_warnings=False):
  """
  get the index of the value closest to the wanted value in an array

  arguments:
  ----------
  value_wanted : number
                 value to be found in the array
  values: ndarray
          array in which to search
  suppress_warnings: bool, default=False
                     flag indicating if a warning must be issued when no exact match can be found

  returns:
  --------
  ifnd : integer
         The index where in the array the wanted value is found; or the closest value
  """
  ifnd_arr = np.argwhere(np.isclose(values, value_wanted)).ravel()
  if ifnd_arr.size > 0:
    ifnd = ifnd_arr.item()
  else:
    # get the closest
    ifnd = np.argmin(np.abs(values - value_wanted)).ravel()[0]
    if not suppress_warnings:
      warnings.warn(f"There is no `exact` match for value = {value_wanted}. "
                    + f"Taking the closest value = {values[ifnd]}")

  return ifnd


def substr2index(substring, strlist):
  """
  find the indices of a certain substring
  """
  index = np.argwhere([substring.lower() in elm.lower() for elm in strlist]).item()

  return index


def listify(input_):
  """
  make into a list
  """
  if isinstance(input_, list):
    return input_

  return make_array_like(input_, 'list')


def tuplify(input_):
  """
  make into a tuple
  """
  if isinstance(input_, tuple):
    return input_

  return make_array_like(input_, 'tuple')


def arrayify(input_):
  """
  make into array
  """
  if isinstance(input_, np.ndarray):
    return input_

  return make_array_like(input_, 'np.ndarray')


def make_array_like(input_, array_like):
  """
  make an input into an array like
  """
  output = input_

  # from single elements to array of 1 element
  if np.ndim(input_) == 0:
    output = [input_,]
  # elif np.ndim(input_) > 1:
  #   output = output.ravel()

  # convert to the right type
  if array_like == 'list' or array_like == 'np.ndarray':
    output = [*output,]
    if array_like in ['array', 'ndarray', 'np.ndarray', 'numpy.ndarray']:
      # check if kthe elements are of the same type
      types = check_types_in_array_like(output)
      # all same type
      if len(types) == 1:
        dtype = types.pop()
      # if different types -> dtype=object
      else:
        dtype = np.object_
      output = np.array(output, dtype=dtype)
  elif array_like == 'tuple':
    output = (*output,)
  else:
    raise ValueError(f"The array_like given ({array_like}) is not valid")

  return output


def check_types_in_array_like(array_like):
  """
  check the type of all elements in an array-like
  """
  types = set()
  [types.add(type(elm)) for elm in array_like]  # pylint: disable=expression-not-assigned

  return types


def _color_vector_simple(nof_points, start_color, end_color):
  """
  create a color vector between two colors
  """
  # convert to vectors
  start_color = np.array(to_rgb(start_color))
  end_color = np.array(to_rgb(end_color))

  r = np.linspace(start_color[0], end_color[0], nof_points)
  g = np.linspace(start_color[1], end_color[1], nof_points)
  b = np.linspace(start_color[2], end_color[2], nof_points)

  cvec = np.vstack((r, g, b)).T

  return cvec


def color_vector(nof_points, c1, c2, cints=None, icints='equal'):
  """
  define a color vector with a via in the center somewhere
  """
  # handle corner case: simple straight color vector
  if cints is None:
    return _color_vector_simple(nof_points, c1, c2)

  # process some shortcuts for the intermediate colors
  if isinstance(cints, str):
    cints_list = [np.array(to_rgb(cints))]
  elif isinstance(cints, (tuple, list)):
    # check the elements
    if np.isscalar(cints[0]):
      if isinstance(cints[0], str):
        cints_list = []
        for cint in cints:
          cints_list.append(np.array(to_rgb(cint)))
      else:  # it is a single value which means cints is [r, g, b] format
        cints_list = [cints]
    else:
      cints_list = [to_rgb(cint) for cint in cints]

  nof_ints = len(cints_list)
  cs = [to_rgb(c1)] + cints_list + [to_rgb(c2)]
  nof_cs = len(cs)

  # handle the positions of the elements
  if icints == 'equal':
    ics = np.linspace(0, nof_points, nof_cs)

  # process the indices of the intermediates
  elif isinstance(icints, (tuple, list, np.ndarray)):
    ics = [0]
    for iint in range(nof_ints):
      if icints[iint] < 1:
        ics.append(nof_points*icints[iint])
      else:
        ics.append(icints[iint])
    ics.append(nof_points - 1)

  # make them all integers
  ics = [np.int_(-0.5 + ic) for ic in ics]

  # paste all parts together
  cvec = np.nan*np.ones((nof_points, 3), dtype=np.float_)

  # loop all parts using *color_vector*
  for ipart in range(0, nof_cs-1):
    nof_points_part = ics[ipart+1] - ics[ipart]
    cvec_part = color_vector(nof_points_part+1, cs[ipart], cs[ipart+1])

    # note the indices
    ifrom = ics[ipart]
    ito = ics[ipart+1]  # this is NOT included!!
    # last color is excluded, will the the first of the next section
    cvec[ifrom:ito, :] = cvec_part[:-1, :]

  # add the final color
  cvec[-1, :] = cs[-1]

  return cvec


def strip_all_spaces(strarrlike_in):
  """
  strip of all spaces from the strings in a list
  """
  strarrlike_out = [string.strip() for string in strarrlike_in]

  if isinstance(strarrlike_in, np.ndarray):
    strarrlike_out = np.array(strarrlike_out, dtype=strarrlike_in.dtype)
  elif isinstance(strarrlike_in, tuple):
    strarrlike_out = tuple(strarrlike_out)

  return strarrlike_out


def select_file(**options):
  """
  select a file to open

  Arguments:
  ----------
  **options : dictionary of keyword arguments, which are:
    defaultextension : str
                       The extension to append
    filetype : list of str
               The file types to list
    initialdir : str
                 The folder to start in
    initialfile: str
                 The initial file name without the extension
    title: The title of the window

  Returns:
  --------
  filename : str
             The file name with the full path attached as a single string
  """
  root = tk.Tk()
  root.withdraw()

  filename = tk.filedialog.askopenfilename(**options)

  return filename


def ask_question(question, **options):
  """
  select a file to open

  Arguments:
  ----------
  **options : dictionary of keyword arguments, which are:
    defaultextension : str
                       The extension to append
    filetype : list of str
               The file types to list
    initialdir : str
                 The folder to start in
    initialfile: str
                 The initial file name without the extension
    title: The title of the window

  Returns:
  --------
  filename : str
             The file name with the full path attached as a single string
  """
  root = tk.Tk()
  root.geometry("10x10+400+500")
  root.withdraw()
  root.lift()

  filename = tk.messagebox.askyesno("I have a question", question, parent=root, **options)

  root.destroy()
  return filename


def select_savefile(defaultextension=None, title=None, initialdir=None, initialfile=None,
                    filetypes=None):
  """
  select a file to save

  Arguments:
  ----------
  **options : dictionary of keyword arguments, which are:
    defaultextension : str
                       The extension to append
    filetype : list of str
               The file types to list
    initialdir : str
                 The folder to start in
    initialfile: str
                 The initial file name without the extension
    title: The title of the window

  Returns:
  --------
  filename : str
             The file name with the full path attached as a single string
  """
  root = tk.Tk()
  root.withdraw()

  if filetypes is None:
    filetypes = [("text files", "*.txt"),
                 ("JPG files", "*.jpg"),
                 ("PNG files", "*.png"),
                 ("GIF files", "*.png"),
                 ("All files", "*.*")]

  # while True:
  filename = tk.filedialog.asksaveasfilename(defaultextension=defaultextension,
                                          initialdir=initialdir, title=title,
                                          initialfile=initialfile, filetypes=filetypes)

  # if check_exists:
  #   if os.path.exists(filename):
  #     answer = dinput('The file already exists. Overwrite? [y/n]', 'y')
  #     if answer[0].lower() == 'y':
  #       # remove
  #       os.remove(filename)
  #       break
  #     elif answer[0].lower() == 'n':
  #       # do nothing and re-ask for filename
  #       pass
  #     else:
  #       print('answer given ({}) not understood. Please select `y` or `n`'.format(answer))
  #   else:
  #     break
  # else:
  #   break

  return filename


def select_folder(**options):
  """
  select a folder. It's a wrapper arouind filedialog from tkinter module

  **options are:
    parent - the window to place the dialog on top of
    title - the title of the window
    initialdir - the directory that the dialog starts in
    initialfile - the file selected upon opening of the dialog
    filetypes - a sequence (list) of (<label>, <pattern>)-tuples in which the '*' wildcard is
                allowed
    defaultextension - the default extension to append to file (save dialogs only)
    multiple - when True, selection of multiple items is allowed (default=False)
  """
  root = tk.Tk()
  root.withdraw()

  dirname = filedialog.askdirectory(**options)

  root.quit()
  root.destroy()

  return dirname


def val2ind(pos, spacing=None, center=False):
  """ get the index where the value is found """

  posm = pos - pos.min()

  if spacing is None:
    # find minimum distnace first (excluding 0!)
    dposs = np.diff(np.sort(posm))
    izero = np.isclose(dposs, 0.)
    spacing = dposs[~izero].min()

  ind = posm/spacing
  if center is True:
    ind -= ind.mean()

  return ind


def calc_frequencies(nof_taps, fs, center_zero=True):
  '''
  Calculate the frequencies belonging to a spectrum based on sample frequency and number of tabs

  Positional arguments:
  ---------------------
  nof_taps : int
             The number of frequencies/taps in the DFT
  fs : float
       The sample frequency in Hz

  Keyword argument:
  -----------------
  center_zero : bool, default=True
               If True apply fftshift such that the center frequency is 0

  Returns:
  --------
  freqs : ndarray of floats
          An array containing the frequencies of the spectrum
  '''

  # subtract to get the number of intervals/steps
  nof_steps = nof_taps
  if center_zero:
    taps = np.arange(-nof_steps//2, nof_steps//2)
  else:
    taps = np.arange(nof_steps)

  freqs = taps*(fs/nof_steps)

  return freqs

# pylint: disable-next=C0103
def spectrum(signal, fs=1., nof_taps=None, scaling=1., center_zero=True, full=True,
             dB=True, makeplot=True, yrange=None, plotspec='b.-', title='auto', **plot_kwargs):
  """
  get the spectrum of a signal
  """
  plot_kwargs_ = dict(ax=None)
  plot_kwargs_.update(plot_kwargs)

  signal = signal.reshape(-1)
  if nof_taps is None:
    nof_taps = signal.size

  freqs_unscaled = calc_frequencies(nof_taps, fs=fs, center_zero=center_zero)

  # overwrite freqs_base by the scaled version
  freqs, sidict = scale_by_si_prefix(freqs_unscaled, base_pref_on_what="max")

  nof_bins = freqs.size
  freq_per_bin = fs/nof_bins
  figtitle = f"1:{nof_bins/signal.size} spectrum, {format_as_si(freq_per_bin)}Hz per bin"

  spect_ = np.fft.fft(signal, n=nof_taps)
  spect = np.abs(spect_)
  if center_zero:
    spect = fftshift(spect)

  if isinstance(scaling, str):
    if scaling == 'default':
      sf = 1.
    elif scaling == 'per_sample':
      sf = nof_taps
    elif scaling.startswith('normalize'):
      sf = np.abs(spect).max()
      figtitle += ", normalized"
    elif scaling.startswith('bw'):
      # estimate the bandwidth
      sf_thres = 0.25
      ampmax = max(spect)
      threshold = ampmax*sf_thres
      tf_valid = spect >= threshold
      sf = np.median(np.abs(spect[tf_valid]))
    else:
      raise NotImplementedError(f"The value for *scaling={scaling}* is not implemented")
  elif isinstance(scaling, (list, tuple)):
    bw_start, bw_end = scaling
    tf_valid_freqs = (freqs >= bw_start/sidict['sf'])*(freqs <= bw_end/sidict['sf'])
    sf = np.median(np.abs(spect[tf_valid_freqs]))
  else:
    sf = np.float_(scaling)
    figtitle += f", {sf:0.1f} scaling"

  spect_ /= sf
  spect /= sf

  if not full:
    figtitle += ", half spectrum"
    nof_samples = freqs.size
    if center_zero:
      freqs = freqs[nof_samples//2 + 1:]
      spect = spect[nof_samples//2 + 1:]
    else:
      freqs = freqs[:nof_samples//2]
      spect = spect[:nof_samples//2]

  if dB:
    spect = logmod(spect)
    if isinstance(scaling, str) and scaling.startswith('normalize'):
      yscale = "Power [dBc]"
    else:
      yscale = "Power [dB]"

    # calculate the yscaling
    ymax = max(spect) + 3.
    if yrange is not None:
      ymin = ymax - yrange
  else:
    yscale = "Power [lin]"

  ax = plot_kwargs_.pop('ax')
  if makeplot:
    # plot the stuff

    if ax is None:
      fig = plt.figure(figname(figtitle))
      ax = fig.add_subplot(111)

    ax = qplot(ax, freqs, spect, plotspec, **plot_kwargs_)
    if isinstance(title, str) and title == 'auto':
      title = figtitle
    ax.set_title(title)
    ax.set_xlabel(f"Frequency [{sidict['sym']}Hz]")
    ax.set_ylabel(yscale)
    if yrange is not None:
      ax.set_ylim(top=ymax, bottom=ymin)  # pylint: disable=E0601
    plt.show(block=False)
    plt.draw()

  return freqs_unscaled, fftshift(spect_), ax


def find_dominant_frequencies(signal, fs, f1p=None, scaling='default', max_nof_peaks=None,
                              min_rel_height_db=10, makeplot=False, **plotkwargs):
  """
  find the frequencies in the spectrum of a signal

  Arguments:
  ----------
  signal : ndarray of floats
           The signal in floats
  fs : scalar
       The sample frequency in Hz
  f1p : scalar or None, default=None
        The 1P frequency. If None, f1p=1.0 will be given. Only if f1p != 1, the plots are adjusted
  max_nof_peaks : [ None | int], default=None
                  The maximum number of peaks to be returned
                  None implies all peaks are returned and is effectively the same as np.inf
  min_rel_height_db : scalar, default=10
                      The minimimum relative height from the main spectral frequency above which
                      additional peaks can be found. the sign is of no importance and will be
                      corrected for
  makeplot : bool, default=False
         Whether to plot the spectrum and the peaks superimposed
  scaling : [ 'default' | 'normalize' | 'per_sample' ], default='default'
                 The fourier transform scaling. These are:
                 - 'default': simple FFT, no scaling
                 - 'normalize': the main frequency is scaled to 0 dB
                 - 'per_sample': the scaling is equal to (1/nof_samples)

  Returns:
  --------
  fpeaks : ndarray of floats
           An array containing the peak frequencies in Hz or xP's, depending on the value for
           *f1p*
  peakvals : ndarray of floats
             The peak values, they take the value of the *scaling* into account


  """
  if f1p is None:
    f1p = 1.0

  # corner case: no signal at all
  if np.isclose(rms(signal), 0.):
    return np.array([])

  # corner case: constant signal
  if np.isclose(np.std(signal), 0.):
    return np.array([0.])

  nof_samples = signal.size
  fs_ = fs/f1p

  signal_unbias = signal - np.mean(signal)
  freqs, Ydb_debias, _ = spectrum(signal_unbias, fs=fs_, center_zero=False, scaling='default',
                                  full=False, makeplot=False, dB=True)

  # calculate the minimum distance between samples
  dP_per_sample = fs_/(2.*nof_samples)
  distance_in_P = 0.4
  distance_in_samples = 1 + int(0.5 + distance_in_P/dP_per_sample)

  # find the peaks taking the distance of P/2 into account
  height = Ydb_debias.max() - np.abs(min_rel_height_db)
  ipeaks = find_peaks(Ydb_debias, height=height, distance=distance_in_samples)[0]

  # remove peaks in the first interval [0, P/2]
  idel = np.argwhere(ipeaks < distance_in_samples)
  ipeaks = np.delete(ipeaks, idel)

  isort = np.argsort(Ydb_debias[ipeaks])[-1::-1]
  ipeaks_sorted = ipeaks[isort]

  nfnd = ipeaks_sorted.size
  if max_nof_peaks is None:
    max_nof_peaks = nfnd
  else:
    max_nof_peaks = np.fmin(nfnd, max_nof_peaks)

  inds = ipeaks_sorted[:max_nof_peaks]
  fpeaks = freqs[inds]
  peakvals = Ydb_debias[inds]

  if peakvals.max() <= (db(signal.sum()) + np.abs(min_rel_height_db)):
    fpeaks = np.insert(fpeaks, 0, 0)
    peakvals = np.insert(peakvals, 0, db(signal.sum()))
    # sort
    isort = np.argsort(fpeaks)
    fpeaks = fpeaks[isort]
    peakvals = peakvals[isort]

  if scaling.endswith('normalize'):
    offset = np.max(peakvals)
  elif scaling.endswith('per_sample'):
    offset = db(signal.size)
  elif scaling.endswith('default'):
    offset = 0.
  else:
    raise NotImplementedError("The *scaling={}* is not implemented (yet)")

  peakvals -= offset

  if makeplot:
    if 'ax' in plotkwargs:
      ax = plotkwargs['ax']
    else:
      fig = plt.figure(figname("spectrum"))
      ax = fig.add_subplot(111)
      ax.set_title("The spectrum and the found peaks")
      if np.isclose(f1p, 1.0):
        ax.set_xlabel("Frequency [Hz]")
      else:
        ax.set_xlabel("relative frequecy [xP]")

      if scaling.endswith("normalize"):
        ax.set_ylabel("Power [dBc]")
      else:
        ax.set_ylabel("Power [dB]")

    xs, ys, _ = spectrum(signal, fs=fs_, center_zero=False, scaling=scaling, full=False,
                         dB=True, makeplot=False)

    ax.plot(xs, Ydb_debias - offset, 'k--')
    ax.plot(xs, ys, 'b-')
    # plot_spectrum(signal, 'b-', fs=fs_, center_zero=False, scaling=scaling, full=False,
    #               ax=ax)
    ax.plot(fpeaks, peakvals, 'ro', mfc='none')
    threshold = ys.max() - np.abs(min_rel_height_db)
    ax.axhline(threshold, color='g', linestyle='--')
    ax.text(xs[-1], threshold, f"threshold @ {float(threshold):0.1f} dBc",
            va='bottom', ha='right')
    plt.show(block=False)
    plt.draw()

  return fpeaks, peakvals


def print_struct_array(arr, varname='', prefix='| ', verbose=True,
                       output_in_list=False, flat=False):
  '''
  Print the content of a structured array.

  Positional arguments:
  ---------------------
  arr : ndarray
        The structured array of which the contents must be displayed.

  Keyword arguments:
  ------------------
  varname : str, default='root'
            The name of the variable
  prefix : str, default=': '
           A formatting prefix which might help to distinguish different levels in the array
  verbose : bool, default: True
            Flag indicating whether to fully display the structure (i.e., expand the folded
            substructures) or not.
  start_from : None or str, default=None
               Indicates after which line the output must be generated
  end_at : None or str, default=none
           Indicates after which string the output generation should stop

  Returns:
  --------
  None

  See Also:
  ---------
  ._print_struct_array_flat_full : prints the array content with flag *verbose=False*
  ._print_struct_array_full : prints the array content with flag *verbose=False*
  ._print_struct_array_flat_compact : prints the array content with flag *verbose=False*
  ._print_struct_array_compact : prints the array content with flag *verbose=True*
  '''

  # print(varname, end='')
  if output_in_list:
    output_array = ['']
  else:
    output_array = None

  if verbose:
    if flat:
      _print_struct_array_flat_full(arr, substr=varname, output_array=output_array)
    else:
      _print_struct_array_full(arr, prefix=prefix, linecount=0, output_array=output_array)

  else:
    if flat:
      _print_struct_array_flat_compact(arr, substr=varname, is_singleton=True,
                                       output_array=output_array)
    else:
      _print_struct_array_compact(arr, prefix=prefix, is_singleton=True,
                                  linecount=0, output_array=output_array)

  return output_array


def _print_struct_array_compact(arr, prefix='| ', level=1, linecount=0,
                                is_singleton=True, output_array=None, **singinfo):
  '''
  Print the structured array structure and field names. In case the values are scalars or strings
  this value is displayed. Otherwise the shape and the data type are shown.

  Note: this is a fully recursive function!

  Arguments:
  ----------
  arr : ndarray
        The structured array of which the contents must be displayed. Due to the recursive nature
        this may be a subarray of the original array

  Keyword arguments:
  ------------------
  prefix : str, default: ': '
           A formatting prefix which might help to distinguish different levels in the array
  level : int
          The level of indentation of the current subarray
  is_singleton : bool, default: True
              This indicates whether a scalar value or string is a "singleton". This implies that
              there exists only one field of this name. If *True* the value/string is printed.
  singinfo : dict or None, default: None
             Dict containing info on the last singleton found

  Returns:
  --------
  *None*
  '''

  linecount += 1
  is_leave = True

  # check if it is a arr or a leave
  if type(arr) in [np.void, np.ndarray]:
    if arr.dtype.names is not None:
      is_leave = False
      if arr.size > 1:
        is_singleton = False

      shp = arr.shape
      _print(f'{shp}', output_array=output_array)

      # loop all subs
      for name in arr.ravel()[0].dtype.names:
        if output_array is not None:
          _print(prefix*(level) + name, end='', output_array=output_array)
        else:
          _print(f'{linecount:3d}. ' + prefix*level + name, end='',
                 output_array=output_array)

        subarr = arr.ravel()[0][name]

        if is_singleton:
          singinfo['line'] = linecount
          singinfo['name'] = name

        linecount = _print_struct_array_compact(subarr, prefix=prefix,
                                                level=level + 1,
                                                linecount=linecount,
                                                is_singleton=is_singleton,
                                                output_array=output_array,
                                                **singinfo)

  # it is a leave!
  if is_leave:
    # NUMERICAL
    if isinstance(arr, (float, np.floating)):
      type_ = type(arr).__name__
      if is_singleton:
        _print(f": {arr:.2} ({type_})", output_array=output_array)
      else:
        _print(f": ... ({type_}) ({singinfo['name']} @ {singinfo['line']})",
               output_array=output_array)

    # INT
    elif isinstance(arr, (np.integer, int)):
      type_ = type(arr).__name__
      if is_singleton:
        _print(f': {arr:d} ({type})', output_array=output_array)
      else:
        _print(f': ... ({type_}) ({singinfo["name"]} @ {singinfo["line"]})',
               output_array=output_array)

    # STRING
    elif isinstance(arr, str):
      type_ = 'str'
      if is_singleton:
        _print(f': "{arr}" ({type_})'.format(arr, type_), output_array=output_array)
      else:
        _print(f': ... ({type}) ({singinfo["name"]} @ {singinfo["line"]})',
               output_array=output_array)

    # NDARRAY
    elif isinstance(arr, np.ndarray):
      type_ = type(arr.ravel()).__name__
      if is_singleton:
        if isinstance(arr[0], (float, complex, np.floating, np.complexfloating)):
          _print(f": {subset_str(arr, '{:0.2}')} ({arr.shape}{type_}) ", output_array=output_array)
        else:
          _print(f': {subset_str(arr)} ({arr.shape}{type_}) ', output_array=output_array)
      else:
        _print(f": ... ({arr.shape}{type_}) ({singinfo['name']} @ {singinfo['line']:d})",
               output_array=output_array)

    # MATLAB FUNCTION
    elif type(arr).__name__ == 'MatlabFunction':
      if is_singleton:
        _print(f": <A MATLAB function> ({type(arr).__name__})", output_array=output_array)
      else:
        _print(f": ... ({type(arr).__name__}) ({singinfo['name']} @ {singinfo['line']})",
               output_array=output_array)

    # NOT YET DEFINED STUFF WILL RAISE AN EXCEPTION
    else:
      raise TypeError('The type is not expected')

  return linecount


def _print_struct_array_flat_compact(arr, substr='<var>', is_singleton=True, output_array=None,
                                     linecount=0):
  '''
  Print the structured array structure and field names. In case the values are scalars or strings
  this value is displayed. Otherwise the shape and the data type are shown.

  Note: this is a fully recursive function!

  Arguments:
  ----------
  arr : ndarray
        The structured array of which the contents must be displayed. Due to the recursive nature
        this may be a subarray of the original array

  Keyword arguments:
  ------------------
  prefix : str, default: ': '
           A formatting prefix which might help to distinguish different levels in the array
  level : int
          The level of indentation of the current subarray
  is_singleton : bool, default: True
              This indicates whether a scalar value or string is a "singleton". This implies that
              there exists only one field of this name. If *True* the value/string is printed.
  singinfo : dict or None, default: None
             Dict containing info on the last singleton found

  Returns:
  --------
  *None*
  '''

  is_leave = True
  # check if it is a arr or a leave
  if type(arr) in [np.void, np.ndarray]:
    if arr.dtype.names is not None:
      is_leave = False
      if arr.size > 1:
        is_singleton = False

      shp = arr.shape
      substr += f"{shp}."

      # loop all subs
      for name in arr.ravel()[0].dtype.names:
        subarr = arr.ravel()[0][name]

        linecount = _print_struct_array_flat_compact(subarr, substr=substr + name,
                                                     is_singleton=is_singleton,
                                                     output_array=output_array,
                                                     linecount=linecount)

  # it is a leave!
  if is_leave:
    # NUMERICAL
    if isinstance(arr, (float, complex, np.floating, np.complexfloating)):
      type_ = type(arr).__name__
      if is_singleton:
        endstr = f': {arr:.2} ({type_})'
      else:
        endstr = f': ... ({type_})'

    # INT
    elif isinstance(arr, int):
      type_ = type(arr).__name__
      if is_singleton:
        endstr = f': {arr:d} ({type_})'
      else:
        endstr = f': ... ({type_})'

    # STRING
    elif isinstance(arr, str):
      type_ = 'str'
      if is_singleton:
        endstr = f': "{arr}" ({type_})'
      else:
        endstr = f': ... ({type_})'

    # NDARRAY
    elif isinstance(arr, np.ndarray):
      type_ = type(arr.ravel()).__name__
      if is_singleton:
        if isinstance(arr[0], (float, complex, np.floating, np.complexfloating)):
          endstr = f": {subset_str(arr, '{:0.2}')} ({type_}{arr.shape}) "
        else:
          endstr = f": {subset_str(arr)} ({type_}{arr.shape}) "
      else:
        endstr = f": ... ({type_}{arr.shape})"

    # MATLAB FUNCTION
    elif type(arr).__name__ == 'MatlabFunction':
      if is_singleton:
        endstr = f": <A MATLAB function> ({type(arr).__name__})"
      else:
        endstr = f": ... ({type(arr).__name__})"

    # NOT YET DEFINED STUFF WILL RAISE AN EXCEPTION
    else:
      raise TypeError('The type is not expected')

    if output_array is not None:
      output_array.append(substr + endstr)
    else:
      print(f"{linecount:d}. " + substr + endstr)

  return linecount + 1


def _print_struct_array_full(arr, prefix='| ', level=1, linecount=0, output_array=None):
  '''
  Print the structured array structure and field names. This will show all values and expand all
  folded structures in *print_struct_array* function.

  Note: this is a fully recursive function!

  Arguments:
  ----------
  arr : ndarray
        The structured array of which the contents must be displayed. Due to the recursive nature
        this may be a subarray of the original array

  Keyword arguments:
  ------------------
  prefix : str, default: ': '
           A formatting prefix which might help to distinguish different levels in the array
  level : int
          The level of indentation of the current subarray

  Returns:
  --------
  *None*
  '''

  is_leave = True
  linecount += 1
  # check if it is a arr or a leave
  if type(arr) in [np.void, np.ndarray]:
    if arr.dtype.names is not None:
      is_leave = False
      shp = arr.shape
      _print(f'{shp}', output_array=output_array)

      # loop all subs
      for isub, subarr in enumerate(arr.ravel()):
        for name in subarr.dtype.names:
          if output_array is not None:
            prefix_this = level*prefix
          else:
            prefix_this = f"{linecount:3d}. " + level*prefix
          if arr.ravel().size > 1:
            prefix_this = prefix_this + f"[{isub:02d}] "

          _print(prefix_this + name, end='', output_array=output_array)
          linecount = _print_struct_array_full(subarr[name], prefix=prefix, level=level+1,
                                               linecount=linecount, output_array=output_array)

  # it is a leave!
  if is_leave:
    # _print(type(arr))
    if isinstance(arr, (float, int, complex)):
      type_ = type(arr).__name__
      _print(f": {arr} ({type_})", output_array=output_array)
    elif isinstance(arr, np.ndarray):
      type_ = type(arr.ravel()).__name__
      if arr.ndim == 0:
        _print(f": {arr[()]} ({arr.ndim:d}D {type_}", output_array=output_array)
      else:
        if arr.size < 5:
          _print(f": {arr} ({arr.ndim:d}D {type_})", output_array=output_array)
        else:
          _print(f": {arr.shape} ({arr.ndim:d}D {type_})", output_array=output_array)
    elif isinstance(arr, str):
      type_ = 'str'
      _print(f': "{arr}" ({type_})', output_array=output_array)
    elif type(arr).__name__ == 'MatlabFunction':
      _print(f": A MATLAB function ({type(arr).__name__})", output_array=output_array)
    else:
      raise TypeError('The type is not expected')

  return linecount


def _print_struct_array_flat_full(arr, substr='<var>', output_array=None, linecount=0):
  '''
  Print the structured array structure and field names. This will show all values and expand all
  folded structures in *print_struct_array* function.

  Note: this is a fully recursive function!

  Arguments:
  ----------
  arr : ndarray
        The structured array of which the contents must be displayed. Due to the recursive nature
        this may be a subarray of the original array

  Keyword arguments:
  ------------------
  prefix : str, default: ': '
           A formatting prefix which might help to distinguish different levels in the array
  level : int
          The level of indentation of the current subarray

  Returns:
  --------
  *None*
  '''

  is_leave = True
  # check if it is a arr or a leave
  if type(arr) in [np.void, np.ndarray]:
    if arr.dtype.names is not None:
      is_leave = False
      shp = arr.shape
      substr += f"{shp}"

      # loop all subs
      for isub, subarr in enumerate(arr.ravel()):
        for name in subarr.dtype.names:
          if arr.ravel().size > 1:
            prefix_this = f"[{isub:02d}]."
          else:
            prefix_this = '.'

          # print(prefix_this + name, end='')
          linecount = _print_struct_array_flat_full(subarr[name],
                                                    substr=substr + prefix_this + name,
                                                    output_array=output_array, linecount=linecount)

  # it is a leave!
  if is_leave:
    if isinstance(arr, (float, int, complex)):
      type_ = type(arr).__name__
      endstr = f": {arr} ({type_})"
    elif isinstance(arr, np.ndarray):
      type_ = type(arr.ravel()).__name__
      if arr.ndim == 0:
        endstr = f": {arr[()]} ({arr.ndim:d}D {type_}"
      else:
        if arr.size < 5:
          endstr = f": {arr} ({arr.ndim:d}D {type_})"
        else:
          endstr = f": {arr.shape} ({arr.ndim:d}D {type_})"
    elif isinstance(arr, str):
      type_ = 'str'
      endstr = f": '{arr}'' ({type_})"
    elif type(arr).__name__ == 'MatlabFunction':
      endstr = f": A MATLAB function ({type(arr).__name__})"
    else:
      raise TypeError('The type is not expected')

    if output_array is not None:
      output_array.append(substr + endstr)
    else:
      print(f"{linecount:d}. " + substr + endstr)

  return linecount + 1


def _print(input_str, sep=' ', end='\n', file=sys.stdout, flush=False, show=True,
           output_array=None):
  """
  print line to array or to screen
  """
  if output_array is None:
    if show:
      print(input_str, sep=sep, end=end, file=file, flush=flush)
    return None
  else:  # place in output array
    output_array[-1] += input_str
    if end == '\n':
      output_array.append('')

    return None


def subset_str(arr, fmt='{}', nof_at_start=2, nof_at_end=1):
  '''
  Creates a string printing a subset of the array in case of too many elements

  Note that if the number of elements is equal to or smaller than nof_at_start + nof_at_end + 1,
  the function will return an array similar to the generic print(arr) output

  Positional argument:
  --------------------
  arr : array-like
        an array-like containing a set of values or strings (maybe whatever)

  Keyword arguments:
  ------------------
  fmt : str, default='{}'
        The formatting string with which to display the values
  nof_at_start : int, default=2
                 The number of elements to plot at the beginning
  nof_at_end : int, default=1
               The number of elements to display at the end

  Returns:
  --------
  subset_str : str
               The output is a formatted string of a subset of the input array

  '''
  min_req_elms = nof_at_start + nof_at_end + 1
  if isinstance(arr, (list, tuple)):
    arrsize = len(arr)
  elif isinstance(arr, np.ndarray):
    arrsize = arr.size
  else:
    raise TypeError(f"The type of 'arr' is {type(arr)}, which is not implemented!")

  if arrsize <= min_req_elms:
    subset_str_new = '[' + ', '.join([fmt.format(elm) for elm in arr]) + ']'
  else:
    subarr_start = ', '.join([fmt.format(elm) for elm in arr[:nof_at_start]])
    subarr_end = ', '.join([fmt.format(elm) for elm in arr[-nof_at_end:]])

    subset_str_new = '[' + subarr_start + ',... ,' + subarr_end + ']'

  return subset_str_new


def extent(xvec, yvec):
  '''
  creates a 4-list of floats to be used in the *extent* keyword when using *imshow* or *matshow*

  *extent* creates the 4-list according to the specified format [xmin, xmax, ymin, ymax] with the
  addition of halve a pixel to ensure proper alignment between axes and pixel centers.

  See the help for *imshow*, keyword *extent* for more information on this padding problem

  Arguments:
  ----------
  xvec : ndarray of floats
         A 1D or 2D array of floats containing the x positions of the data
  yvec : ndarray of floats
         A 1D or 2D array of floats containing the y positions of the data

  Returns:
  --------
  ext : 4-list of floats
        contains the values for [xmin, xmax, ymin, ymax] plus a padding of halve a pixel.

  See Also:
  ---------
  matplotlib.pyplot.imshow : show an image in an axes
  matplotlib.pyplot.matshow : show a matrix in an axes
  '''
  if xvec.ndim > 1:
    xvec = xvec[0, :]

  if yvec.ndim > 1:
    yvec = yvec[:, 0]

  dx = np.diff(xvec).mean()
  dy = np.diff(yvec).mean()

  ext = [xvec[0] - dx/2, xvec[-1] + dx/2, yvec[0] - dy/2, yvec[-1] + dy/2]

  return ext


def normalize(data):
  '''
  normalizes the data in the array to the interval [0, 1]

  Argument:
  ---------
  data : array-like or ndarray of floats
         The data to normalize

  Returns:
  --------
  out : array-like or ndarray of floats
        Same type and shape as *data*, but with values normalized to the interval [0, 1]
  '''

  input_ = deepcopy(data)
  dtype = type(input_)
  if dtype in [list, tuple]:
    input_ = np.array(input_)

  input_ -= input_.min()
  input_ /= input_.max()

  if dtype is list:
    out = list(input_)
  elif dtype is tuple:
    out = tuple(input_)
  else:
    out = input_

  return out


def inputdlg(strings, defaults=None, types=None, windowtitle='Input Dialog'):
  '''
  Creates a window with a set of input entry boxes plus text

  Arguments:
  ----------
  strings : (list of) str
            a list of strings to display before the entry boxes. A None value omits the string
            completely
  defaults : (list of) any type, optional
             A set of defaults which is either None (indicating no defaults whatsoever) or of
             equal length as *strings*.
  types : (list of) type designations, optional
          A list of types to which the str types are cast before being returned. None implies all
          string objects.

  Returns:
  --------
  A list of values  of any type possible equal to what was type in cast to the correct type if
  requested
  '''

  def pressed_return():
    master.quit()

    return None

  if isinstance(strings, (list, tuple)):
    nof_rows = len(strings)
    if defaults is None:
      defaults = [None]*nof_rows
    if types is None:
      types = [str]*nof_rows
  else:  # else: single row
    nof_rows = 1
    strings = [strings]
    defaults = [defaults]
    if types is None:
      types = str
    types = [types]

  master = tk.Tk(className=windowtitle)
  tkvar = []
  entry = []
  for irow in np.arange(nof_rows):

    # create tkvars of the correct type
    if types[irow] in (np.floating, float):
      tkvar.append(tk.DoubleVar(master, value=defaults[irow]))

    elif types[irow] in (np.integer, int):
      tkvar.append(tk.IntVar(master, value=defaults[irow]))

    elif types[irow] == str:
      tkvar.append(tk.StringVar(master, value=defaults[irow]))

    else:
      raise ValueError(f"The type '{types[irow]}' is not recognized")

    # set and grid label
    label = tk.Label(master=master, text=strings[irow])
    label.grid(row=irow, column=0)

    # set and grid entry
    entry.append(tk.Entry(master=master, textvariable=tkvar[irow]))
    entry[irow].grid(row=irow, column=1)
    entry[irow].index(0)

  # set focussing and stuff
  entry[0].selection_range(0, 1000)
  entry[0].index(0)
  entry[0].focus()
  entry[-1].bind('<Return>', pressed_return)

  # start mainloop
  master.mainloop()

  try:
    # list comprehension to get output list
    out = [tkvar[irow].get() for irow in np.arange(nof_rows)]
  except Exception:  # pylint: disable=broad-exception-caught
    print('unknown exception raised. None returned')
    return None

  if len(out) == 1:
    returnval = out[0]
  else:
    returnval = out

  master.destroy()
  return returnval


def rms(signal, axis=None, weights=None):
  '''
  Calculate the root-mean-squared value of a signal in time

  Arguments:
  ----------
  signal : ndarray of floats or complex floats
           the signal in time
  axis : None or int, default=-1
         The axis along which the RMS value must be determined. In case *None* the RMS value will
         be determined for all elements in the array
  weights : None or array-like of floats
            if not None, this array gives the weights. Note that the mean weight should be equal to
            1.0 to prevent scaling of the original data. Use with caution

  Returns:
  --------
  s_rms : float
         The rms value of the signal
  '''

  # if axis is None, take rms over entire array
  if axis is None:
    signal = signal.reshape(-1)
    axis = -1

  if weights is None:
    weights = np.ones(signal.shape, dtype=float)

  if not np.isclose(1., np.nanmean(weights)):
    warnings.warn("The weights are not averaging to unity. Scaling applied!")
    weights = weights/np.nanmean(weights)

  is_complex = np.any(np.iscomplex(signal))

  if is_complex:
    i_rms = np.sqrt(np.nanmean(np.real(signal*weights)**2, axis=axis))
    q_rms = np.sqrt(np.nanmean(np.imag(signal*weights)**2, axis=axis))
    s_rms = i_rms + 1j * q_rms
  else:
    s_rms = np.sqrt(np.nanmean((weights*signal)**2, axis=axis))

  return s_rms


def phase2snr(phase, phase_units='rad', snr_units='db', mode='calc', nof_samples=1e6):
  '''
  computates the SNR related to a certain phase error sigma

  Arguments:
  ----------
  phase : float or array-like of floats
          The phase sigma
  phase_units : ('deg') {'rad', 'deg'}
                Whether the phases are given in radians ('rad') or degrees ('deg')
  snr_units : ('snr') {'db', 'lin'}
             Whether the SNR values are given in decibells ('db') or linear ('lin')
  mode : ('calc') {'calc', 'sim'}
         How to calculate the SNR. *Calc* gives an analytical derivation, while *sim* does a
         simulation using the keyword argument *nof_samples*
  nof_samples : (1e6) int or int-like float
                the number of samples when running a simulation. For mode 'calc' this keyword is
                moot.

  Returns:
  --------
  snr : float
        The SNR value matching the phase sigma given

  '''

  if isinstance(phase, (list, tuple)):
    phase = np.array(phase)
  elif isinstance(phase, np.ndarray):
    pass
  else:
    phase = np.array(phase)

  phase = phase.astype(float)

  if phase_units == 'deg':
    phase *= np.pi / 180
  elif phase_units == 'rad':
    pass
  else:
    raise ValueError(f'The phase units "{phase_units:s}" are not accepted. '
                     + 'Only "rad" and "deg" are.')

  if mode == 'sim':
    nof_samples = int(nof_samples)
    randsamps = np.random.randn(*phase.shape, nof_samples)
    sreal = np.exp(1j * randsamps * phase.reshape(*phase.shape, 1))
    noise_power = np.real(sreal).var(axis=-1) + np.imag(sreal).var(axis=-1)

    snr = 1 / noise_power

  elif mode == 'calc':
    snr = 1 / (np.tan(np.abs(phase))**2)
  else:
    raise ValueError("the *mode* keyword argument only accepts values 'calc' and 'sim'")

  if snr_units == 'db':
    snr = 10 * np.log10(snr)
  elif snr_units == 'lin':
    pass
  else:
    raise ValueError(f'The SNR units "{snr_units:s}" are not accepted. Only "db" and "lin" are.')

  return snr


def snr2phase(snr, snr_units='db', phase_units='rad', mode='calc'):
  '''
  Computates the phase sigma related to a certain SNR

  Arguments:
  ----------
  snr : float
        The signal-to-noise value
  snr_units : {'db', 'lin'}
              Whether the snr value is given in decibells ('db') or linear ('lin')
  phase_units : {'rad', 'deg'}
                Whether the phase sigma is given in radians ('rad') or degrees ('deg')

  Returns:
  --------
  phase : float
          The phase sigma belonging to a certain input signal-to-noise ratio

  Author:
  -------
  Joris Kampman, Thales NL, 2017
  '''

  if type(snr) in [list, tuple]:
    snr = np.array(snr)
  elif isinstance(snr, np.ndarray):
    pass
  else:
    snr = np.array(snr)

  if snr_units == 'db':
    snr = 10**(snr / 10)
  elif snr_units == 'lin':
    pass
  else:
    raise ValueError(f'The SNR units "{snr_units}" are not accepted. Only "db" and "lin" are.')

  if mode == 'sim':
    raise NotImplementedError("The mode 'sim' is not implemented yet.")

  if mode == 'calc':
    phase = np.abs(np.arctan(1 / snr))
  else:
    raise ValueError("the *mode* keyword argument only accepts values 'calc' and 'sim'")

  if phase_units == 'rad':
    pass
  elif phase_units == 'deg':
    phase *= 180 / np.pi
  else:
    raise ValueError(f"The phase error units '{phase_units}' is not accepted. "
                     + "Only 'rad' and 'lin' are.")

  return phase


def ndprint(arr, fs='{:0.2f}'):
  '''
  pretty prints a ndarray according to a specific formatting rule

  Arguments:
  ----------
  arr : ndarray
        The data to pretty print
  fs : str, optional
       The format string according to generic string formatting rules

  Author:
  -------
  Joris Kampman, Thales NL, 2017
  '''

  # convert lists and tuples to arrays
  if type(arr) in [tuple, list]:
    arr = np.array(arr)

  print([fs.format(elm) for elm in arr])


def dinput(question, default, include_default_in_question=True):
  '''
  A modification to keyboard input, with a default value in case nothing is given (return only)

  Arguments:
  ----------
  question : str
             The question to ask for an input
  default : <any type>
            The default value in case no answer provided
  include_default_in_question : bool
                                show the default value in the question between brackets

  Returns:
  --------
  output : <any type>
           The output from either keyboard input or the preset default

  Author:
  -------
  Joris Kampman, Thales NL, 2017
  '''

  if include_default_in_question:
    question = question.strip()
    question = question.strip(':')
    question = f"{question:s} (default={default}): "

  answer = input(question)

  if not bool(answer):
    output = default
  else:
    output = answer

  return output


def round_to_int(floats, inttype='nearest', intloc='nearest'):
  """
  round the integers

  arguments:
  ----------
  floats : (array-like) of float(s)
  inttype : [ 'nearest' | 'odd' | 'even'], default='nearest'
            which type of integer wanted. Default is the nearest integer, but it can also be an odd
            or even number
  intloc : [ 'nearest' | 'above' | 'below'], default='nearest'
           Whether the location is below or above the floating value.

  Returns:
  --------
  ints : (array-like of) integers(s)

  """
  isscal = np.isscalar(floats)

  # make into array of floats
  floats = arrayify(floats)

  # neighbor
  nearest = np.int_(0.5 + floats)

  if inttype.startswith('nearest'):
    step = 1
    intvals = nearest
  elif inttype.startswith(('odd', 'even')):
    step = 2
    intvals = np.int_(0.5 + floats/2)*2
    if inttype.startswith('odd'):
      delta_to_nearest_odd = np.sign(floats - intvals)
      # closest odd neighbor
      intvals = np.int_(0.5 + intvals + delta_to_nearest_odd)
  else:
    raise ValueError(f"The given value for 'inttype' ({inttype}) is not valid")

  if intloc.startswith('nearest'):
    pass
  elif intloc.startswith('above'):
    tf_below = intvals < floats
    intvals[tf_below] = intvals[tf_below] + step
  elif intloc.startswith('below'):
    tf_above = intvals > floats
    intvals[tf_above] = intvals[tf_above] - step
  else:
    raise ValueError(f"the given value for 'intloc' ({intloc}) is not valid")
  # closest even neighbor

  if isscal:
    intvals = intvals.item(0)

  return intvals


def round_to_values(data, rnd2info):
  '''
  rounds a set of values to a set of allowed values

  Arguments:
  ----------
  data : ndarray of floats
         A set of values
  rnd2info : int or ndarray
             in case an integer is given, the value is the number of states allowed onto which
             to map the data
             In case of a ndarray vector, this vector represents the allowed states onto which
             to map the data

  Returns:
  --------
  A ndarray of the same shape as "data", but rounded to the closest allowed state/value

  Author:
  -------
  Joris Kampman, Thales NL, 2017
  '''

  # import numpy as np

  states = rnd2info

  if rnd2info.size == 1:
    nof_states = rnd2info

    if nof_states > 1:
      states = np.linspace(data.min(), data.max(), nof_states)

  # vectorize data
  shp = data.shape

  # minus via singleton expansion
  imin = np.abs(data.reshape(-1, 1) - states.reshape(1, -1)).argmin(axis=1)

  output = states[imin].reshape(*shp)

  return output


def figname(figname_base):
  '''
  *figname* is a function which checks if a figure having the candidate name exists, and if it
  does exists, modifies it by adding a number to the name withing straight brackets ([ and ]).and

  positional argument:
  --------------------
  figname_base    [str] a string with the candidate name. For instance: 'figure of plots'

  *figname* returns the unmodified figname_base in case there exists no figure holding the name
  given in figname_base. However, figname return the modified name when the figname base DOES
  exists.

  For example, if there exists a figure with the name 'figure of plots', *figname* returns
  'figure of plots[1]'.

  However, if 'figure of plots[1]' also exists, *figname* returns 'figure of plots[2]' etc., etc.

  author: Joris Kampman, Thales NL, 2017
  '''

  # create figure
  figname_new = figname_base
  counter = 1

  # check if name exists
  while plt.fignum_exists(figname_new):
    figname_new = figname_base + f'[{counter}]'
    counter += 1

  return figname_new


def db(linval):
  '''
  *lin2db* converts a complex float value or set thereof to decibell values via 10*log10(|x|)

  positional argument:
  --------------------
  linval [ndarray of complex floats] the value or values to be converted to decibell values

  *lin2db* returns the decibell value(s) in the same format/shape as the input data in "linval"

  See also: jktools.db2lin(), jktools.logmod()

  Author: Joris Kampman, Thales NL, 2017
  '''
  with np.errstate(divide='ignore'):
    output = 10*np.log10(np.abs(linval))

  return output


def lin(dbval, mult=10):
  '''
  Converts a decibell value to it's linear equivalent value

  Argument:
  --------------------
  dbval : ndarray of nums
          decibell values to be converted

  Returns:
  --------
  out : array-like of floats
        a set of decibell values to their linear equivalents.

  See Also:
  ---------
  jktools.lin2db : converts linear values to decibells
  jktools.logmod : converts linear values to logmods
  '''

  return 10**(dbval/mult)


def logmod(x, multiplier=20):
  '''
  logmod converts a linear amplitude or power to decibells. Note that a keyword argument
  named "multiplier" is used to distinguish between linear amplitudes (multiplier=20) and
  linear powers (multiplier=10).

  further note that for a pure sinusoidal signal the power is the amplitude**2/2, thus relating the
  amplitude to the power and thus also explaining the use of 10 vs 20 as a multiplier.return

  A peculiarity is the omission of the division by 2. Since logmods are generally relative,
  omitting the factor 1/2 on both the numerator and denominator is allowed. However, converting
  a linear power or amplitude to a logmod value thus is 3 dB to high!

  positional arguments:
  ---------------------
  x   [float(s)] the linear power or amplitude to convert to logmod values. x might be a single
                 value or a numpy array of (complex) floats

  keyword arguments:
  ------------------
  multiplier=20 --> this keyword argument takes a float or int and is used in the calculation:
                    multiplier*log10(abs(x)). This determines whether the input x is treated as
                    a linear amplitude (multiplier=20, the default) or linear power (multiplier=10)
                    Note that the default assumed values for x are (complex-valued) amplitudes

  the function logmod returns the logmod value(s) of the (numpy array) of (complex-valued) floats.

  author: Joris Kampman, Thales NL, 2017
  '''
  # catch the runtime warning
  with np.errstate(divide='ignore'):
    output = multiplier*np.log10(np.abs(x))

  return output


def are_in_bracket(values, bracket_, include_edges=(True, True), merge_result=True,
                   overlap_is_ok=False, order='C'):
  """
  check if a set of values are inside a bracket
  """
  valarr = arrayify(values)

  # keep shape to unravel it back later after raveling
  valshape = np.shape(valarr)

  invec = np.ravel(valarr, order=order)
  outvec = np.zeros_like(invec, dtype=bool)
  for ival, val in enumerate(invec):
    is_inside = is_in_bracket(val, bracket_, include_edges=include_edges)
    outvec[ival] = is_inside

  if merge_result:
    if overlap_is_ok:
      output = bool(np.sum(outvec))
    else:
      output = bool(np.prod(outvec))
  else:
    # unravel back
    output = np.reshape(outvec, valshape, order=order)

  return output


def is_in_bracket(value, bracket_, include_edges=(True, True)):
  """ check if a value is in a bracket """
  if isinstance(include_edges, bool):
    include_edges = [include_edges]*2

  if include_edges[0]:
    tf_begin = value >= bracket_[0]
  else:
    tf_begin = value > bracket_[0]

  if include_edges[1]:
    tf_end = value <= bracket_[1]
  else:
    tf_end = value < bracket_[1]

  tf_overall = bool(tf_begin*tf_end)

  return tf_overall


def find_bracket(value, bracketlist, include_edges=(True, False), allow_multi=False,
                 return_tf=False):
  """
  Find the bracket from a list of brackets in which the value lies

  Arguments:
  ----------
  value : int, float
          The value for which the interval must be found
  bracketlist : array-like of 2-array-likes
                A list/tuple of 2-list/tuples. The 2-list/tuple contain the min and max of the
                bracket
  include_edges : 2-array of bool, default=[True, False]
                  Whether or not to include the edges of the bracket
  """
  is_fnd_list = []
  for bracket_ in bracketlist:
    is_in_this = is_in_bracket(value, bracket_, include_edges=include_edges)
    is_fnd_list.append(is_in_this)

  ifnd_list = np.argwhere(is_fnd_list).ravel()
  if ifnd_list.size > 1 and not allow_multi:
    raise ValueError(f"There are multiple ({ifnd_list.size}) brackets found. This is not allowed")

  if return_tf:
    return is_fnd_list

  return ifnd_list


def bracket(x, axis=-1):
  '''
  *bracket* returns the minimum and maximum values of a ndarray of unlimited dimensions.of

  positional argument:
  --------------------
  x [ndarray] is the argument of whatever shape (multidimensional)

  *bracket* returns a 2-tuple holding the minimum and maximum value of the ndarray "x"
  respectively.

  See also: jktools.range

  Author: Joris Kampman, Thales NL, 2017
  '''
  x = arrayify(x)
  return np.nanmin(x, axis=axis), np.nanmax(x, axis=axis)


def datarange(vals, axis=-1):
  '''
  *range* gives the value range for a - multidimensional - ndarray of numerical values.

  positional argument:
  --------------------
  vals [ndarray of nums] is an array of values for which the value range must be
                         calculated.Author

  *range* returns a single numerical value representing the difference between the minimum and
  maximum value in the ndarray.

  see also: jktools.bracket

  Author = Joris Kampman, Thales NL, 2017.
  '''

  mini, maxi = bracket(vals, axis=axis)
  return maxi - mini


def signal_gain(coefs, fs, scale='lin', freqs=0):
  """
  calcluate the signal gain for a specific set of frequencies
  """
  nof_freqs = freqs.size
  nof_taps = coefs.size

  tvec = np.r_[0:nof_taps]/fs

  gains = np.empty((nof_freqs, nof_taps), dtype=np.complex_)
  for ifreq in range(nof_freqs):
    gains[ifreq] = ((1/nof_taps)*np.abs(np.sum(coefs)).T
                    *np.exp(-2j*np.pi*freqs[ifreq]*tvec))

  if scale == 'db':
    gains = db(gains)
  elif scale == 'logmod':
    gains = logmod(gains)

  return gains


def filter_gains(coefs, axis=-1, scale='db'):
  '''
  *filter_gain* calculates the gains of a filter specified by it's (complex-valued)
  coefficients.

  the following is calculated and returned in a dict:
   - noise gain
   - signal gain
   - signal-to-noise (snr) gain

  positional arguments:
  ---------------------
  coefs : ndarray of complex floats
          The coefficients of the filter. Note that the first dimension must hold the different
          filters, this implies that a set of 3 filters with N taps, must be given as a 2xN
          ndarray.

  keyword arguments:
  ------------------
  scale : [ 'db' | 'lin'], default='db'
                   scale='db' gives the noise gain in decibell
                   scale='lin' gives the noise gain as a linear ratio

  Returns:
  --------
  gains : dict
          Holding the gain values with keys `noise`, `signal` and `snr`. Every key-value pair has
          the same shape; np.shape(gains['noise']) = (N,)
  '''

  # if coefs.ndim == 1:
  #   nof_filters = 1
  # else:
  #   nof_filters = coefs.shape[0]

  # reshape according to filter_axis and nof_filters
  # coefs_2d = coefs.reshape(nof_filters, -1)

  # calculate noise gain
  noise_gain = np.sum(np.abs(coefs)**2, axis=axis)
  signal_gain_est = np.abs(coefs).sum(axis=axis)**2
  snr_gain = signal_gain_est/noise_gain

  gains = {'noise': noise_gain,
           'signal': signal_gain_est,
           'snr': snr_gain}

  if scale == 'lin':
    pass
  elif scale == 'db':
    for key, value in gains.items():
      gains[key] = db(value)
  elif scale == 'lm':
    for key, value in gains.items():
      gains[key] = logmod(value)

  return gains


def subplot_layout(nof_subplots, wh_ratio=np.sqrt(2)):
  '''
  *subplot_layout* calculates the optimal number of rows and columns needed in a regular grid to
  hold a specific number of subplots. This depends on the length/width-ratio of the figure.

  positional arguments:
  ---------------------
  nof_subplots [int] The number of subplots which must be places on a regular grid

  keyword argument:
  -----------------
  wh_ratio=sqrt(2) [handle, float] in case the wh_ratio type is a figure handle, the ratio of the
                   existing figure is taken. Otherwise the value is directly the ratio between the
                   width and the height.

                   Note that the default value (sqrt(2)) is equal to A-format in landscape
                   orientation. Portrait orientation would therefore be sqrt(2)/2.

  *subplot_layout* returns a 2-tuple containing the number of rows and number of columns required
  to hold the number of subplots requested with the required width/height ratio.

  See also: jktools.ind2sub()
            jktools.sub2ind()
            jktools.resize_figure()
            jktools.tighten()

  Author: Joris Kampman, Thales NL, 2017
  '''

  n_sqr = np.sqrt(nof_subplots)
  sf = np.power(wh_ratio, 1 / 2)

  parts = n_sqr * np.array([1 / sf, sf])
  fracs = parts % 1

  # determine a division in integer rows and columns
  if fracs[0] >= fracs[1]:
    nof_rows = np.round(parts[0]).astype(int)
    nof_cols = np.ceil(nof_subplots / nof_rows).astype(int)

    if nof_rows * nof_cols - nof_cols == nof_subplots:
      nof_rows -= 1

  else:
    nof_cols = np.round(parts[1]).astype(int)
    nof_rows = np.ceil(nof_subplots / nof_cols).astype(int)

    if nof_rows * nof_cols - nof_rows == nof_subplots:
      nof_cols -= 1

  return (nof_rows, nof_cols)

# pylint: disable-next=W0102
def save_animation(anim, filename, fps=30,
                   extra_args={'-vcodec': 'h264', '-preset': 'veryslow', '-crf': '23'}, **metadata):
  '''
  *save_animation* creates a moviefile via ffmpeg using an animation object from matplotlib as an
  input

  positional arguments:
  ---------------------
  anim [animation object] The animation object as created in the caller. See the module
                          matplotlib.animation for more info.
  filename [str] The filename with full path for the file to create. Note that the correct
           extension is added (.mp4)


  keyword arguments:
  ------------------
  fps=30 [int] is the number of frames per second, defaulting to 30

  metadata [dict] is the metadata to a movie. See matplotlib.animation for more info on possible
           keywords in the metadata dict.

  extra_args= <see syntax> [dict] arguments for the ffmpeg encoder. The default set was found via
                           trial and error and is tested to give good results with reasonable
                           filesizes for even non-realistic movies (pure graphs and lines and such,
                           GIF stuff)

  *save_animation* returns None

  See also: matplotlib.animation (module)

  Author: Joris Kampman, Thales NL, 2017
  '''
  # update the metadata
  metadata = {'artist': 'Joris Kampman, Saxion SMART Mechatronics and RoboTics'}
  metadata.update(metadata)

  print('Saving animation ..', end='', flush=True)
  anim.save(filename, writer='ffmpeg', fps=fps, metadata=metadata, extra_args=extra_args)
  print('finished')

  return None


def paper_A_dimensions(index, units="m", orientation='landscape'):
  """
  calculate the paper dimensions for A<x> paper sizes

  The return value is a tuple (short size, long size)
  """
  if units == 'm':
    sf = 1.
  elif units == 'mm':
    sf = 1000.
  if units == 'cm':
    sf = 100.
  elif units.startswith("inch"):
    sf = 1./0.0254
  else:
    raise ValueError(f"The units={units} is not recognized")

  # define the base alphaA
  alpha_A = sf*(2)**(1./4.)

  sht = alpha_A*2**(-(index+1)/2.)
  lng = alpha_A*2**(-index/2.)

  if orientation == 'landscape':
    w = lng
    h = sht
  elif orientation == 'portrait':
    w = sht
    h = lng
  else:
    raise ValueError(f"The value for *orientation* ({orientation}) is not valid")

  return w, h


def get_nof_monitors(warning_thres=0.25):
  """
  get, or better, estimate the number of monitors attached.

  Works only with an aspect ratio of about 1.78 (1920/1080)
  """
  default_ratio = 1920/1080

  wpix, hpix = get_screen_dims(units='pix')
  nof_monitors_est_fl = wpix/(default_ratio*hpix)
  nof_monitors = int(0.5 + nof_monitors_est_fl)

  # give warning in case uncertainty is too high
  uncertainty = nof_monitors_est_fl%1
  if warning_thres <= uncertainty <= (1. -warning_thres):
    warnings.warn(f"Number of estimated monitors (={nof_monitors}) "
                  + f"has high uncertainty (={uncertainty})")

  return nof_monitors


def get_max_a_size_for_display(orientation='landscape', nof_monitors='auto',
                               units='inches'):
  """
  get the maximum size with A-ratio that can be displayed
  """
  wfigmax, hfigmax = get_screen_dims(units=units)
  # check if there are multiple monitors

  if isinstance(nof_monitors, str) and nof_monitors == 'auto':
    nof_monitors = int(0.5 + np.log2(wfigmax/hfigmax))

  wfigmax /= nof_monitors
  wA0, hA0 = paper_A_dimensions(0, units=units, orientation=orientation)

  # ratio for the conversion of max height to width
  whratio = wA0/hA0

  # Height can always be maximized, width follows from ratio
  height = hfigmax
  width = height*whratio

  return width, height


def smooth_data(data, filtsize=0.07, std_filt=2.5, makeplot=False,
                downsample=False, nof_pts_per_filt='auto', return_indices=False,
                return_filt=False, edge='mirror'):
  """
  smooth the data

  arguments:
  ----------
  data : np.ndarray
         The data points to be smoothed

  filtsize : float or int, default=0.07
             The filter size.
             If < 1., it is the fraction of the data length
             If > 1., it is the number of taps (rounded to nearest integer)
  std_filt : float, default=2.5
             The standard deviation of the smoothing mask
  makeplot : bool, default=False
             If True creates a plot showing the raw data and the smoothed result
  downsample : bool, default=True
               Whether to downsample the smoothed data
  nof_pts_per_filt : ['auto' | int], default='auto'
                     If 'downsample' is true, these are the number of points per filter. 'auto'
                     will return a data equal to the standard deviation
  return_indices : bool, default=False
                   if True, the indices of the DOWNSAMPLED data set are returned
  return_filt : bool, default=False
                If True, the applied filter coefficients are returned. Usefull in case the filter
                was automatically created based on the input data set
  edge : [ 'mirror' | 'sample' ], None or float, default='mirror'
         How to process the edges. Options are:
         - None     : nothing is done. The edges are appended with zeros
         - mirror   : The samples are mirrored and copied
         - sample   : the first and last samples are taken
         - float    : a value is taken for the edge value

  Outputs:
  --------
  ipts : np.ndarray
         The interpolated and downsampled points such that the raw (high sample rate) and
         downsampled data graphs are overlaying. If no downsampling, this is the basis set of 0 to
         nof original samples
  data_f : (optional) np.ndarray
           The filtered and possibly downsampled data
  filt : (optional) np.ndarray
         the actual filter used in smoothing

  Author:
  -------
  Joris Kampman, Thales NL, 2023
  """
  if filtsize <= 1.:
    filtsize = 2*(np.round(filtsize*data.size)//2)
  else:
    if float(filtsize).is_integer():
      pass
    else:
      warnings.warn(f"The filter size given ({filtsize:0.3f}) is not an integer"
                    + f"It will be rounded to {np.round(filtsize).astype(int):d}")
      filtsize = np.round(filtsize)

  # make it into a size of a filter (nof samples)
  nof_filt_samples = int(0.5 + filtsize)

  if nof_filt_samples == 0:
    warnings.warn("The filter is of zero length. Function returned")
    return data

  if nof_filt_samples%2 == 0:
    warnings.warn(f"The number of filter samples ({nof_filt_samples}) is EVEN. Be carefull!")

  if nof_filt_samples < 1:
    data_f = data
    filt = np.array([1.], dtype=float)
  else:
      # make a gaussian filter
    std = nof_filt_samples/std_filt
    filt = spsw.gaussian(nof_filt_samples, std, sym=True)
    filt -= filt.min()
    filt /= filt.sum()

    nof_ext_samples = int(0.5 + (nof_filt_samples - 1)/2)
    if edge is None:
      data_ext = data
    elif isinstance(edge, (float, np.floating, complex, np.complexfloating)):
      data_ext = np.concatenate((data[0]*nof_ext_samples, data, data[-1]*nof_ext_samples))
    elif isinstance(edge, str):
      if edge == 'mirror':
        data_ext = np.concatenate((data[nof_ext_samples:0:-1], data, data[-nof_ext_samples:]))
      elif edge == 'sample':
        data_ext = np.concatenate(([data[0]]*nof_ext_samples,
                                   data,
                                   [data[-1]]*nof_ext_samples))
      else:
        raise ValueError(f"The given value for 'edge' ({edge}) is not valid!")
    else:
      raise TypeError(f"The type for argument 'edge' ({type(edge)} is not implemented")

    # determine convolution mode and filter
    if data_ext.size > data.size:
      conv_mode = 'valid'
    else:
      conv_mode = 'same'
    data_f = np.convolve(data_ext, filt, mode=conv_mode)

  ipts = np.r_[:data_f.size]
  if makeplot:
    ax = qplot(data, 'k.-', label="Unfiltered")
    qplot(ax, ipts, data_f, 'r.-', label="Filtered - same sample-rate")

  # if downsample
  if downsample:
    if isinstance(nof_pts_per_filt, str) and nof_pts_per_filt == 'auto':
      nof_pts_per_filt = int(0.5 + std_filt)

    stepsize = int(0.5 + (filt.size - 1)/nof_pts_per_filt)
    ipts = ipts[::stepsize]
    data_f = data_f[::stepsize]

  if makeplot:
    ax = qplot(ax, ipts, data_f, 'g.-', label="Filtered - reduced sample-rate")
    add_figtitles("Smoothed data")

  output = data_f
  if return_indices:
    output = (data_f, ipts)
  if return_filt:
    if return_indices:
      output = (data_f, ipts, filt)
    else:
      output = (data_f, filt)

  return output


def abc(a, b, c):
  """
  calculates the ABC equation

  :param a:
  :param b:
  :param c:
  :return:
  """
  D = b**2 - 4*a*c
  x_min = (-b - np.sqrt(D))/(2*a)
  x_max = (-b + np.sqrt(D))/(2*a)

  return x_min, x_max


def f1_score(nof_true_pos, nof_false_pos, nof_false_neg):
  """
  calculates the F1 score of a cross-validity check on a trained set

  :param nof_true_pos:
  :param nof_true_negs:
  :param nof_false_pos:
  :param nof_false_negs:
  :return:
  """
  precision = nof_true_pos/(nof_true_pos + nof_false_pos)
  recal = nof_true_pos/(nof_true_pos + nof_false_neg)

  F1 = 2*precision*recal/(precision + recal)

  return F1


def scale_filter_coefs(coefs, axis=-1, gain_type='noise', value_db=0.):
  """
  Scales filter coefficients to either noise, signal or snr gain

  //TODO: fill in docstring
  """

  # calculate the gains
  gains = filter_gains(coefs, axis=axis, scale='db')[gain_type]

  offset_needed = value_db - gains

  sf_lin = np.power(10, offset_needed/20)

  coefs_scaled = coefs*sf_lin.reshape(-1, 1)

  return coefs_scaled


def power2amp(power, power_units='lin'):
  """
  convert a value in power to an amplitude
  """
  if power_units == 'db':
    power = lin(power)
  elif power_units == 'lin':
    pass
  else:
    raise ValueError(f'the *power_units* value: "{power_units}" is not valid.\n'
                     + 'Only "lin" and "db" are valid choices')

  return np.sqrt(2*power)


def power2rms(power, power_units='lin'):
  """
  convert a value in power to an amplitude
  """
  if power_units == 'db':
    power = lin(power)
  elif power_units == 'lin':
    pass
  else:
    raise ValueError(f'the *power_units* value: "{power_units}" is not valid.\n'
                     + 'Only "lin" and "db" are valid choices')

  return np.sqrt(power)


def exp_fast(data):  # pylint: disable=unused-argument
  """
  calculates the fast exp via numexpr module
  """

  return ne.evaluate('exp(data)')


def find_blob_edges(blob, threshold=1., return_mask=False):
  """
  blob must be convex
  """
  blob_floats = blob.astype(float)
  blob_floats[blob_floats < threshold] = 0.
  blob_floats[blob_floats >= threshold] = 1.

  mask = np.ones((3, 3), dtype=float)

  # do 2D convolution
  convres = convolve2d(blob_floats, mask, 'same')

  # set points that are not edges to zero
  convres[convres < 5] = 0.
  convres[convres > 7] = 0.
  convres[convres > 0] = 1.

  tf_edges = convres.astype(bool)

  if return_mask:
    retval = tf_edges
  else:

    # get the order of the pixels to prevent jumping edges
    shp = blob.shape
    Rgrid, Cgrid = np.mgrid[:shp[0], :shp[1]]

    Xgrid = Cgrid - np.mean(Cgrid)
    Ygrid = Rgrid - np.mean(Rgrid)

    radii = np.sqrt(Xgrid**2 + Ygrid**2)
    phs = np.arctan2(Ygrid, Xgrid)

    z_valids = (radii*np.exp(1j*phs))[tf_edges]
    is_fnd = -1*np.ones(tf_edges.sum(), dtype=int)

    # find the minimum angle
    ifnd = np.argmin(np.abs(np.angle(z_valids)))
    is_fnd[ifnd] = 0

    # loop for all
    for ipt in range(1, tf_edges.sum()):
      is_unused = np.argwhere(is_fnd == -1).ravel()

      # find the closest point
      z_valids_unused = z_valids[is_unused]
      iiclosest = np.argmin(np.abs(z_valids_unused - z_valids[is_fnd.argmax()]))
      ifnd = is_unused[iiclosest]
      is_fnd[ifnd] = ipt

    isort = np.argsort(is_fnd)

    # get the indices that belonw to the edge in coordinates
    edge_coords = np.argwhere(tf_edges)[isort, :]
    edge_coord_lots = [(r, c) for r, c in edge_coords]

    retval = edge_coord_lots

  return retval


def qplot(*args, center=False, aspect=None, rot_deg=0., thin='auto',
          mark_endpoints=False, endpoints_as_text=False, endpoint_color='k',
          split_complex_vals=True, colors='jetmodb', legend=True, legend_loc='upper right',
          legkwargs=None, figtitles=None, txt_rot='auto', margins=0.01, grid=None,
          datetime_fmt='auto', return_lobjs=False, plotfun='plot', plotnow=True, **plotkwargs):
  """
  a quicklook plot

  positional arguments:
  ----------------------
  ax: [ Axes | None | 'h' ], OPTIONAL, default=None
      The axes in which the plots must be created. optiouns are:
      - Axes object --> this will add the plot to an existing axes. plt.gca() also works
      - None       _arts --> a new plot will be created
      - 'h'         --> the current plot will be held. Hence 'h' for 'hold' (similar to matlab)
  args[1:] : <see matplotlib.pyplot.plot>
             may be 1, 2 or 3 arguments:
             - 1 argument  --> The data for the vertical axis OR 2D complex valued data
             - 2 arguments --> Both the horizontal and vertical axis data. In case both arguments
                               are complex.. well, nothing's been implemented for that now!!
             - 3 arguments --> data for both axis, plus a formatting string, e.g., 'b.-'

  keyword arguments:
  ------------------
  ax : [None | axes | 'hold'], default='hold'
       The axes where to plot it. None will create a new axes, while 'hold' will use the existing
       one. An axes instance will just plot in the given axes (this is the preferred way)
  center : bool, default=False
           Whether the create a plot centered around zero with equal width and height. Works shit
           in case of outliers though
  aspect : ['auto' | 'equal'], default='auto'
           Sets the aspect ratio of the axes. See matplotlib.pyplot.set_aspect.
  rot_deg : float, default=0.
            The angle with which to rotate all points. By default there is no rotation, thus set
            to 0. degrees. In case this is too slow, make a switch
  mark_endpoints : bool, default=False
                   mark the start and endpoint with a marker. The start is indicated by a black
                   square withouth a face (so it circumvents the point itself), while the
                   endpoint is a faceless black circle.
  endpoints_as_text : bool, default=False
                      Whether to replace the endpoint markers (square and circle) by the texts
                      'start' and 'end', this might be beneficial in some cases
  endpoint_color : [ str| 3-array-like ], default='k'
                   The color of the endpoints
  return_kid : bool, default=False
               whether or not to not only return the axes, but also the line objects themselves
  split_complex_vals : bool, default=False
                  Whether or not to split the data into 2 graphs: the real and complex valued data

  **kwargs : dictionary
             keyword arguments to be given to the underlying matplotlib.pyplot.plot function
             to which this function is a wrapper

  returns:
  --------
  lobj : Line2D
         The line object
  ax : axes
       The axes object containing the plots
  """
  if legkwargs is None:
    legkwargs = {}

  legkwargs_arts = dict(lw=1.5, markersize=6, alpha=1.)

  thini_settings = {0: dict(lw=1.5, markersize=6, alpha=1),
                    1: dict(lw=1., markersize=2, alpha=0.5),
                    2: dict(lw=0.5, markersize=0.5, alpha=0.2)}

  kwargs = dict()
  if not isinstance(args[-1], str):
    kwargs = dict(marker='.', ls='-')

  kwargs.update(**plotkwargs)
  legkwargs_arts.update(**plotkwargs)

  # if first argument is an axes --> use this axes
  if isinstance(args[0], plt.Axes):
    ax = args[0]
    args = args[1:]
  # else: if first argument is None --> new axes
  elif args[0] is None:
    _, ax = plt.subplots(1, 1)
    args = args[1:]
  # elif: first argument is 'h' --> use the current axes
  elif isinstance(args[0], str) and args[0].startswith('h'):
    ax = plt.gca()
    args = args[1:]
  # nothing given --> new axes
  else:
    _, ax = plt.subplots(1, 1)

  # ========= type of plot (plot, semilogx, semilogy)  ========================================
  if plotfun == 'plot':
    pfun = ax.plot
  elif plotfun == 'semilogx':
    pfun = ax.semilogx
  elif plotfun == 'semilogy':
    pfun = ax.semilogy
  else:
    raise ValueError("The value given for 'plotfun' (='{plotfun}') is not valid")

  # set the aspect ratio ('auto' and 'equal' are accepted)
  if aspect is not None:
    ax.set_aspect(aspect)

  # convert thin to integer
  if isinstance(thin, str):
    if thin == 'auto':
      thin = 0
      # check the number of elements in the data
      nof_points = 1
      for arg in args:
        if isinstance(arg, np.ndarray):
          nof_points_ = arg.size
        else:
          nof_points_ = np.array(arg).size
        nof_points = max(nof_points, nof_points_)
      # check what to do for a certain amount of points
      if nof_points > 5000:
        thin = 1
        if nof_points > 20000:
          thin = 2
    else:
      raise ValueError(f"the given value for 'thin' ({thin}) is not valid")

  thinkwargs = thini_settings[int(thin)]
  if isinstance(thinkwargs, dict):
    thinkwargs.update(**kwargs)
    kwargs = thinkwargs

  # check if there is an formatting argument given, this is always the last one
  if isinstance(args[-1], str):
    format_str_list = [args[-1]]
    datalist = args[:-1]

  else:
    format_str_list = []
    datalist = args

  if len(datalist) == 1:
    xdata = None
    ydata = np.squeeze(datalist[0])
  # there are 2 separate x and y sets given
  elif len(datalist) == 2:
    xdata = np.squeeze(args[0])
    ydata = np.squeeze(args[1])
  else:
    raise ValueError(f"There are more than 2 input arguments given ({len(args)})")

  # check dimensions (prevent this weird 0 dimension thing)
  if xdata is not None and xdata.ndim == 0:
    xdata = np.array([xdata])

  if ydata.ndim == 0:
    ydata = np.array([ydata])

  # check if empty
  if not np.isscalar(ydata) and len(ydata) == 0:
    warnings.warn("The data set is empty!")

    label = kwargs.pop('label') if 'label' in kwargs else ''

    lobj = pfun([], [], *format_str_list, label=label, **kwargs)[0]
    if return_lobjs:
      return ax, lobj

  if np.isscalar(ydata):
    ydata = [ydata]

  if np.isscalar(xdata):
    xdata = [xdata]

  # check if ydata is a list of data or simply plain data
  is_multiple = isinstance(ydata[0], (list, tuple, np.ndarray))
  # make multiple of 1 if not multiple
  if not is_multiple:
    ydata = [ydata]

  # adjust the xdata if still None
  if xdata is None:
    xdata = np.arange(len(ydata[0]))

  # check if the xdata is a list too
  if not isinstance(xdata[0], (list, tuple, np.ndarray)):
    xdata = [xdata]*len(ydata)

  # set the text rotation in case the xdata is strings
  if txt_rot.startswith('auto'):
    if isinstance(xdata[0][0], str):
      txt_rot = 45.
    else:
      txt_rot = 0.

  # special treatment for x-axis dates
  is_datetime_data = True if isinstance(np.squeeze(xdata).item(0), dtm.datetime) else False

  # special treatment for complex data
  is_complex = np.any([np.iscomplex(ys).sum() > 0 for ys in ydata])
  if is_complex:
    # split the real and imaginary parts into separate graphs
    xdata_ext = []
    ydata_ext = []
    if split_complex_vals:
      for xs, ys in zip(xdata, ydata):
        xdata_ext.append(xs)
        ydata_ext.append(np.real(ys))
        xdata_ext.append(xs)
        ydata_ext.append(np.imag(ys))
    else:
      for ys in ydata:
        xdata_ext.append(np.real(ys))
        ydata_ext.append(np.imag(ys))
    # overwrite the existing data sets
    xdata = xdata_ext
    ydata = ydata_ext

  if not np.isclose(rot_deg, 0.):
    xdata_rot = []
    ydata_rot = []
    for xs, ys in zip(xdata, ydata):
      xs, ys = rot2D(xs, ys, np.deg2rad(rot_deg))
      xdata_rot.append(xs)
      ydata_rot.append(ys)
    xdata = xdata_rot
    ydata = ydata_rot

  # set the label if present
  nof_plots = len(ydata)
  if isinstance(colors, str):
    if colors.startswith('jetmod'):
      modifiers = colors[6:]
      bright = True if 'b' in modifiers else False
      invert = True if 'i' in modifiers else False
      negative = True if 'n' in modifiers else False
      interpolation = 'nearest' if '_' in modifiers else 'linear'
      colors = jetmod(nof_plots, 'vector', bright=bright, invert=invert, negative=negative,
                      interpolation=interpolation)
    else:
      raise NotImplementedError("Other colormaps than 'jetmod' are not implemented yet")

  label_list = listify(kwargs.pop('label')) if 'label' in kwargs else ['']*nof_plots
  # change None to ''
  label_list = ['' if label is None else label for label in label_list]
  if is_complex and split_complex_vals:
    label_ext = []
    colors = ['r', 'b']
    for label in label_list:
      label_ext.append(f"real({label})")
      label_ext.append(f"imag({label})")
    # place back into label list
    label_list = label_ext

  ax.set_prop_cycle(color=colors)
  # how many plots to make?
  lobjs = []

  # check if it must be thin
  nof_points = np.array(ydata).size
  for xs, ys, label in zip(xdata, ydata, label_list):
    lobj = pfun(arrayify(xs), arrayify(ys), *format_str_list, label=label, **kwargs)[0]
    lobjs.append(lobj)

  if center:
    center_plot_around_origin(ax=ax)
    # ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=3)

  if mark_endpoints:
    if endpoint_color.startswith('match'):
      endpoint_color = lobj[0].get_color()

    if endpoints_as_text:
      for xs, ys in zip(xdata, ydata):
        ax.text(xs[0], ys[0], 'start', fontsize=8, fontweight='bold', ha='center',
                va='center', alpha=0.5, color=endpoint_color)
        ax.text(xs[-1], ys[-1], 'end', fontsize=8, fontweight='bold', ha='center',
                va='center', alpha=0.5, color=endpoint_color)
    else:
      for xs, ys in zip(xdata, ydata):

        pfun(xs[0], ys[0], 's', mfc='none', markersize=10, markeredgewidth=2,
                alpha=0.5, color=endpoint_color)
        pfun(xs[-1], ys[-1], 'o', mfc='none', markersize=10, markeredgewidth=2,
                alpha=0.5, color=endpoint_color)

  is_label_present = np.all([len(label) > 0 for label in label_list])
  if is_label_present and legend:
    legkwargs_base = dict(fontsize='small', numpoints=1, scatterpoints=1, **legkwargs)
    legkwargs_base.update(legkwargs)
    if legend_loc == 'above':
      legend_loc = 'lower center'
      legkwargs_base.update(loc='lower center',
                            bbox_to_anchor=[0.5, 1.],
                            ncol=max(5, len(label_list)))
    else:
      legkwargs_base.update(loc=legend_loc)
    leg = ax.legend(**legkwargs_base)

    for legobj in leg.legend_handles:
      legobj.set_alpha(1)
      if thin > 0:
        # if marker is not existing
        if legobj.get_marker() == '' and legobj.get_linestyle().lower().startswith('none'):
          legobj.set_marker('o')
          legobj.set_markersize(2)
          legobj.set_markerfacecolor(legobj.get_color())
        else:
          legobj.set_linewidth(legkwargs_arts['lw'])
          legobj.set_markersize(legkwargs_arts['markersize'])
        # pdb.set_trace()
      # plt.draw()

  if figtitles is not None:
    add_figtitles(figtitles)

  if is_datetime_data:
    if datetime_fmt != 'auto':
      fmt = mdates.DateFormatter(datetime_fmt)
      ax.xaxis.set_major_formatter(fmt)
    # ax.figure.autofmt_xdate()

  # rotate the xtick labels if not a date (this will be automatically updated)
  if not np.isclose(txt_rot, 0.):
    ax.tick_params(axis='x', labelrotation=txt_rot, labelsize='small')

  if margins is None:
    margins = [None]*2
  elif np.isscalar(margins):
    margins = [margins]*2
  else:
    pass

  ax.margins(x=margins[0], y=margins[1])
  # en-, or disable grid
  if grid is not None:
    ax.grid(grid)

  if plotnow:
    plt.show(block=False)
    plt.pause(1e-4)
    plt.draw()
    plt.pause(1e-2)

  retval = ax
  if return_lobjs:
    # check if it is a singleton
    if len(lobjs) == 1:
      lobjs = lobjs[0]

    # create retval tuple
    retval = (ax, lobjs)

  return retval


def center_plot_around_origin(ax=None, aspect='equal'):
  """
  center the plot based on the current content
  """
  # get current bounds
  if ax is None:
    ax = plt.gca()

  dev = np.max([*np.abs(ax.dataLim.max), *np.abs(ax.dataLim.min), 0.])

  ax.set_xlim(left=-dev, right=dev)
  ax.set_ylim(top=dev, bottom=-dev)

  if aspect is not None:
    ax.set_aspect(aspect)

  return ax


def center_plot(axs=None, axis='auto', tight=False, dist=None):
  """
  center a plot around x and or y axes

  tight indicates to take the smallest distance
  """
  inds = dict(x=0, y=1)
  if axs is None:
    axs = plt.gca()

  axs = listify(axs)
  # get the extents
  extents_list = []
  for ax in axs:
    extents_list.append(ax.dataLim.extents)

  extarr = np.array(extents_list)

  dist = listify(dist)
  if len(dist) == 1:
    dist = dist*2

  tight = listify(tight)
  if len(tight) == 1:
    tight = tight*2

  if axis == 'both':
    ax2mod = ['x', 'y']
  elif axis == 'auto':
    # check if the data extends to both sides of the origin
    ax2mod = []
    if np.any(extarr[:, 0] < 0.) and np.any(extarr[:, 2] > 0.):
      ax2mod.append('x')
    if np.any(extarr[:, 1] < 0.) and np.any(extarr[:, 3] > 0.):
      ax2mod.append('y')
  else:
    ax2mod = [axis]

  for axname in ax2mod:
    minvals = extarr[:, inds[axname]]
    maxvals = extarr[:, inds[axname]+2]

    if dist[inds[axname]] is None:
      if tight[inds[axname]]:
        dist_ = min(*np.abs(minvals), *np.abs(maxvals))
      else:
        dist_ = max(*np.abs(minvals), *np.abs(maxvals))
    else:
      dist_ = dist[inds[axname]]

    if axname == 'x':
      for ax in axs:
        ax.set_xlim(left=-dist_, right=dist_)
    else:
      for ax in axs:
        ax.set_ylim(bottom=-dist_, top=dist_)

  plt.show(block=False)
  plt.draw()
  plt.pause(1e-3)

  return None


def ctform_mat(angles_xyz, translation_xyz, order_rotations, augment=True):
  '''
  gives the rotation matrix for a sequence of rotations around the principal axes

  Arguments:
  ----------
  angles_xyz : array-like(3) of floats
               The rotations around the x, y and z axes in radians
  translation_xyz : array-like(3) of floats
                    The translations in the x, y and z directions
  order_rotations : ('zxy') str (3), optional
                    A string of 3 chararcters giving the rotations order
  augment : (True) Boolean
            Indicates whether to augment the matrix, that is, to append a 1 to the 3x1 vector to
            make it into a 4x1 vector

  returns:
  --------
  out : 3x3 or 4x4 ndarray of floats
        The rotation matrix (3x3) or augmented transformation matrix (4x4)

  See Also:
  ---------
  coordinate_transforms.Rotate : class handling all kinds of rotations
  coordinate_transforms.ctform : rotates a set of points using the matrix from *ctform_mat*
  coordinate_transforms.rotx : gives the rotation matrix around the x-axis
  coordinate_transforms.roty : gives the rotation matrix around the y-axis
  coordinate_transforms.rotz : gives the rotation matrix around the z-axis
  '''

  if angles_xyz is None:
    angles_xyz = [0.]*3

  if translation_xyz is None:
    translation_xyz = np.zeros((3, 1), dtype=float)

  # convert to matrix
  if type(translation_xyz) in [list, tuple]:
    translation_xyz = np.array(translation_xyz).reshape(-1, 1)

  # create M = [R, T]
  rotmat = rot3(angles_xyz, order_rotations)

  # combine R and T
  tform_matrix = np.hstack((rotmat, translation_xyz))

  # check if it has to be augmented
  if augment:
    tform_matrix = np.vstack((tform_matrix, np.array([0., 0., 0., 1.])))

  return tform_matrix


def ctform(points, angles_xyz, translation_xyz, order_rotations):
  '''
  coordinate transformation given a set of points

  Arguments:
  ----------
  points : 3xN ndarray of floats
           the 3D coordinates of the points in the non-transformed coordinate system
  angles_xyz : array-like(3) of floats
               The rotations around the x, y and z axes in radians
  translation_xyz : array-like(3) of floats
                    The translations in the x, y and z directions
  order_rotations : ('zxy') str (3)
                    A string of 3 chararcters giving the rotations order

  returns:
  --------
  out : 3xN ndarray of floats
        The rotated and translated points in the coordinate system

  See Also:
  ---------
  coordinate_transforms.Rotate : class handling all kinds of rotations
  coordinate_transforms.ctform_mat : Calculates the rotation matrix used in *ctform*
  coordinate_transforms.rotx : gives the rotation matrix around the x-axis
  coordinate_transforms.roty : gives the rotation matrix around the y-axis
  coordinate_transforms.rotz : gives the rotation matrix around the z-axis
  '''
  tform_matrix = ctform_mat(angles_xyz, translation_xyz, order_rotations, augment=False)

  # augment points with ones
  points_aug = np.vstack((points, np.ones(points[0].shape)))

  # do the transformation
  points_tformed = tform_matrix.dot(points_aug)

  return points_tformed


def rotx(rx):
  '''
  rotation around the x-axis

  Arguments:
  ----------
  rx : float
       The angles in radians with which to rotate

  Returns:
  --------
  rotmat : 3x3 ndarray of floats
           The rotation matrix around the x-axis based on *rx*
  '''
  rotmat = np.array([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]])
  return rotmat


def roty(ry):
  '''
  rotation around the y-axis

  Arguments:
  ----------
  ry : float
       The angles in radians with which to rotate

  Returns:
  --------
  rotmat : 3x3 ndarray of floats
           The rotation matrix around the y-axis based on *ry*
  '''

  rotmat = np.array([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]])
  return rotmat


def rotz(rz):
  '''
  rotation around the z-axis

  Arguments:
  ----------
  rz : float
       The angles in radians with which to rotate

  Returns:
  --------
  rotmat : 3x3 ndarray of floats
           The rotation matrix around the z-axis based on *rz*
  '''

  rotmat = np.array([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]])
  return rotmat


def rot3(angles_xyz, order_rotations):
  '''
  create a 3x3 rotation matrix built-up from 3 principal rotation matrices

  Positional Arguments:
  ---------------------
  angles_xyz : array-like (3)
               an array like of 3 elements containing the rotations around the x, y and z-axis
               respectively
  order_rotations : str
                    Indicating the order of rotations. May only contain characters 'x', 'y' and 'z'
                    in any order or permutation. The number of characters == 3.

  Returns:
  --------
  rot : 3x3 ndarray
        The rotation matrix

  See Also:
  ---------
  .rotx : rotation around the cs' x-axis
  .roty : rotation around the cs' y-axis
  .rotz : rotation around the cs' z-axis
  .ctform : transport a set of points via a rotation and translation
  .ctform_mat : gives the transformation matrix with which points are rotated
  '''

  # create rotation matrix
  rotmat = np.eye(3)
  for iaxis in range(len(order_rotations) - 1, -1, -1):
    if order_rotations[iaxis] == 'x':
      rotmat = np.dot(rotx(angles_xyz[0]), rotmat)

    elif order_rotations[iaxis] == 'y':
      rotmat = np.dot(roty(angles_xyz[1]), rotmat)

    elif order_rotations[iaxis] == 'z':
      rotmat = np.dot(rotz(angles_xyz[2]), rotmat)

    else:
      raise ValueError(f'The given axis "{order_rotations[iaxis]}" is not valid')

  return rotmat


def rot2D(xs, ys, angle):
  """
  do a 2D rotation. Angle in RADIANS
  """
  xsa = arrayify(xs)
  ysa = arrayify(ys)

  poss = np.vstack((xsa, ysa))

  rotmat = rotz(angle)[:2, :2]

  xsr, ysr = rotmat@poss

  return xsr, ysr


def subtr_angles(angle1, angle2):
  """
  subtract angles
  """

  return np.angle(np.exp(1j*(angle1 - angle2)))


def add_angles(angle1, angle2):
  """
  add angles
  """

  return np.angle(np.exp(1j*(angle1 + angle2)))


def mean_angle(angle1, angle2):
  """
  calculate the mean angle
  """

  return np.angle(np.exp(1j*angle1) + np.exp(1j*angle2))


def str2int(strnum):
  """
  convert a string to an int
  """
  floatval = np.float_(strnum)
  if not np.isclose(floatval%1, 0):
    warnings.warn(f"The number string *{strnum}* is no integer, so it will be rounded first",
                  UserWarning)

  return int(0.5 + floatval)


def str2timedelta(numstr):
  """
  convert a string for format '10Y' or '1s' to a time delta
  Valid characters are:
    - d: days
    - H/h: hours
    - M/m: minutes
    - s: seconds
  """

  # initialize all components
  days = 0
  hours = 0
  minutes = 0
  seconds = 0

  # overwrite the correct component
  val = str2int(numstr[:-1])
  if numstr.endswith('d'):
    days = val
  elif numstr.endswith(('H', 'h')):
    hours = val
  elif numstr.endswith(('M', 'm')):
    minutes = val
  elif numstr.endswith('s'):
    seconds = val
  else:
    raise ValueError(f"The string '{numstr}' does not end with a valid character")

  dt_delta = dtm.timedelta(days=days,
                           hours=hours,
                           minutes=minutes,
                           seconds=seconds)

  return dt_delta


def _convert_to_list_of_tuples(input_):
  """
  make an input to a list of tuples
  """
  # make it al into a list of tuple(s)
  if input_ is None:
    return None

  if isinstance(input_, list):
    for item, _ in enumerate(input_):
    # for item in range(len(input_)):
      if isinstance(input_[item], str):
        input_[item] = (input_[item],)

  if isinstance(input_, tuple):
    input_ = [input_]

  if isinstance(input_, str):
    input_ = [(input_,),]

  return input_.copy()


def ind2rgba(arr, cmap, alphas=None):
  '''
  convert indexed image to rgb[a]
  '''
  arr_norm = (arr - np.nanmin(arr))/(np.nanmax(arr) - np.nanmin(arr))

  # do interpolation
  fi = interp1d(np.linspace(0., 1., cmap.shape[0]), cmap, axis=0)
  arr_rgb = fi(arr_norm)

  if alphas is not None:
    return np.concatenate((arr_rgb, np.expand_dims(alphas, -1)), axis=-1)
  else:
    return arr_rgb


def short_string(str_, maxlength=None, what2keep='edges', placeholder="..."):
  """
  shorten a long string to keep only the start and end parts connected with dots
  """
  # if no length is given, take the terminal size
  if maxlength is None:
    maxlength = os.get_terminal_size().columns
  # if <0, deduct the amount from the maximum terminal size
  if maxlength < 0:
    maxlength += os.get_terminal_size().columns

  strlen = len(str_)
  pllen = len(placeholder)
  if strlen <= maxlength:
    return str_

  # check if it is in the middle
  if what2keep in ('middle', 'center', 'centre'):
    what2keep = strlen//2 - pllen

  if isinstance(what2keep, (np.integer, int, float, np.floating)):
    what2keep = int(what2keep + 0.5)
    nof_chars = maxlength - 2*pllen
    istart_keep = int(what2keep + 0.5)
    iend_keep = istart_keep + nof_chars
    # if start point is less than length of placeholder there is no point in using placeholder
    if istart_keep < pllen:
      delta = pllen - istart_keep
      strout = str_[0:iend_keep+delta] + placeholder

    elif iend_keep > strlen:
      delta = iend_keep - strlen
      strout = placeholder + str_[istart_keep-delta:]
    else:
      strout = placeholder + str_[istart_keep:iend_keep] + placeholder
  elif what2keep == 'edges':
    nstart = (maxlength - pllen)//2
    nend = maxlength - pllen - nstart
    strout = str_[:nstart] + placeholder + str_[-nend:]
  elif what2keep in ('start', 'begin'):
    strout = str_[:(maxlength - pllen)] + placeholder
  elif what2keep == 'end':
    strout = placeholder + str_[-(maxlength - pllen):]
  else:
    raise ValueError(f"The value for `what2keep` ({what2keep}) is not valid")

  return strout


def find_elm_containing_substrs(substrs, list2search, is_case_sensitive=False, nreq=None,
                                return_strings=False, strmatch='full', raise_except=True,
                                if_multiple_take_shortest=False):
  """
  search the variable names for a certain substring list. Case insensitivity may be enforced

  arguments:
  ----------
  substrs : [None | list of tuple of str | tuple of str | str]
            A (list of) string(s) containing the substrings being looked for
            A tuple element implies an AND search
            Every element in the list is an OR search
            If the first element is an exclamation point (!), this is a NOT included
            *None* (default) will show all variables logged
  list2search : array-like of str
             A list of all variable names
  is_case_sensitive : bool, default=True
                      Whether the search must be case sensitive
  nreq : [None | int], default=None
         The number of elements which must be found. Raises an error if the number requested to be
         found is not exact.
  return_strings : bool, default=False
                   returns the strings themselves instead of the indices that where found
  strmatch : ['all', 'any', 'full'], default='full'
             The type of matching to be done. The options are:
             - 'all': every single part of the string must be present in the found string
             - 'any': at least one part of the string must be present in the found string
             - 'full': the full string must match exactly

  returns:
  --------
  fnd_varslist : (list of) list of str
                 A list of variable names which contain the substrings. In case there is more than
                 one, it is a list of lists of strs
  """
  class ShortestElementTakenWarning(UserWarning):
    """ the shortest element of multiple found is taken (warning) """

  class EmptyListReturnedWarning(UserWarning):
    """ empty list is returned warning """

  class NothingFoundError(Exception):
    """ nothing found error """

  if substrs is None:
    return []

  if isinstance(substrs, str):
    if strmatch == 'all':
      substrs = (*substrs.split(),)
    elif strmatch == 'any':
      substrs = substrs.split()
    elif strmatch == 'full':
      pass
    else:
      raise ValueError(f"The setting for `strmatch` ({strmatch}) is not valid")

  # process fully if substrs is not NONE
  list2search = arrayify(list2search)
  if substrs is None:
    list2search_fnd = list2search.copy()

  else:
    substrs = _convert_to_list_of_tuples(substrs)

    if not is_case_sensitive:
      list2search_sens = list2search.copy()
      # make substrs/list2search lower case
      substrs = [tuple([substr.lower() for substr in subtup],) for subtup in substrs]
      list2search = [elm.lower() for elm in list2search]

    ifnd = np.array([], dtype=int)
    for subtup in substrs:
      ifnd_and = None
      for substr in subtup:
        if substr[0] == "!":
          tf_this = [substr[1:] not in varname for varname in list2search]
        else:
          tf_this = [substr in varname for varname in list2search]
        ifnd_and_this = np.argwhere(tf_this).ravel()
        if ifnd_and is None:
          ifnd_and = np.array(ifnd_and_this)
        else:
          ifnd_and = np.intersect1d(ifnd_and, ifnd_and_this)

      ifnd = np.union1d(ifnd, ifnd_and)

    # get the names
    if not is_case_sensitive:
      list2search_fnd = (np.array(list2search_sens)[ifnd]).tolist()
    else:
      list2search_fnd = (np.array(list2search)[ifnd]).tolist()

  # check outputs
  output = ifnd

  if return_strings:
    output = list2search_fnd

  if nreq is not None:
    if len(output) == 0:
      raise NothingFoundError(f"The substr ({substrs}) was not found in : {list2search}")

    elif len(output) == nreq:
      if nreq == 1:
        output = output[0]
    else:
      if raise_except:
        raise ValueError(f"There must be exactly {nreq} ouputs. "
                         + "This case found {len(list2search_fnd)} outputs")
      else:
        if if_multiple_take_shortest:
          # find shortest
          isort = np.argsort([len(elm) for elm in list2search_fnd])
          output = arrayify(output)[isort[:nreq]].tolist()
          warnings.warn(f"{ifnd.size} elements found, while {nreq} was requested."
                        + "The shortest is/are taken! Beware",
                        category=ShortestElementTakenWarning)
        else:
          output = np.array([])
          warnings.warn(f"{ifnd.size} elements found, while {nreq} was requested. "
                        + "Empty list returned! Beware", category=EmptyListReturnedWarning)

  return output


def data_scaling(data, minval=0., maxval=1., func='linear'):
  """
  Scale the data accurding to some minimum and maximum value. Default is a bracket between 0 and 1
  """
  if not isinstance(data, np.ndarray):
    warnings.warn("The data type will be transformed to an array")
    data = arrayify(data)

  # scale to unit interval
  if func == 'linear':
    pass
  elif func == 'pow10':
    data = np.log10(data)
  elif func == 'exp':
    data = np.log(data)
  elif func == 'pow2':
    data = np.log2(data)
  else:
    raise ValueError(f"The `func` keyword value ({func}) is not valid")

  dmin = data.min()
  dmax = data.max()

  drange = dmax - dmin
  wrange = maxval - minval

  dunit = (data - dmin)/drange
  dwanted = dunit*wrange + minval

  return dwanted


def modify_strings(strings, globs=None, specs=None):
  """
  modify the strings in a list or array by two means: global or specific replacements.

  global_replacements will search for the string and replace it with another (sub) string
  specific_replacements will search for a substring and replace it. The latter allows only a single
  substring to be found

  global replacements are performed before specific replacements, so be careful!

  Arguments:
  ----------
  strings : [ array-like of strings | str]
            the string(s) to be modified
  globs : [ None | tuple | list of tuples ], default=None
          The global replacements. This is a simple (but case-INsensitive) substring replace
  specs : [ None | tuple | list of tuples], default=None
          The specific replacements. This uses *find_elm_containing_substrs* to search for a single
          specific string to replace. Exactly 1 can be found, otherwise nothing is done
  """

  modstrings = listify(strings)
  globs = _convert_to_list_of_tuples(globs)
  specs = _convert_to_list_of_tuples(specs)

  if globs is not None:
    for glrep in globs:
      replace = glrep[0]
      by = glrep[1] if glrep[1] is not None else ''
      modstrings = [re.sub(replace, by, str_, flags=re.I).strip() for str_ in modstrings]

  if specs is not None:
    for reptup in specs:
      if len(reptup) == 2:
        reptup = (*reptup, 'all')

      ifnd = find_elm_containing_substrs(reptup[0], modstrings, nreq=1, strmatch=reptup[2],
                                         raise_except=False)
      if ifnd.size == 1:
        modstrings[ifnd] = reptup[1]

  return modstrings


def plot_matrix(matdata, txtdata=None, ax=None, alphas=None, cmap='jetmodb', aspect='equal',
                clabels=None, rlabels=None, show_values=True,
                fmt="{:0.2f}", clim='wide', ccenter=None, grid=True, fontsize=6,
                fontweight='bold', nan_color='w', nan_alpha=1.,
                imkwargs=None, txtkwargs=None, gridkwargs=None):
  """
  create a matrix plot via matshow with some extras like show the values
  """
  # check if int
  is_int = isinstance(matdata[0, 0], (int, np.integer))

  # change default fmt (only if not overwritten by something else!)
  if is_int and fmt == "{:0.2f}":
    fmt = "{:d}"

  # get textdata from matdata
  if txtdata is None:
    txtdata = matdata.copy()

  if alphas is None:
    alphas = np.ones_like(matdata, dtype=float)

  # parameter dictionaries
  imkwargs_ = dict(interpolation='nearest', cmap=cmap, alpha=alphas, aspect=aspect)
  if imkwargs is not None:
    imkwargs_.update(imkwargs)

  txtkwargs_ = dict(fontsize=fontsize,
                    ha='center',
                    va='center',
                    clip_on=True,
                    fontweight=fontweight,
                    bbox={'boxstyle':'square', 'pad':0.0,
                          'facecolor': 'none', 'lw': 0.,
                          'clip_on': True})
  if txtkwargs is not None:
    txtkwargs_.update(txtkwargs)

  gridkwargs_ = dict(color='k',
                      lw=2,
                      ls='-',
                      alpha=0.8)
  if gridkwargs is not None:
    gridkwargs_.update(gridkwargs)

  # process the 'ivalid' argument
  if ccenter is None:
    ccenter = np.mean(bracket(matdata.ravel()))
  else:
    if isinstance(ccenter, str):
      if ccenter.startswith('mean'):
        ccenter = np.mean(matdata)
      elif ccenter.startswith('median'):
        ccenter = np.median(matdata)
      else:
        raise ValueError(f"the value given for 'ccenter' ('{ccenter}') is not valid!\n"
                          + "Try 'mean' or 'median'")
    if clim is None:
      raise ValueError("If a value for 'ccenter' is given, then 'clim' may not be None")

  # calculate possible border options
  maxval = np.nanmax(matdata)
  minval = np.nanmin(matdata)
  maxdev = maxval - ccenter
  mindev = minval - ccenter
  if isinstance(clim, str):
    if clim.startswith('tight'):
      shortest_max = np.nanmin((np.abs(maxdev), np.abs(mindev)))
      clim = [ccenter - shortest_max, ccenter + shortest_max]
    elif clim.startswith('wide'):
      largest_max = np.nanmax((np.abs(maxdev), np.abs(mindev)))
      clim = [ccenter - largest_max, ccenter + largest_max]
    elif clim.startswith('average'):
      avg_max = np.nanmean((np.abs(maxdev), np.abs(mindev)))
      clim = [ccenter - avg_max, ccenter + avg_max]
  else:
    clim = listify(clim)

  if isinstance(nan_color, str):
    nan_color = to_rgb(nan_color)

  # get the axis and shapes
  nr, nc = matdata.shape
  if ax is None:
    _, ax = plt.subplots(1, 1, num=figname('plot_matrix'))

  # convert the data to rgba values
  f_cvals = get_cmap(cmap)
  # convert to scaled images (between 0 and 1)
  climdata = matdata.copy()
  if is_int:
    matrgbas = f_cvals(climdata)
    matrgbas[..., 3] = alphas
  else:
    tf_isinf = np.isinf(climdata)
    climdata[tf_isinf] = np.nan

    climdata -= np.nanmin(climdata)
    climdata /= np.nanmax(climdata)

    climdata = (matdata - clim[0])/(clim[1] - clim[0])
    matrgbas = f_cvals(climdata)

    # add the alphas
    matrgbas[..., 3] = alphas

    tf_isnan = np.isnan(climdata)
    matrgbas[tf_isnan, :3] = nan_color
    matrgbas[tf_isnan, 3] = nan_alpha

  ax.imshow(matrgbas, **imkwargs_)

  # set the grid lines
  ax.grid(False)
  if grid:
    nof_cols, nof_rows = matdata.shape
    rvec = np.r_[:(nof_rows+1)] - 0.5
    cvec = np.r_[:(nof_cols+1)] - 0.5
    rarr, carr = np.meshgrid(rvec, cvec)
    plot_grid(rarr, carr, ax=ax, **gridkwargs_)

  # ----------- TICKS AND LABELS ------------------
  # x axis
  ax.set_xticks(np.r_[:nc])
  if clabels is None:
    ax.set_xticklabels(ax.get_xticks(), fontsize=7)
  else:
    ax.set_xticklabels(clabels, fontsize=7, rotation=45, va='top', ha='right')

  # y axis
  ax.set_yticks(np.r_[:nr])
  if rlabels is None:
    ax.set_yticklabels(ax.get_yticks(), fontsize=7)
  else:
    ax.set_yticklabels(rlabels, fontsize=7)
  ax.tick_params(axis='both', which='major', length=0)

  # make minor ticks for dividing lines
  ax.set_xticks(np.r_[-0.5:nc+0.5:1], minor=True)
  ax.set_yticks(np.r_[-0.5:nr+0.5:1], minor=True)

  # ----------- SHOW VALUES ----------------------------
  if show_values:
    # get the colormap
    f_cvals = get_cmap(cmap)

    scaledgrays = rgb2gray(matrgbas[..., :3])
    for irow in range(nr):
      for icol in range(nc):

        cellval = scaledgrays[irow, icol]
        # check color
        if cellval < 0.5:
          color = [0.95, 0.95, 0.95]
        else:
          color = [0., 0., 0]
        ax.text(icol, irow, fmt.format(txtdata[irow, icol]), color=color, **txtkwargs_)

  plt.show(block=False)
  plt.draw()

  return ax


def get_file(filepart=None, dirname=None, ext=None):
  """
  get the file if only a part is given, otherwise it is transparent
  """

  # handle missing dirname
  if dirname is None:
    dirname = ''

  # handle missing ext or just without a leading dot
  if ext is None:
    ext = ''

  # if no filepart is given go immediately to select_file()
  if filepart is None:
    files_found = []
  else:

    # work on the given filepart -> split off extension if present
    filepart, given_ext = os.path.splitext(filepart)
    if ext is None:
      ext = given_ext
    # check if it has a leading dot .
    elif not ext.startswith('.'):
      ext = "." + ext

    # load the files
    files_found = glob.glob(os.path.join(dirname, filepart + "*" + ext))

  # ------------- files_found determines what's next ----------------------------
  # check if anything was found
  if len(files_found) == 1:
    filename = files_found[0]
  elif len(files_found) > 1:
    print("Multiple files found. Select the wanted one")
    for ifile, file in enumerate(files_found):
      print(f"[{ifile}] {file}")
    index_chosen = int(input("Select the file to load: "))
    filename = files_found[index_chosen]
  else:
    filetypes = [("All files", "*.*")]
    defaultextension = None
    if ext.startswith('.'):
      filetypes = [(f"{ext} files", ext)] + filetypes
      defaultextension = ext

    filename = select_file(initialdir=dirname, filetypes=filetypes, title="select a file",
                           defaultextension=defaultextension)

  if not os.path.exists(filename):
    raise FileNotFoundError(f"The file *{filename}* does not exist")

  return filename


def wrap_string(string, maxlen, glue=True, offset=None, offset_str=' '):
  """
  break a string and introduce a newline (\n)
  """
  # corner case: string short enough
  if len(string) <= maxlen:
    return string

  if offset is None:
    offset = 0

  # store the lines separately and return integrated line if needed
  lines = []

  # remove leading and trailing spaces
  string = string.strip()
  # get indices of spaces
  leftover = string

  # loop until broken because nothing was left
  while True:
    # check the number of characters left
    nof_left = len(leftover)

    # check if the ibreak is further than the number left
    if maxlen > nof_left:
      ibreak = nof_left - 1

    # if the chunck is still too large
    else:
      # find where there is a space
      ispaces = np.array([pos for pos, char in enumerate(leftover) if char == ' '])

      # what to do if there are no spaces?
      if ispaces.size > 0:
        ispaces = np.delete(ispaces, np.nonzero(ispaces > maxlen))

        # if there is a space which can be used, set it to that one
        if ispaces.size > 0:
          ibreak = ispaces[-1]
        else:
          ibreak = maxlen - 1

    # give the line and the remaining leftover
    string = leftover[:(ibreak+1)].strip()  # pylint: disable=E0606
    lines.append(string)
    leftover = leftover[(ibreak+1):]

    # if nothing left -> break
    if len(leftover) == 0:
      break

  if glue:
    prefix = offset*offset_str
    lines_pf = listify(lines[0]) + [prefix + line for line in lines[1:]]
    bstring = "\n".join(lines_pf)
    return bstring

  else:
    return lines


def eval_string_of_indices(string):
  """
  evaluate the sting that contains indices, may be a list or a 1:10:10 tyoe thing
  """
  # split into parts
  parts = [part0.strip() for part0 in string.split(',')]
  isels = []
  for part in parts:
    # if syntax a:b[:c]
    if len(part.split(':')) > 1:
      split_parts = [int(index) for index in part.split(':')]

      # if a:b
      if len(split_parts) == 2:
        ifrom, ito = split_parts
        istep = 1
      # else: a:b:c
      else:
        ifrom, ito, istep = split_parts

      # make into a list
      isels += list(np.r_[ifrom:ito:istep])

    # else: single index
    else:  # no smart indexing
      isels.append(int(part))

  if len(isels) == 1:
    isels = isels[0]

  return isels


def select_from_list(list_, multi=False, return_indices=False):
  """
  select an option form a list of options

  arguments:
  ----------
  list_ : array-like
          The array-like from which to select an option
  multi : bool, default=False
          flag that indicates if it is acceptable to select multiple options
  return_indices : bool, default=False
                   whether to return the indices or the options themselves (strings)

  returns:
  --------
  out : array-like of str or ints
        Depending on the *return_indices* flag it returns an array-like containing strings or
        indices
  """
  class MultiError(Exception):
    """ error when only a single selection is valid, but multiple are found """

  for idx, item in enumerate(list_):
    print(f"  [{idx:2d}] {item}")

  # change string depending on multi
  if multi:
    qstring = "select items from the list - multiple items allowed: "
  else:
    qstring = "select a single item in the list: "

  answer = input(qstring)
  indices = eval_string_of_indices(answer)

  # check if an error must be raised
  if multi is False:
    if not np.isscalar(indices):
      raise MultiError(f"{len(indices)} indices returned, but only 1 is allowed")

  # check if the answer makes sense
  valid_indices = np.r_[:len(list_)]
  if np.union1d(valid_indices, indices).size > valid_indices.size:
    raise ValueError(f"The given option indices (={indices}) are not (all) valid options")

  if return_indices:
    return indices
  else:
    return list_[indices]


def markerline(marker, length=None, text=None, doprint=True, edge=None):
  """
  print a header line with some text in the middle

  arguments:
  ----------
  marker : str
           The string with which to make the line
  length : None or int, default=None
           The length of the line, if None, the terminal size is taken as the length
  text : str or None, default=None
         The text to be placed in the center. If None, not text is displayed
  doprint : bool, default=True
            flag to state if the line must be printed or only returned
  edge : None or str, default=None
         The character at the edges of the line. If None edge=marker

  returns:
  --------
  line : str
         The created line
  """
  if length is None:
    length = os.get_terminal_size().columns

  if text is None:
    text = ""

  if edge is None:
    edge = marker

  offset = len(text)
  lsize1 = (length - 2 - offset)//2
  lsize2 = length - 2 - lsize1 - offset

  line = f"{edge}{lsize1*marker}{text}{lsize2*marker}{edge}"

  if doprint:
    print(line)

  return line


def print_in_columns(strlist, maxlen=None, sep='', colwidths=None, print_=True,
                     shorten_last_col=False, hline_at_index=None, hline_marker='-',
                     what2keep='end'):
  """
  print a list of strings as a row with n columns
  """
  if maxlen is None:
    maxlen = os.get_terminal_size().columns

  # check the column widths if they are given
  if colwidths is not None:
    if maxlen != arrayify(colwidths).sum():
      raise ValueError(f"The sum of column widths ({arrayify(colwidths).sum()}) is not equal "
                       + "to the max length ({maxlen})")

  # check how many rows
  strarr = arrayify(strlist)
  nr, nc = strarr.shape

  # get maximum per column in characters
  colsizes_needed = np.zeros((nc,), dtype=np.int_)
  for ic in range(nc):
    for ir in range(nr):
      colsizes_needed[ic] = np.fmax(colsizes_needed[ic], len(strarr[ir, ic]))

  # get total size
  total_size_needed = sum(colsizes_needed) + (2 + len(sep))*(nc - 1)

  # check if the total size does not exceed the available size
  if total_size_needed > maxlen:
    if shorten_last_col:
      colsizes_needed[-1] = colsizes_needed[-1] - total_size_needed + maxlen - 1
    else:
      maxlen = total_size_needed - 2
      # raise ValueError("The total amount of space required does not fit in the available space")

  # build the string
  lines = []
  for ir in range(nr):
    line = ''
    for ic in range(nc):
      sstr = short_string(strarr[ir][ic], colsizes_needed[ic], what2keep=what2keep)
      line += f" {sstr:{colsizes_needed[ic]}s} {sep}"

    # adjust edges
    line = line[1:-1]

    # if print_:
    #   print(line)

    lines.append(line)

  if hline_at_index is not None:
    nof_lines = len(lines)
    hline = hline_marker*(min(maxlen, total_size_needed) + len(sep))
    hline_at_indices_lst = [nof_lines + idx + 1 if idx < 0 else idx
                            for idx in listify(hline_at_index)]
    # get the indices backwards
    hlines_at_indices = np.sort(hline_at_indices_lst)[-1::-1]
    for iline in hlines_at_indices:
      lines.insert(iline, hline)

  if print_:
    for line in lines:
      print(line)
    return None
  else:
    return lines


def pixels_under_line(abcvec, xvec, yvec, mode='ax+by=c', upscale_factor=4):
  """
  find all the pixels under a line for a line equation
  """
  dx = np.mean(np.diff(xvec))
  dy = np.mean(np.diff(yvec))

  upscale_factor = arrayify(upscale_factor)
  # correct for single us factor value
  if upscale_factor.size == 1:
    upscale_factor = upscale_factor*np.ones((2,), dtype=np.int_)

  # normalize resolution
  xveci = xvec/dx
  yveci = yvec/dy

  # make vector 1:x (remove negative values)
  xvecn = xveci - np.min(xveci)
  yvecn = yveci - np.min(yveci)

  # upscale to allow overlap and fill the line
  xvecn = np.r_[xvecn[0]:xvecn[-1]:1/upscale_factor[0]]
  yvecn = np.r_[yvecn[0]:yvecn[-1]:1/upscale_factor[1]]

  # convert mode to 'ax+by=c'
  if mode == 'y=ax+b':
    a = abcvec[0]
    b = -1
    c = -abcvec[1]

  elif mode == 'x=ay+b':
    a = 1
    b = -abcvec[0]
    c = abcvec[1]

  elif mode == 'ax+by+c':
    a = abcvec[0]
    b = abcvec[1]
    c = abcvec[2]
  else:
    raise ValueError(f"The chosen *mode* ({mode}) is not valid")

  # modify abc according to scaling done before
  a = a*dx
  b = b*dy
  c = c + a + b - a*xveci.min() - b*yveci.min()

  size_grid = (yvec.size, xvec.size)
  # check all x positions and find corresponding y positions
  xfnd = (-b*yvecn + c)/a
  xfndr = np.int_(xfnd + 0.5)
  yfnd = (-a*xvecn + c)/b
  yfndr = np.int_(yfnd + 0.5)
  xvecnr = np.int_(xvecn + 0.5)
  yvecnr = np.int_(yvecn + 0.5)

  # pylint: disable=unsubscriptable-object,E1135,E1133
  Ivalid_y = np.argwhere([elm in yvecnr for elm in yfndr]).ravel()
  Ipix_y = np.ravel_multi_index((yfndr[Ivalid_y], xvecnr[Ivalid_y]), size_grid)

  # Check all y positions and find corresponding x positions
  Ivalid_x = np.argwhere([elm in xvecnr for elm in xfndr]).ravel()
  Ipix_x = np.ravel_multi_index((yvecnr[Ivalid_x], xfndr[Ivalid_x]), size_grid)
  # pylint: enable=unsubscriptable-object,E1135,E1133
  # output
  Ipix = np.union1d(Ipix_x, Ipix_y)

  return Ipix


def _axplusbyisc_to_yisaxplusb(coefs):
  """
  convert ax+by=c to y=ax+b
  """
  a_in, b_in, c_in = coefs

  a_out = -a_in/b_in
  b_out = c_in/b_in

  return a_out, b_out


def _axplusbyisc_to_xisayplusb(coefs):
  """
  convert ax+by=c to x=ay+b
  """
  a_in, b_in, c_in = coefs

  a_out = -b_in/a_in
  b_out = c_in/a_in

  return a_out, b_out


def _yisaxplusb_to_axplusbyisc(coefs):
  """
  convert y=ax+b to ax+by=c
  """
  a_in, b_in = coefs
  return -a_in, 1., b_in


def _xisayplusb_to_axplusbyisc(coefs):
  """
  convert x=ay+b to ax+by=c
  """
  a_in, b_in = coefs
  return 1., -a_in, b_in


def _yisaxplusb_to_xisayplusb(coefs):
  """
  convert y=ax+b to x=ax+b
  """
  a_in, b_in = coefs

  return 1./a_in, -b_in/a_in


def _xisayplusb_to_yisaxplusb(coefs):
  """
  convert x=ay+b to y=ax+b
  """
  a_in, b_in = coefs
  return 1./a_in, -b_in/a_in


def convert_line_coefficients(coefs, fmt_in, fmt_out='ax+by=c'):
  """
  convert line formats
  """
  # corner case: input equal to output format
  coefs = [np.float_(coef) for coef in coefs]
  if fmt_in == fmt_out:
    return coefs

  function_table = {'ax+by=c': {'y=ax+b': _axplusbyisc_to_yisaxplusb,
                                'x=ay+b': _axplusbyisc_to_xisayplusb},
                    'y=ax+b': {'ax+by=c': _yisaxplusb_to_axplusbyisc,
                               'x=ay+b': _yisaxplusb_to_xisayplusb},
                    'x=ay+b': {'ax+by=c': _xisayplusb_to_axplusbyisc,
                               'y=ax+b': _xisayplusb_to_yisaxplusb}}
  # select function
  func = function_table[fmt_in][fmt_out]

  # get the coefficients
  coefs = func(coefs)

  return coefs


def check_parallel_lines(line1, line2, fmt1='ax+by=c', fmt2='ax+by=c'):
  """
  check if two lines are parallel
  """
  # convert to ax+by=c
  line1_ = convert_line_coefficients(line1, fmt_in=fmt1, fmt_out='ax+by=c')
  line2_ = convert_line_coefficients(line2, fmt_in=fmt2, fmt_out='ax+by=c')

  if line1_[0] == line2_[0] and line1_[1] == line2_[1]:
    return True
  else:
    return False


def intersection_infinite_lines(line1, line2, fmt1='ax+by=c', fmt2='ax+by=c', no_int_value=None):
  """
  calculate the intersect point of two lines
  """
  # convert to ax+by=c formats

  a1, b1, c1 = convert_line_coefficients(line1, fmt1)
  a2, b2, c2 = convert_line_coefficients(line2, fmt2)

  # corner case: parallel lines
  if check_parallel_lines(line1, line2, fmt1=fmt1, fmt2=fmt2):
    return no_int_value

  # is horizontal line
  if np.isclose(a1, 0.) and np.isclose(b1, 1.):
    y_int = c1
  elif np.isclose(a2, 0.) and np.isclose(b2, 1.):
    y_int = c2
  else:
    y_int = (c2/a2 - c1/a1)/(b2/a2 - b1/a1)

  # is vertical line
  if np.isclose(a1, 1.) and np.isclose(b1, 0.):
    x_int = c1
  elif np.isclose(a2, 1.) and np.isclose(b2, 0.):
    x_int = c2
  else:
    x_int = (c2/b2 - c1/b1)/(a2/b2 - a1/b1)

  return x_int, y_int


def intersection_finite_lines(p1, p2, q1, q2, no_int_value=None):
  """
  calculate the intersection point between two finite lines (or line sections) if any
  """
  # calculate the brackets for p and q regarding x and y positions
  px = bracket(np.array([p1[0], p2[0]]))
  qx = bracket(np.array([q1[0], q2[0]]))
  py = bracket(np.array([p1[1], p2[1]]))
  qy = bracket(np.array([q1[1], q2[1]]))

  # check where/if the infinite lines have an intersection
  linep = line_coefs_from_points(*p1, *p2)
  lineq = line_coefs_from_points(*q1, *q2)

  s_int = no_int_value
  s_int_inf = intersection_infinite_lines(linep, lineq, no_int_value=None)
  if s_int_inf is not None:
    # check if s_int_inf is between the end points of BOTH lines
    is_valid = px[0] <= s_int_inf[0] <= px[1]
    is_valid *= py[0] <= s_int_inf[1] <= py[1]
    is_valid *= qx[0] <= s_int_inf[0] <= qx[1]
    is_valid *= qy[0] <= s_int_inf[1] <= qy[1]
    if is_valid:
      s_int = s_int_inf

  return s_int


def intersection_finite_and_infinite_lines(line, p1, p2, no_int_value=None):
  """
  calculate the intersection of a finite and an infinite line
  """
  # calculate the brackets for p and q regarding x and y positions
  px = bracket(np.array([p1[0], p2[0]]))
  py = bracket(np.array([p1[1], p2[1]]))

  # check where/if the infinite lines have an intersection
  linep = line_coefs_from_points(*p1, *p2)

  s_int = no_int_value
  s_int_inf = intersection_infinite_lines(linep, line, no_int_value=None)
  if s_int_inf is not None:
    # check if s_int_inf is between the end points of BOTH lines
    is_valid = px[0] <= s_int_inf[0] <= px[1]
    is_valid *= py[0] <= s_int_inf[1] <= py[1]
    if is_valid:
      s_int = s_int_inf

  return s_int


def distance_point_to_line(xpt, ypt, line, linefmt='ax+by=c'):
  """
  calculate the distance between a point and a line
  """
  a_line, b_line, c_line = convert_line_coefficients(line, linefmt)

  # c-coefficient indicates the distance
  c_pt = a_line*xpt + b_line*ypt

  # this is the distance (perpendicular)
  delta_c = np.abs(c_pt - c_line)

  return delta_c


def distance_point_to_line_segment_INCORRECT(pt, pline1, pline2):
  """
  calculate the distance between a line SEGMENT and a point

  for the distance between a point and a INFINITE line given by ax+b, see 'distance_point_to_line'

  arguments:
  ----------
  pt : 2-array-like of floats
       The point given as an array-like of 2 elements (x and y)
  pline1 : 2-array-like of floats
           The first end-point of the line segment in (x, y) coordinates
  pline2 : 2-array-like of floats
           The second end-point of the line segment in (x, y) coordinates

  returns:
  --------
  retval : [ float | bool]
           if test_against is None, the return value is the shortest distance between the point
           and the line segment.
           if test_against is a float, the return value is a boolean indicating whether the
           distance is LARGER than the test_against value.
  """
  # break into pieces for easier reading
  x0, y0 = pt
  x1, y1 = pline1
  x2, y2 = pline2

  # algo is taken from wikipedia 'Distance_from_a_point_to_a_line' page!
  numerator = np.abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1))
  denominator = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
  dist = numerator/denominator

  # this distance is the entire line, it must be cut
  return dist


def line_coefs_from_points(x1, y1, x2, y2, fmt='ax+by=c'):
  """
  get the coefficient for a line from two points
  """
  x1 = np.float_(x1)
  x2 = np.float_(x2)
  y1 = np.float_(y1)
  y2 = np.float_(y2)

  dx = x1 - x2
  dy = y1 - y2

  if np.isclose(dx, 0.):
    # is vertical line
    a = 1.
    b = 0.
    c = np.mean([x1, x2])
  elif np.isclose(dy, 0.):
    a = 0.
    b = 1.
    c = np.mean([y1, y2])
  else:
    slope = dy/dx
    offset = np.mean([y1 - slope*x1,
                      y2 - slope*x2])
    a = -slope
    b = 1.
    c = offset

  # output format
  if fmt == 'ax+by=c':
    return a, b, c
  elif fmt == 'y=ax+b':
    a_ = -a/b
    b_ = c/b
    return a_, b_
  elif fmt == 'x=ay+b':
    a_ = -b/a
    b_ = c/a
    return a_, b_
  else:
    raise ValueError(f"The value given for *fmt={fmt}* is not valid.")


def is_point_on_line(lp1, lp2, pt, infinite_line=True):
  """
  check if a point is on a line section
  """
  is_on_line = False
  line = line_coefs_from_points(*lp1, *lp2)

  dist = distance_point_to_line(*pt, line)

  if np.isclose(dist, 0.):
    if infinite_line:
      is_on_line = True
    else:
      # check if it is between the lp1 and lp2
      if np.fmin(lp1[0], lp2[0]) <= pt[0] <= np.fmax(lp1[0], lp2[0]):
        if np.fmin(lp1[1], lp2[1]) <= pt[1] <= np.fmax(lp1[1], lp2[1]):
          is_on_line = True

  return is_on_line


def intersections_line_and_circle(circ, p1, p2, is_segment=False, makeplot=False):
  """
  Find the intersection points (0, 1 or 2) of a line and an circle
  """
  # unpack the circle definition (x, y and radius)
  x1_, y1_ = p1
  x2_, y2_ = p2
  xc, yc, rc = circ

  # correct for the xc, yc not being equal to zero..to be able to use a simpler equation
  x1 = x1_ - xc
  x2 = x2_ - xc
  y1 = y1_ - yc
  y2 = y2_ - yc

  # some helper variables
  dx = x2 - x1
  dy = y2 - y1
  dr = np.sqrt(dx**2 + dy**2)
  D = np.linalg.det(np.array([[x1, x2],
                              [y1, y2]]))

  # calculate the number of intersection points
  det = (rc**2)*(dr**2)-D**2
  if det < 0:
    xis = np.array([])
    yis = np.array([])

  # else: there is at least 1 intersecting point
  else:
    # calculate points of intersection for the infinite line (not a segment yet)
    xis = (D*dy + np.array([1, -1])*np.sign(dy)*dx*np.sqrt((rc*dr)**2 - D**2))/(dr**2)
    yis = (-D*dx + np.array([1, -1])*np.abs(dy)*np.sqrt((rc*dr)**2 - D**2))/(dr**2)

    # correct for the offset again
    xis += xc
    yis += yc

  # check against a segment
  if is_segment:
    is_valids = []
    for xi, yi in zip(xis, yis):
      is_valid = is_point_on_line(p1, p2, (xi, yi), infinite_line=False)
      is_valids.append(is_valid)

    # keep only the valid ones
    xis = xis[is_valids]
    yis = yis[is_valids]

  # plot
  if makeplot:
    colors = ['g', 'b', 'r']
    color = colors[1 + np.sign(det).astype(int)]
    # plot the circle
    _, ax = plt.subplots(1, 1, num=figname("intersections line and circle plot"))
    qplot(ax, (x1_, x2_), (y1_, y2_), 'ko-', aspect='equal')
    if len(xis) > 0:
      qplot(ax, xis, yis, 'ko', mfc='none')
    circart = Circle((xc, yc), rc, fc=color, ec='k')

    ax.add_patch(circart)
    ax.relim()
    ax.autoscale(True)
    plt.draw()
    plt.pause(1e-2)

  return xis, yis


def intersections_line_and_box(bl, tr, line, line_fmt='ax+by=c'):
  """
  calculate the intersection points for a line with a rectangular box
  """

  # handle inputs
  xbl, ybl = bl
  xtr, ytr = tr

  line_ = convert_line_coefficients(line, line_fmt, 'ax+by=c')

  # make 4 lines for the box
  linecoefs = {'left': [1., 0., xbl],
               'bottom': [0., 1., ybl],
               'top': [0., 1., ytr],
               'right': [1., 0., xtr]}

  sectionedges = {'left': [(xbl, ybl), (xbl, ytr)],
                  'bottom': [(xbl, ybl), (ytr, ybl)],
                  'top': [(xbl, ytr), (xtr, ytr)],
                  'right': [(xtr, ybl), (xtr, ytr)]}
  # make the output dictionary
  is_inside = False
  outdict = dict.fromkeys(linecoefs.keys())
  for name in linecoefs:
    p_int = intersection_finite_and_infinite_lines(line_, *sectionedges[name])
    outdict[name] = p_int

    if p_int is not None:
      is_inside = True

  outdict['is_inside'] = is_inside

  return outdict


def is_line_from_points_in_box(bl, tr, pt1, pt2):
  """
  calculate if a line through 2 points is inside a box
  """
  line = line_coefs_from_points(*pt1, *pt2)

  intdict = intersections_line_and_box(bl, tr, line)

  return intdict['is_inside']


def strip_sep_from_string(text, sep='_'):
  """
  process the text parts regarding glueing character (underscore)
  """
  # sep at beginning
  text_ = text[1:] if text.startswith(sep) else text
  _text_ = text_[:-1] if text_.endswith(sep) else text_

  return _text_

# ====================== WARNINGS AND SETTINGS ============================
# set the warnings format
# update the display of warnings
# pylint: disable-next=unused-argument
def _formatted_warning(message, category, filename, lineno, line=None):
  strbuf = "  "
  strline = "-"*30
  message = ("\"" + str(message).strip() + "\"").replace("\n", "\n" + strbuf + " ")
  message = strbuf + message

  # get length of top part
  # pylint: disable=duplicate-string-formatting-argument
  str_top = f"{strline} {category.__name__}, {os.path.basename(filename)}:{lineno} {strline}"
  nchars_top = len(str_top)
  fmt = f"\n{str_top}\n{message}\n{'-'*nchars_top}\n\n"
  return fmt


def set_warnings_format():
  """
  set the format of the warnings to what I am accustomed to
  """
  print("overwriting warnings format .. ", end="")
  warnings.formatwarning = _formatted_warning
  print("done!")

  return None


def set_autolimit_mode():
  """ set the autolimit mode to 'round_numbers' to force limits on the axis' min/max """

  print("Setting the autolimit_mode to 'round_numbers' to force limits on the axis' min/max .. ",
        end="")
  plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
  print("done!")

  return None

# =============== CODE TO RUN ==================================
set_warnings_format()

set_autolimit_mode()
