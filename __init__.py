"""
auxtools module

this module contains all kinds of helpfull scripts that are stripped of any relations with projects
or other thales-based links
"""

# import files
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.special import factorial
import numexpr as ne
import re
from copy import deepcopy

from scipy.fftpack import fftshift
from scipy.signal import find_peaks
import os # noqa
import sys # noqa
import warnings as wn # noqa
import warnings
import datetime as dt
import pdb  # noqa
from warnings import warn
from matplotlib.colors import to_rgb
from PyQt5.QtWidgets import QApplication
from matplotlib.patches import Ellipse
import glob
import pandas as pd
import inspect

# import all subfunctions
from .cmaps import * # noqa

# constants
lightspeed = 299706720.0
boltzmann = 1.3806485279e-23
r_earth = 6371.0088e3
radius_earth = r_earth
d2r = np.pi/180
r2d = 180/np.pi
T0 = 290

confidence_table = pd.Series(
    index=[0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95,
           0.975, 0.99, 0.995],
    data=[0.010, 0.020, 0.051, 0.103, 0.211, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
          np.nan, np.nan, np.nan, 4.605, 5.991, 7.378, 9.210, 10.597])

# interpolate the nan's
confidence_table.interpolate(method='cubic', inplace=True)


# CLASSES
class IncorrectNumberOfFilesFound(Exception):
  pass


# FUNCTIONS
def inspect_dict(dict_, searchfor=None):
  """
  show all key value pairs in a dictionary
  """
  keys = list(dict_.keys())

  if searchfor is not None:
    keys = [key for key in keys if key.find(searchfor) > -1]

  markerline("=", text=" Dictionary inspector ")
  list_to_print = [['NAME', 'TYPE(#)', 'VALUES/CONTENT']]
  for key in keys:
    value = dict_[key]
    if np.isscalar(value):
      if isinstance(value, (int, np.int_)):
        item_for_list = [key, 'integer', '{:d}'.format(value)]
      elif isinstance(value, str):
        item_for_list = [key, 'string', "'{:s}'".format(value.strip())]
      elif isinstance(value, (float, np.float_)):
        item_for_list = [key, 'float', '{:f}'.format(value)]
    elif isinstance(value, dict):
      item_for_list = [key, '{:d}-dict'.format(len(value.keys())),
                       '{:s}'.format(print_list(list(value.keys())))]
    elif isinstance(value, np.ndarray):
      item_for_list = [key, '{}-array ({})'.format(value.shape, value.dtype),
                       print_list(value.ravel())]
    elif isinstance(value, list):
      item_for_list = [key, '{:d}-list ({})'.format(len(value), np.array(value).dtype),
                       print_list(value)]
    elif isinstance(value, tuple):
      item_for_list = [key, '{:d}-tuple ({})'.format(len(value), np.array(value).dtype),
                       print_list(listify(value))]
    elif value is None:
      item_for_list = [key, 'None', 'None']
    elif isinstance(value, object):
      item_for_list = [key, 'object', value.__class__.__name__]
    else:
      print("huh????")
      pdb.set_trace()

    list_to_print.append(item_for_list)
  print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.')

  return None


def inspect_object(obj, searchfor=None, show_methods=True, show_props=True, show_unders=False,
                   show_dunders=False):
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

  markerline("=", text=" Object inspector ")
  print("\nClass: '{:s}' ".format(obj.__class__.__name__))
  print("Docstring: {:s}".format(obj.__doc__.strip()))
  if searchfor is not None:
    print("\nATTENTION: Searching for string: '{:s}' in attribute name".format(searchfor))

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

    print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.')

  if show_props:
    print("\n")
    markerline('-', text=' PROPERTIES ')
    print("")
    list_to_print = [['NAME', 'TYPE(#)', 'VALUES/CONTENT']]
    for propname in props:
      prop = getattr(obj, propname)
      if np.isscalar(prop):
        if isinstance(prop, (int, np.int_)):
          item_for_list = [propname, 'integer', '{:d}'.format(prop)]
        elif isinstance(prop, str):
          item_for_list = [propname, 'string', "'{:s}'".format(prop.strip())]
        elif isinstance(prop, (float, np.float_)):
          item_for_list = [propname, 'float', '{:f}'.format(prop)]
      elif isinstance(prop, dict):
        item_for_list = [propname, '{:d}-dict'.format(len(prop.keys())),
                         '{:s}'.format(print_list(list(prop.keys())))]
      elif isinstance(prop, np.ndarray):
        item_for_list = [propname, '{}-array ({})'.format(prop.shape, prop.dtype),
                         print_list(prop.ravel())]
      elif isinstance(prop, list):
        item_for_list = [propname, '{:d}-list ({})'.format(len(prop), np.array(prop).dtype),
                         print_list(prop)]
      elif isinstance(prop, tuple):
        item_for_list = [propname, '{:d}-tuple ({})'.format(len(prop), np.array(prop).dtype),
                         print_list(listify(prop))]
      elif prop is None:
        item_for_list = [propname, 'None', 'None']
      elif isinstance(prop, object):
        item_for_list = [propname, 'object', prop.__class__.__name__]
      else:
        print("huh????")
        pdb.set_trace()

      list_to_print.append(item_for_list)
    print_in_columns(list_to_print, what2keep='begin', hline_at_index=1, hline_marker='.')
  markerline("=", text=" End of class content ")


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


def interpret_sequence_string(seqstr, lsep=",", rsep=':', typefcn=float, check_if_int=True):
  """
  interpret a sequence string like '0, 10, 3' or '10.4' or 10:0.1:20'
  """
  # split the string
  pos_parts_list = seqstr.split(':')
  if len(pos_parts_list) == 1:
    # see if they are separate values
    seq_values = np.array([typefcn(pos.strip()) for pos in seqstr.strip().split(',')])
  elif len(pos_parts_list) == 2:
    begin_, end_ = [typefcn(val) for val in pos_parts_list]
    incr_angle = 1.
    seq_values = np.arange(begin_, end_+0.001, incr_angle)
  elif len(pos_parts_list) == 3:
    begin_, incr_, end_ = [typefcn(val) for val in pos_parts_list]
    seq_values = np.arange(begin_, end_+0.001, incr_)
  else:
    raise ValueError("The values in the sequence '{:s}' cannot be determined"
                     .format(seqstr))

  if check_if_int:
    is_int_seq = np.alltrue([elm.is_integer() for elm in seq_values])
    if is_int_seq:
      seq_values = np.int_(seq_values)

  return seq_values


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


def plot_grid(data_cplx, *args, ax=None, **kwargs):
  """
  plot a grid from a 2D set of complex data
  """
  xmeas = np.real(data_cplx)
  ymeas = np.imag(data_cplx)

  nr, nc = data_cplx.shape

  # plot a grid of lines
  if ax is None:
    ax = plt.gca()

  for irow in range(nr):
    # plot horizontal line
    ax.plot(xmeas[irow, :], ymeas[irow, :], *args, **kwargs)
  for icol in range(63):
    # plot vertical line
    ax.plot(xmeas[:, icol], ymeas[:, icol], *args, **kwargs)

  plt.show(block=False)

  return ax


def add_text_inset(text_inset_strs_list, x=None, y=None, loc='upper right', ax=None,
                   ha='right', va='top', left_align_lines=True, boxcolor=[0.8, 0.8, 0.8],
                   fontweight='normal', fontsize=8, fontname='monospace', fontcolor='k'):
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
      xpos = 0.02

  if ypos is None:
    if loc.lower().find("upper") or loc.lower().find("top"):
      ypos = 0.98
    elif loc.lower().find("lower") or loc.lower().find("bottom"):
      ypos = 0.02

  # get axees
  if ax is None:
    ax = plt.gca()

  # info is on the right side, calculate the offset
  if ha == 'right' and left_align_lines:
    # determine the size of the box
    nof_chars_right_box = max([len(str_) for str_ in text_inset_strs_list])
    text_inset_strs_list = ['{{:<{:d}s}}'.format(nof_chars_right_box).format(str_) for str_ in
                            text_inset_strs_list]

  # glue the lines
  text_inset_text = '\n'.join(text_inset_strs_list)

  # add the text to the axes
  ax.text(xpos, ypos, text_inset_text, fontsize=fontsize, fontweight=fontweight,
          fontname=fontname, ha=ha, va=va, bbox=dict(boxstyle="Round, pad=0.2", ec='k',
                                                     fc=boxcolor),
          transform=ax.transAxes, color=fontcolor)

  plt.draw()

  return ax


def plot_cov(data_or_cov, plotspec='k-', ax=None, center=None, nof_pts=101, fill=False,
             conf=0.67, remove_outliers=True, **kwargs):
  """
  plot the covariance matrix
  sf = 5.99 corresponds to the 95% confidence interval
  """
  cov_to_calc = False
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
      tf_valid_i = ~find_outliers(data[0])
      tf_valid_q = ~find_outliers(data[1])
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

  t = np.linspace(0, 2*np.pi, nof_pts, endpoint=True)
  eigvals, eigvecs = np.linalg.eig(cov)

  # create non-skewed ellipse
  xy_unit_circle = np.array([np.cos(t), np.sin(t)])

  # make scale factor
  sf = np.interp(conf, confidence_table.index, confidence_table.values)
  xy_straight_ellipse = np.sqrt(sf*eigvals).reshape(2, 1)*xy_unit_circle
  xy_ellipse = eigvecs@xy_straight_ellipse  # noqa
  xt_, yt_ = xy_ellipse

  # add the center point
  xt = xt_ + center[0]
  yt = yt_ + center[1]
  ax.plot(xt, yt, plotspec, **kwargs)

  if fill:
    ax.fill(xt, yt, plotspec, **kwargs)
  plt.show(block=False)
  plt.draw()

  return ax, cov


def print_list(list2glue, sep=', ', pfx='', sfx='', floatfmt='{:f}', intfmt='{:d}',
               strfmt='{:s}', cplxfmt='{:f}', maxlen=None, **short_kws):
  """
  glue a list of elements to a string
  """
  types_conv_dict = {str: strfmt,
                     int: intfmt,
                     np.int64: intfmt,
                     float: floatfmt,
                     np.complex_: cplxfmt,
                     complex: cplxfmt,
                     np.bool_: '{}',
                     bool: '{}',
                     dict: "<dict>"}

  # check the three types
  fmtlist = list2glue.copy()
  for key in types_conv_dict.keys():
    fmtlist = [types_conv_dict[key] if isinstance(elm, key) else elm for elm in fmtlist]

  output_parts = [fmtstr.format(elm) for (fmtstr, elm) in zip(fmtlist, list2glue)]
  output_string = pfx + sep.join(output_parts) + sfx

  if maxlen is not None:
    output_string = short_string(output_string, maxlen, **short_kws)

  return output_string


def print_dict(dict2glue, sep=": ", pfx='', sfx='', glue_list=False, glue="\n", floatfmt='{:f}',
               intfmt='{:d}', strfmt='{:s}', maxlen=None, **short_kws):
  """
  print a dict
  """
  types_conv_dict = {str: strfmt,
                     int: intfmt,
                     np.int64: intfmt,
                     float: floatfmt,
                     np.ndarray: '{}'}

  # check the three types
  nof_elms = len(dict2glue)
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
      if type(value) in types_conv_dict.keys():
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


def RENAME_FILES_AND_FOLDERS_NOT_FINISHED(basedir, str2replace, replacement, recursive=True,
                                          dryrun=True, verbose=True):
  """
  rename the files and folders encountered
  """
  if not os.path.exists(basedir):
    raise FileNotFoundError("The basedir ({:s}) is not found!".format(basedir))

  contents_org = os.listdir(basedir)
  contents_fnd = [name for name in contents_org if str2replace in name]
  contents_new = []
  contents_old = []
  dirs = []
  results = []
  for name in contents_fnd:
    new_name = name.replace(str2replace, replacement)
    if verbose:
      print("{:s} --> {:s}".format(name, new_name))
    contents_new.append(new_name)
    contents_old.append(name)
    dirs.append(basedir)
    if recursive and os.path.isdir(os.path.join(basedir, name)):
      (contents_old_,
       contents_new_,
       dirs_,
       results_) = rename_files_and_folders(os.path.join(basedir, name), str2replace, replacement,
                                            recursive=recursive, dryrun=dryrun, verbose=verbose)
      contents_old += contents_old_
      contents_new += contents_new_
      dirs += dirs_
      results += results_

  if dryrun:
    print("Only a dry-run requested. Set *dryrun* to False to actually change files and folder")
  else:
    results = [os.rename(os.path.join(dirname, oldname), os.path.join(dirname, newname))
               for dirname, oldname, newname in zip(dirs, contents_old, contents_new)]

  return contents_old, contents_new, dirs, results


def extract_value_from_strings(input2check, pattern2match, output_fcn=str,
                               notfoundvalue=None):
  """
  extract values from a list of strings
  """
  list_of_strings = listify(input2check)
  fmt = conv_fmt(pattern2match)

  # do the search
  search_results = [re.search(fmt, str_) for str_ in list_of_strings]

  # get the strings
  value_strings = [res.group() if res is not None else notfoundvalue for res in search_results]

  # make the values
  values = [output_fcn(valstr) if valstr is not None else notfoundvalue
            for valstr in value_strings]

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
  pyfmt = pyfmt.replace('+', '\\+')
  py2re = {'*': '\S+',
           '%c': '.',
           '%nc': '.{n}',
           '%d': '[-+]?\d+',
           '%e': '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%E': '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%f': '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '%g': '[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?',
           '$i': '[-+]?(0[xX][\dA-Fa-f]+|0[0-7]*|\d+)',
           '%o': '[-+]?[0-7]+',
           '%n': '[0-9_]*',
           '%w': '[a-zA-z]*',
           '%u': '\d+',
           '%s': '\S+',
           '%x': '[-+]?(0[xX])?[\dA-Fa-f]+',
           '%X': '[-+]?(0[xX])?[\dA-Fa-f]+'}

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
  for key in py2re.keys():
    refmt = refmt.replace(key, py2re[key])

  return refmt


def find_pattern(pattern, list_of_strings, nof_expected=None, nof_expected_mode='exact',
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

  if nof_expected is None:
    pass

  elif isinstance(nof_expected, (int, np.int_, np.int)):
    errstr = ("The expected number of found files is ({}), but only ({}) are found. "
              .format(nof_expected, nof_found)
              + "This is incompatible with *nof_expected_mode*=`{}`".format(nof_expected_mode))
    if nof_found == nof_expected:
      pass
    elif nof_found > nof_expected:
      if nof_expected_mode == 'min':
        pass
      elif nof_expected_mode in ('max', 'exact'):
        raise IncorrectNumberOfFilesFound(errstr)
    elif nof_found < nof_expected:
      if nof_expected_mode == 'max':
        pass
      elif nof_expected_mode in ('min', 'exact'):
        raise IncorrectNumberOfFilesFound(errstr)
    else:
      raise FileNotFoundError(errstr)

  # check if I have to squeeze
  if squeeze:
    if nof_found == 1:
      valid_filenames = valid_filenames[0]

  # return the stuff
  return valid_filenames


def find_filename(filepattern, dirname, nof_expected=1, nof_expected_mode='exact',
                  squeeze=True):
  """
  find a (python) formatted file name in a directory
  """
  # get the contents of the folder
  contents = os.listdir(dirname)

  found_strings = find_pattern(filepattern, contents, nof_expected=nof_expected,
                               nof_expected_mode=nof_expected_mode, squeeze=squeeze)

  return found_strings


def pconv(dirname):
  """
  convert the path in windows format to linux
  """
  win2lin = {'s:': '/work',
             'm:': '/home/dq968/'}

  # replace all backslashes with forward slashes
  dirname = dirname.replace('\\', os.sep)

  for wkey, sub in win2lin.items():
    dirname = dirname.replace(wkey, sub)
    dirname = dirname.replace(wkey.upper(), sub)

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

  if isinstance(_args[0], int):
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
      warn("There is no `exact` match for value = {}. Taking the closest value = {}".
           format(value_wanted, values[ifnd]))

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
  output = deepcopy(input_)

  # from single elements to array of 1 element
  if np.ndim(input_) == 0:
    output = [input_,]

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
        dtype = np.object
      output = np.array(output, dtype=dtype)
  elif array_like == 'tuple':
    output = (*output,)
  else:
    raise ValueError("The array_like given ({}) is not valid".format(array_like))

  return output


def check_types_in_array_like(array_like):
  """
  check the type of all elements in an array-like
  """
  types = set()
  [types.add(type(elm)) for elm in array_like]

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
    spacing = nof_points/(nof_ints+1)
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


def goto(subject=None, must_make_choice=True):
  '''
  goto a certain path belonging to a project. Path is added to sys.path as well

  If *subject=None*, all available `gotos` are shown after which the user may select a subject

  Arguments:
  ----------
  subject : str or None, default=None
            A string indicating the subject or project. Names can be shown when value is None
  must_make_choice : bool, default=True
                     If set to True, the user is asked to make a choice based on the list given.
                     If set to False, only the list is displayed

  Returns:
  --------
  None
  '''
  def _provide_dict_key():

    print('The following `gotos` are available:\n')
    for idx, (key, pth) in enumerate(sorted(gotos.items())):
      # check if path exists
      if os.path.exists(os.path.join(base, pth[7:])):
        prefix = '  '
      else:
        prefix = '  <<NOT AVAIL.>>'
      # check if it has an alias
      if key in full_names.keys():
        print('{}[{:2d}] {:s} ("{:s}")-> {:s}'.format(prefix, idx, key, full_names[key], pth))
      else:
        print('{}[{:2d}] {:s} -> {:s}'.format(prefix, idx, key, pth))

    print('\n<base> = "{}"\n'.format(base))

    if must_make_choice:
      answer = dinput("Please select a subject by index, *None* is no choice: ", None,
                      include_default_in_question=True)

      # get key from answer
      if answer is not None:
        key = list(sorted(gotos.keys()))[np.int(answer)]
      else:
        key = None

    return key

  user = os.environ['USER'].lower()

  # base path
  if user == 'joris':
    base = os.path.join(os.environ['HOME'], 'Documents', 'thales')
  elif user == 'dj754':
    base = os.path.join(os.environ['HOME'], 'mydoc', 'thales')

  # dictionary of gotos (use full names)
  gotos = dict(efocus='<base>/non_system_specific/efocus/python/',
               qr='<base>/non_system_specific/quantum_radar/python/',
               dsg='<base>/non_system_specific/distributed_signal_generation/python/',
               jbn='<base>/non_system_specific/jammed_beam_nulling/python/',
               cc='<base>/non_system_specific/neural_nets/classification_challenge_ext/python/',
               bgest='<base>/non_system_specific/neural_nets/background_estimation/python/',
               festive='<base>/non_system_specific/festive/python/',
               slicer='<base>/non_system_specific/slicer/python/',
               fla='<base>/flycatcher_mk2/alignment/python/')

  # aliases or abbreviations
  full_names = dict(qr='quantum radar',
                    jbn='jammed beam nulling',
                    dsg='distributed signal generation',
                    cc='classification challenge',
                    bgest='background estimation',
                    fla='flycatcher mk2 alignment')

  if subject is None:  # show contents of dict
    key = _provide_dict_key()

  else:  # subject is not None
    if subject in gotos.keys():
      key = subject
    elif subject in aliases.keys():
      key = aliases[subject]
    else:
      print('The subject "{}" is not valid key or alias\n'
            'Press <enter> to continue...'.format(subject))
      input()  # pseudo-pause
      key = _provide_dict_key()

  if key is not None:
    goto_w_base = gotos[key]
    goto = goto_w_base.replace('<base>', base)

    # check if exists
    if os.path.exists(goto):
      # add to path
      if goto not in sys.path:
        sys.path.append(goto)

      # change to directory
      os.chdir(goto)
      print('\nchanged folder to: ''{}''\n'.format(goto))

      # print .py functions in dir
      print('The following .py files are found:')
      for file in os.listdir():
        if file.endswith('.py'):
          print(' - ', file)

    else:
      raise ValueError('The path "{}" does not exist'.format(goto))

  return None


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
  root = Tk()
  root.withdraw()

  filename = filedialog.askopenfilename(**options)

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
  root = Tk()
  root.withdraw()

  if filetypes is None:
    filetypes = [("text files", "*.txt"),
                 ("JPG files", "*.jpg"),
                 ("PNG files", "*.png"),
                 ("GIF files", "*.png"),
                 ("All files", "*.*")]

  # while True:
  filename = filedialog.asksaveasfilename(defaultextension=defaultextension,
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
  root = Tk()
  root.withdraw()

  dirname = filedialog.askdirectory(**options)

  root.quit()
  root.destroy()

  return dirname


def val2ind(pos, spacing=None, center=False):

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


def spectrum(signal, fs=1., nof_taps=None, scaling=1., center_zero=True, full=True,
             dB=True, Plot=True, **plot_kwargs):
  """
  get the spectrum of a signal
  """
  signal = signal.reshape(-1)
  if nof_taps is None:
    nof_taps = signal.size

  freqs = calc_frequencies(nof_taps, fs=fs, center_zero=center_zero)

  spect = np.abs(np.fft.fft(signal, n=nof_taps))
  if center_zero:
    spect = fftshift(spect)

  if isinstance(scaling, str):
    if scaling == 'default':
      sf = 1.
    elif scaling == 'per_sample':
      sf = nof_taps
    elif scaling == 'normalize':
      sf = np.abs(spect).max()
    else:
      raise NotImplementedError("The value for *scaling={}* is not implemented".format(scaling))
  else:
    sf = np.float_(scaling)

  spect /= sf

  if not full:
    nof_samples = freqs.size
    freqs = freqs[:nof_samples//2]
    spect = spect[:nof_samples//2]

  if dB:
    spect = logmod(spect)

  if Plot:
    # plot the stuff
    if 'ax' in plot_kwargs.keys():
      ax = plot_kwargs.pop('ax')
    else:
      fig = plt.figure(figname('{:d}-point spectrum'.format(freqs.size)))
      ax = fig.add_subplot(111)

    ax.plot(freqs, spect, 'b-', **plot_kwargs)
    plt.show(block=False)

  return freqs, spect


def find_dominant_frequencies(signal, fs, f1p=None, scaling='default',
                              max_nof_peaks=None, min_rel_height_db=10, Plot=False, **plotkwargs):
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
  Plot : bool, default=False
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
  freqs, Ydb_debias = spectrum(signal_unbias, fs=fs_, center_zero=False, scaling='default',
                               full=False, Plot=False, dB=True)

  # calculate the minimum distance between samples
  dP_per_sample = fs_/(2.*nof_samples)
  distance_in_P = 0.4
  distance_in_samples = 1 + np.int(0.5 + distance_in_P/dP_per_sample)

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

  if Plot:
    if 'ax' in plot_kwargs.keys():
      ax = plot_kwargs['ax']
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

    xs, ys = spectrum(signal, fs=fs_, center_zero=False, scaling=scaling, full=False,
                      dB=True, Plot=False)

    ax.plot(xs, Ydb_debias - offset, 'k--')
    ax.plot(xs, ys, 'b-')
    # plot_spectrum(signal, 'b-', fs=fs_, center_zero=False, scaling=scaling, full=False,
    #               ax=ax)
    ax.plot(fpeaks, peakvals, 'ro', mfc='none')
    threshold = ys.max() - np.abs(min_rel_height_db)
    ax.axhline(threshold, color='g', linestyle='--')
    ax.text(xs[-1], threshold, "threshold @ {:0.1f} dBc".format(float(threshold)),
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
                                is_singleton=True, singinfo=dict(line=0, name=None),
                                output_array=None, flat=False):
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
      _print('{}'.format(shp), output_array=output_array)

      # loop all subs
      for name in arr.ravel()[0].dtype.names:
        if output_array is not None:
          _print(prefix*(level) + name, end='', output_array=output_array)
        else:
          _print('{:3d}. '.format(linecount) + prefix*level + name, end='',
                 output_array=output_array)

        subarr = arr.ravel()[0][name]

        if is_singleton:
          singinfo['line'] = linecount
          singinfo['name'] = name

        linecount = _print_struct_array_compact(subarr, prefix=prefix,
                                                level=level + 1,
                                                linecount=linecount,
                                                is_singleton=is_singleton,
                                                singinfo=singinfo,
                                                output_array=output_array)

  # it is a leave!
  if is_leave:
    # NUMERICAL
    if type(arr) in [float, complex, np.float_, np.float16, np.float32, np.float64, np.float128,
                     np.complex64, np.complex128, np.complex256]:
      type_ = type(arr).__name__
      if is_singleton:
        _print(': {:.2} ({})'.format(arr, type_), output_array=output_array)
      else:
        _print(': ... ({}) ({} @ {:d})'.format(type_, singinfo['name'], singinfo['line']),
               output_array=output_array)

    # INT
    elif type(arr) is int:
      type_ = type(arr).__name__
      if is_singleton:
        _print(': {:d} ({})'.format(arr, type_), output_array=output_array)
      else:
        _print(': ... ({}) ({} @ {})'.format(type_, singinfo['name'], singinfo['line']),
               output_array=output_array)

    # STRING
    elif type(arr) is str:
      type_ = 'str'
      if is_singleton:
        _print(': "{}" ({})'.format(arr, type_), output_array=output_array)
      else:
        _print(': ... ({}) ({} @ {:d})'.format(type_, singinfo['name'], singinfo['line']),
               output_array=output_array)

    # NDARRAY
    elif type(arr) is np.ndarray:
      type_ = type(arr.ravel()).__name__
      if is_singleton:
        if type(arr[0]) in [float, complex, np.float_, np.float16, np.float32, np.float64,
                            np.float128, np.complex64, np.complex128, np.complex256]:
          _print(': {} ({}{}) '.format(subset_str(arr, '{:0.2}'), arr.shape, type_),
                 output_array=output_array)
        else:
          _print(': {} ({}{}) '.format(subset_str(arr), arr.shape, type_),
                 output_array=output_array)
      else:
        _print(': ... ({}{}) ({} @ {:d})'.format(arr.shape, type_, singinfo['name'],
                                                 singinfo['line']), output_array=output_array)

    # MATLAB FUNCTION
    elif type(arr).__name__ == 'MatlabFunction':
      if is_singleton:
        _print(': <A MATLAB function> ({})'.format(type(arr).__name__), output_array=output_array)
      else:
        _print(': ... ({}) ({} @ {})'.format(type(arr).__name__, singinfo['name'],
                                             singinfo['line']), output_array=output_array)

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
      substr += '{}.'.format(shp)

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
    if type(arr) in [float, complex, np.float_, np.float16, np.float32, np.float64, np.float128,
                     np.complex64, np.complex128, np.complex256]:
      type_ = type(arr).__name__
      if is_singleton:
        endstr = ': {:.2} ({})'.format(arr, type_)
      else:
        endstr = ': ... ({})'.format(type_)

    # INT
    elif type(arr) is int:
      type_ = type(arr).__name__
      if is_singleton:
        endstr = ': {:d} ({})'.format(arr, type_)
      else:
        endstr = ': ... ({})'.format(type_)

    # STRING
    elif type(arr) is str:
      type_ = 'str'
      if is_singleton:
        endstr = ': "{}" ({})'.format(arr, type_)
      else:
        endstr = ': ... ({})'.format(type_)

    # NDARRAY
    elif type(arr) is np.ndarray:
      type_ = type(arr.ravel()).__name__
      if is_singleton:
        if type(arr[0]) in [float, complex, np.float_, np.float16, np.float32, np.float64,
                            np.float128, np.complex64, np.complex128, np.complex256]:
          endstr = ': {} ({}{}) '.format(subset_str(arr, '{:0.2}'), type_, arr.shape)
        else:
          endstr = ': {} ({}{}) '.format(subset_str(arr), type_, arr.shape)
      else:
        endstr = ': ... ({}{})'.format(type_, arr.shape)

    # MATLAB FUNCTION
    elif type(arr).__name__ == 'MatlabFunction':
      if is_singleton:
        endstr = ': <A MATLAB function> ({})'.format(type(arr).__name__)
      else:
        endstr = ': ... ({})'.format(type(arr).__name__)

    # NOT YET DEFINED STUFF WILL RAISE AN EXCEPTION
    else:
      raise TypeError('The type is not expected')

    if output_array is not None:
      output_array.append(substr + endstr)
    else:
      print('{:d}. '.format(linecount) + substr + endstr)

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
      _print('{}'.format(shp), output_array=output_array)

      # loop all subs
      for isub, subarr in enumerate(arr.ravel()):
        for name in subarr.dtype.names:
          if output_array is not None:
            prefix_this = level*prefix
          else:
            prefix_this = '{:3d}. '.format(linecount) + level*prefix
          if arr.ravel().size > 1:
            prefix_this = prefix_this + '[{:02d}] '.format(isub)

          _print(prefix_this + name, end='', output_array=output_array)
          linecount = _print_struct_array_full(subarr[name], prefix=prefix, level=level+1,
                                               linecount=linecount, output_array=output_array)

  # it is a leave!
  if is_leave:
    # _print(type(arr))
    if type(arr) in [float, int, complex]:
      type_ = type(arr).__name__
      _print(': {} ({})'.format(arr, type_), output_array=output_array)
    elif type(arr) is np.ndarray:
      type_ = type(arr.ravel()).__name__
      if arr.ndim == 0:
        _print(': {} ({:d}D {}'.format(arr[()], arr.ndim, type_), output_array=output_array)
      else:
        if arr.size < 5:
          _print(': {} ({:d}D {})'.format(arr, arr.ndim, type_), output_array=output_array)
        else:
          _print(': {} ({:d}D {})'.format(arr.shape, arr.ndim, type_), output_array=output_array)
    elif type(arr) is str:
      type_ = 'str'
      _print(': "{}" ({})'.format(arr, type_), output_array=output_array)
    elif type(arr).__name__ == 'MatlabFunction':
      _print(': A MATLAB function ({})'.format(type(arr).__name__), output_array=output_array)
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
      substr += '{}'.format(shp)

      # loop all subs
      for isub, subarr in enumerate(arr.ravel()):
        for name in subarr.dtype.names:
          if arr.ravel().size > 1:
            prefix_this = '[{:02d}].'.format(isub)
          else:
            prefix_this = '.'

          # print(prefix_this + name, end='')
          linecount = _print_struct_array_flat_full(subarr[name],
                                                    substr=substr + prefix_this + name,
                                                    output_array=output_array, linecount=linecount)

  # it is a leave!
  if is_leave:
    if type(arr) in [float, int, complex]:
      type_ = type(arr).__name__
      endstr = (': {} ({})'.format(arr, type_))
    elif type(arr) is np.ndarray:
      type_ = type(arr.ravel()).__name__
      # endstr = ('{}'.format(arr[:4]), end='')
      if arr.ndim == 0:
        endstr = (': {} ({:d}D {}'.format(arr[()], arr.ndim, type_))
      else:
        if arr.size < 5:
          endstr = (': {} ({:d}D {})'.format(arr, arr.ndim, type_))
        else:
          endstr = (': {} ({:d}D {})'.format(arr.shape, arr.ndim, type_))
    elif type(arr) is str:
      type_ = 'str'
      endstr = (': "{}" ({})'.format(arr, type_))
    elif type(arr).__name__ == 'MatlabFunction':
      endstr = (': A MATLAB function ({})'.format(type(arr).__name__))
    else:
      raise TypeError('The type is not expected')

    if output_array is not None:
      output_array.append(substr + endstr)
    else:
      print('{:d}. '.format(linecount) + substr + endstr)

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
  if type(arr) in [list, tuple]:
    arrsize = len(arr)
  elif type(arr) is np.ndarray:
    arrsize = arr.size

  if arrsize <= min_req_elms:
    subset_str = '[' + ', '.join([fmt.format(elm) for elm in arr]) + ']'
  else:
    subarr_start = ', '.join([fmt.format(elm) for elm in arr[:nof_at_start]])
    subarr_end = ', '.join([fmt.format(elm) for elm in arr[-nof_at_end:]])

    subset_str = '[' + subarr_start + ',... ,' + subarr_end + ']'

  return subset_str


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

  def pressed_return(event):
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
    if types[irow] in (np.float_, float):
      tkvar.append(tk.DoubleVar(master, value=defaults[irow]))

    elif types[irow] in (np.int_, int):
      tkvar.append(tk.IntVar(master, value=defaults[irow]))

    elif types[irow] == str:
      tkvar.append(tk.StringVar(master, value=defaults[irow]))

    else:
      raise ValueError('The type "{}" is not recognized'.format(types[irow]))

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
  except:  # noqa
    print('unknown exception raised. None returned')
    return None

  if len(out) == 1:
    returnval = out[0]
  else:
    returnval = out

  master.destroy()
  return returnval


def rms(signal, axis=None):
  '''
  Calculate the root-mean-squared value of a signal in time

  Arguments:
  ----------
  signal : ndarray of floats or complex floats
           the signal in time
  axis : None or int, default=-1
         The axis along which the RMS value must be determined. In case *None* the RMS value will
         be determined for all elements in the array

  Returns:
  --------
  s_rms : float
         The rms value of the signal
  '''

  # if axis is None, take rms over entire array
  if axis is None:
    signal = signal.reshape(-1)
    axis = -1

  if type(signal) is np.ndarray:
    is_complex = signal.dtype is np.complex128
  else:
    raise TypeError('The type of the signal must be a numpy.ndarray')

  if is_complex:
    i_rms = np.sqrt(np.nanmean(np.real(signal)**2, axis=axis))
    q_rms = np.sqrt(np.nanmean(np.imag(signal)**2, axis=axis))
    s_rms = i_rms + 1j * q_rms
  else:
    s_rms = np.sqrt(np.nanmean(signal**2, axis=axis))

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

  if type(phase) in [list, tuple]:
    phase = np.array(phase)
  elif type(phase) is np.ndarray:
    pass
  else:
    phase = np.array(phase)

  phase = phase.astype(float)

  if phase_units == 'deg':
    phase *= np.pi / 180
  elif phase_units == 'rad':
    pass
  else:
    raise ValueError('The phase units "{:s}" are not accepted. Only "rad" and "deg" are.'
                     .format(phase_units))

  if mode == 'sim':
    nof_samples = int(nof_samples)
    randsamps = np.random.randn(*phase.shape, nof_samples)
    sreal = np.exp(1j * randsamps * phase.reshape(*phase.shape, 1))
    noise_power = np.real(sreal).var(axis=-1) + np.imag(sreal).var(axis=-1)

    snr = 1 / noise_power

  elif mode == 'calc':
    snr = 1 / (np.tan(np.abs(phase))**2)
  else:
    raise ValueError('the *mode* keyword argument only accepts values ''calc'' and ''sim''')

  if snr_units == 'db':
    snr = 10 * np.log10(snr)
  elif snr_units == 'lin':
    pass
  else:
    raise ValueError('The SNR units "{:s}" are not accepted. Only "db" and "lin" are.'
                     .format(snr_units))

  return snr


def snr2phase(snr, snr_units='db', phase_units='rad', mode='calc', nof_samples=1e6):
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
  elif type(snr) is np.ndarray:
    pass
  else:
    snr = np.array(snr)

  if snr_units == 'db':
    snr = 10**(snr / 10)
  elif snr_units == 'lin':
    pass
  else:
    raise ValueError('The SNR units "{:s}" are not accepted. Only "db" and "lin" are.'
                     .format(snr_units))

  if mode == 'sim':
    raise NotImplementedError('The mode ''sim'' is not implemented yet.')
  elif mode == 'calc':
    phase = np.abs(np.arctan(1 / snr))
  else:
    raise ValueError('the *mode* keyword argument only accepts values ''calc'' and ''sim''')

  if phase_units == 'rad':
    pass
  elif phase_units == 'deg':
    phase *= 180 / np.pi
  else:
    raise ValueError('The phase error units "{:s}" is not accepted. Only "rad" and "lin" are.'
                     .format(phase_units))

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
    question = '{0:s} (default={1}): '.format(question, default)

  answer = input(question)

  if not bool(answer):
    output = default
  else:
    output = answer

  return output


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
  nr, nc = data.shape

  # minus via singleton expansion
  imin = np.abs(data.reshape(-1, 1) - states.reshape(1, -1)).argmin(axis=1)

  output = states[imin].reshape(nr, nc)

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
  figname = figname_base
  counter = 1

  # check if name exists
  while plt.fignum_exists(figname):
    figname = figname_base + '[{}]'.format(counter)
    counter += 1

  return figname


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

  return 10*np.log10(np.abs(linval))


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

  return 10**(dbval / mult)


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

  return multiplier * np.log10(np.abs(x))


def bracket(x):
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
  return np.nanmin(x), np.nanmax(x)


def datarange(vals):
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

  mini, maxi = bracket(vals)
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
  noise_gain = np.sum(np.abs(coefs)**2, axis=axis).reshape(-1)
  signal_gain = (np.abs(coefs).sum(axis=axis)**2).reshape(-1)
  snr_gain = signal_gain/noise_gain

  gains = {'noise': noise_gain,
           'signal': signal_gain,
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


def save_animation(anim, filename, fps=30, metadata={'artist': 'Joris Kampman, 2-B Energy'},
                   extra_args={'-vcodec': 'h264', '-preset': 'veryslow', '-crf': '23'}):
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
    raise ValueError("The units={} is not recognized".format(units))

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
    raise ValueError("The value for *orientation* ({}) is not valid".format(orientation))

  return w, h


def get_max_figure_size_inches():
  """
  get the maximum figure size of a figure
  """
  fig = plt.figure()
  plt.get_current_fig_manager().window.showMaximized()
  wfig, hfig = fig.get_size_inches()
  plt.close(fig)

  return wfig, hfig


def get_max_a_size_for_display(fig=None, orientation='landscape', units='mm'):
  """
  get the maximum size with A-ratio that can be displayed
  """
  wfigmax, hfigmax = get_max_figure_size_inches()
  wA0, hA0 = paper_A_dimensions(0, units=units, orientation=orientation)

  # ratio for the conversion of max height to width
  whratio = wA0/hA0

  # Height can always be maximized, width follows from ratio
  height = hfigmax
  width = height*whratio

  return width, height


def move_figure_to_monitor(imon):
  """
  move the figure to a specified monitor
  """
  mng = plt.get_current_fig_manager()
  mng.window.move()


def resize_figure(fig=None, size='amax', sf=1., orientation='landscape'):
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

  # set figure manager
  mng = plt.get_current_fig_manager()

  # reset to normal
  mng.window.showNormal()

  # if; maximize
  if size.endswith("maximize"):
    # first: maximize
    mng.window.showMaximized()

  # else: A paper dimensions (a/b=sqrt(2))
  elif size.startswith('a'):
    if size == 'amax':
      # Height can always be maximized
      width, height = get_max_a_size_for_display(fig=fig, units='inches', orientation=orientation)

    else:
      width, height = paper_A_dimensions(np.int(size[1:]), units='inches', orientation=orientation)

    # use width and height to set the figure size
    fig.set_size_inches(sf*width, sf*height, forward=True)

  # else: witdth and height are given
  else:
    fig.set_size_inches(*size, forward=True)

  return fig


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
    raise ValueError('the *power_units* value: "{}" is not valid\n'.format(power_units),
                     'Only "lin" and "db" are valid choices')

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
    raise ValueError('the *power_units* value: "{}" is not valid\n'.format(power_units),
                     'Only "lin" and "db" are valid choices')

  return np.sqrt(power)


def exp_fast(data):
  """
  calculates the fast exp via numexpr module
  """

  return ne.evaluate('exp(data)')


def qplot_(*args, **kwargs):
  """
  a quicklook plot which will create a new figure
  """
  return qplot(*args, ax=None, **kwargs)


def qplot(*args, ax="hold", center=False, aspect='auto', rot_deg=0.,
          mark_endpoints=False, endpoints_as_text=False, endpoint_color='k', return_kid=False,
          **plotkwargs):
  """
  a quicklook plot

  positional arguments:
  ----------------------
  args : <see matplotlib.pyplot.plot>

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
  kwargs = dict()
  if not isinstance(args[-1], str):
    kwargs = dict(marker='.', color='b', linestyle='-')
  kwargs.update(**plotkwargs)

  # plot in current figure
  if ax is None:
    fig, ax = plt.subplots(1, 1)
  else:
    if isinstance(ax, str) and ax.endswith("hold"):
      ax = plt.gca()

  # set the aspect ratio ('auto' and 'equal' are accepted)
  ax.set_aspect(aspect)

  if len(args) == 1:
    args_ = ()
    if np.iscomplex(args[0]).sum() > 0:
      xs = np.real(args[0])
      ys = np.imag(args[0])
    else:
      ys = args[0]
      xs = np.arange(len(ys))
  elif len(args) >= 2:
    # where does the string start
    if isinstance(args[1], str):
      iarg_start = 1
    else:
      iarg_start = 2

    args_ = args[iarg_start:]
    if iarg_start == 1:
      if np.iscomplex(args[0]).sum() > 0:
        xs = np.real(args[0])
        ys = np.imag(args[0])
      else:
        ys = args[0]
        if np.isscalar(ys):
          xs = 0
        else:
          xs = np.arange(len(ys))
    elif iarg_start == 2:
      xs = args[0]
      ys = args[1]

  if not np.isclose(rot_deg, 0.):
    xs, ys = rot2D(xs, ys, np.deg2rad(rot_deg))

  xs = arrayify(xs)
  ys = arrayify(ys)
  lobj = ax.plot(xs.ravel(), ys.ravel(), *args_, **kwargs)

  if center:
    center_plot_around_origin(ax=ax)
    ax.plot(0, 0, 'k+', markersize=10, markeredgewidth=3)

  if mark_endpoints:
    if endpoint_color.startswith('match'):
      endpoint_color = lobj[0].get_color()

    if endpoints_as_text:
      ax.text(xs.ravel()[0], ys.ravel()[0], 'start', fontsize=8, fontweight='bold', ha='center',
              va='center', alpha=0.5, color=endpoint_color)
      ax.text(xs.ravel()[-1], ys.ravel()[-1], 'end', fontsize=8, fontweight='bold', ha='center',
              va='center', alpha=0.5, color=endpoint_color)
    else:
      ax.plot(xs.ravel()[0], ys.ravel()[0], 's', mfc='none', markersize=10, markeredgewidth=2,
              alpha=0.5, color=endpoint_color)
      ax.plot(xs.ravel()[-1], ys.ravel()[-1], 'o', mfc='none', markersize=10, markeredgewidth=2,
              alpha=0.5, color=endpoint_color)

  plt.show(block=False)
  plt.draw()

  if return_kid:
    return ax, lobj
  else:
    return ax


def center_plot_around_origin(ax=None, aspect='equal'):
  """
  center the plot based on the current content
  """
  # get current bounds
  if ax is None:
    ax = plt.gca()

  xbound = ax.get_xbound()
  ybound = ax.get_ybound()

  dev = np.max([*np.abs(xbound), *np.abs(ybound)])

  ax.set_xlim(left=-dev, right=dev)
  ax.set_ylim(top=dev, bottom=-dev)

  if aspect is not None:
    ax.set_aspect(aspect)

  return ax


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
      raise ValueError('The given axis "{axis}" is not valid'.format(axis=order_rotations[iaxis]))

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
    warnings.warn("The number string *{:s}* is no integer, so it will be rounded first".
                  format(strnum), UserWarning)

  return np.int(0.5 + floatval)


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
    raise ValueError("The string *{%s}* does not end with a valid character".format(numstr))

  dt_delta = dt.timedelta(days=days,
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
    for item in range(len(input_)):
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


def short_string(str_, maxlength, what2keep='edges', placeholder="..."):
  """
  shorten a long string to keep only the start and end parts connected with dots
  """
  strlen = len(str_)
  pllen = len(placeholder)
  if strlen <= maxlength:
    return str_

  # check if it is in the middle
  if what2keep in ('middle', 'center', 'centre'):
    what2keep = strlen//2 - pllen

  if isinstance(what2keep, (np.int_, int, float, np.float_)):
    what2keep = np.int(what2keep + 0.5)
    nof_chars = maxlength - 2*pllen
    istart_keep = np.int(what2keep + 0.5)
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
    raise ValueError("The value for `what2keep` ({}) is not valid".format(what2keep))

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
  class RequestedOutputCountError(Exception):
    pass

  class ShortestElementTakenWarning(UserWarning):
    pass

  class EmptyListReturnedWarning(UserWarning):
    pass

  class NothingFoundError(Exception):
    pass

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
      raise ValueError("The setting for `strmatch` ({}) is not valid".format(strmatch))

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
      raise NothingFoundError("The substr ({}) was not found in : {}".format(substrs, list2search))

    elif len(output) == nreq:
      if nreq == 1:
        output = output[0]
    else:
      if raise_except:
        raise ValueError("There must be exactly {:d} ouputs. This case found {:d} outputs".
                         format(nreq, len(list2search_fnd)))
      else:
        if if_multiple_take_shortest:
          # find shortest
          isort = np.argsort([len(elm) for elm in list2search_fnd])
          output = arrayify(output)[isort[:nreq]].tolist()
          warn("{:d} elements found, while {:d} was requested. The shortest is/are taken! Beware".
               format(ifnd.size, nreq), category=ShortestElementTakenWarning)
        else:
          output = np.array([])
          warn("{:d} elements found, while {:d} was requested. Empty list returned! Beware".
               format(ifnd.size, nreq), category=EmptyListReturnedWarning)

  return output


def data_scaling(data, minval=0., maxval=1., func='linear'):
  """
  Scale the data accurding to some minimum and maximum value. Default is a bracket between 0 and 1
  """
  if not isinstance(data, np.ndarray):
    warn("The data type will be transformed to an array")
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
    raise ValueError("The `func` keyword value ({}) is not valid".format(func))

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
    for irepl, reptup in enumerate(specs):
      if len(reptup) == 2:
        reptup = (*reptup, 'all')

      ifnd = find_elm_containing_substrs(reptup[0], modstrings, nreq=1, strmatch=reptup[2],
                                         raise_except=False)
      if ifnd.size == 1:
        modstrings[ifnd] = reptup[1]

  return modstrings


def improvedshow(matdata, clabels=None, rlabels=None, show_values=True, fmt="{:0.1g}",
                 invalid=None, ax=None, title=None, fignum=None, **kwargs):
  """
  create a matrix plot via matshow with some extras like show the values
  """

  if np.isscalar(invalid):
    invalid = [invalid - np.spacing(invalid), invalid + np.spacing(invalid)]
  nr, nc = matdata.shape
  if ax is None:
    fig, ax = plt.subplots(1, 1, num=figname(fignum))
  else:
    fig = ax.figure

  ax.imshow(matdata, interpolation='nearest', **kwargs)

  ax.set_xticks(np.r_[:nc])
  if clabels is None:
    ax.set_xticklabels(ax.get_xticks(), fontsize=7)
  else:
    ax.set_xticklabels(clabels, fontsize=7, rotation=45, va='top', ha='right')

  ax.set_yticks(np.r_[:nr])
  if rlabels is None:
    ax.set_yticklabels(ax.get_yticks(), fontsize=7)
  else:
    ax.set_yticklabels(rlabels, fontsize=7)
  ax.tick_params(axis='both', which='major', length=0)

  # make minor ticks for dividing lines
  ax.set_xticks(np.r_[-0.5:nc+0.5:1], minor=True)
  ax.set_yticks(np.r_[-0.5:nr+0.5:1], minor=True)

  ax.grid(which='minor', linewidth=1)

  if 'aspect' not in kwargs.keys():
    kwargs['aspect'] = 'auto'
  # show the values in the matrix
  if show_values:
    for irow in range(nr):
      for icol in range(nc):
        cellval = matdata[irow, icol]
        # check if is valid
        if (invalid is None) or not (invalid[0] < cellval < invalid[1]):
          # check color
          # if cellval.sum() < 0.5:
          #   color = [0.75, 0.75, 0.75]
          # else:
          #   color = [0., 0., 0]
          ax.text(icol, irow, fmt.format(matdata[irow, icol]), fontsize=6, ha='center',
                  color='k', va='center', clip_on=True, bbox={'boxstyle':'square', 'pad':0.0,
                                                              'facecolor': 'none', 'lw': 0.,
                                                              'clip_on': True})

  if title is not None:
    ax.set_title(title, fontsize=11, fontweight='bold')

  plt.show(block=False)
  fig.tight_layout()
  plt.draw()

  return fig, ax


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
      print("[{:d}] {:s}".format(ifile, file))
    index_chosen = np.int(input("Select the file to load: "))
    filename = files_found[index_chosen]
  else:
    filetypes = [("All files", "*.*")]
    defaultextension = None
    if ext.startswith('.'):
      filetypes = [("{0:s} files".format(ext), ext)] + filetypes
      defaultextension = ext

    filename = select_file(initialdir=dirname, filetypes=filetypes, title="select a file",
                           defaultextension=defaultextension)

  if not os.path.exists(filename):
    raise FileNotFoundError("The file *{:s}* does not exist".format(filename))

  return filename


def wrap_string(string, maxlen, break_at_space=True, glue=True, offset=None,
                offset_str=' '):
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
    string = leftover[:(ibreak+1)].strip()
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
      split_parts = [np.int(index) for index in part.split(':')]

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
      isels.append(np.int(part))

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
    pass

  for idx, item in enumerate(list_):
    print("  [{:2d}] {}".format(idx, item))

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
      raise MultiError("{:d} indices returned, but only 1 is allowed".format(len(indices)))

  # check if the answer makes sense
  valid_indices = np.r_[:len(list_)]
  if np.union1d(valid_indices, indices).size > valid_indices.size:
    raise ValueError("The given option indices (={}) are not (all) valid options".format(indices))

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

  line = "{:s}{:s}{:s}{:s}{:s}".format(edge, lsize1*marker, text, lsize2*marker, edge)

  if doprint:
    print(line)

  return line


def print_in_columns(strlist, maxlen='auto', sep='', colwidths=None, print_=True,
                     shorten_last_col=True, hline_at_index=None, hline_marker='-',
                     what2keep='end'):
  """
  print a list of strings as a row with n columns
  """
  if maxlen is None:
    maxlen = 100000000
  elif maxlen == 'auto':
    maxlen = os.get_terminal_size().columns

  # check the column widths if they are given
  if colwidths is not None:
    if maxlen != arrayify(colwidths).sum():
      raise ValueError("The sum of column widths ({:d}) is not equal to the max length ({:d})".
                       format(arrayify(colwidths).sum(), maxlen))

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
      raise ValueError("The total amount of space required does not fit in the available space")

  # build the string
  lines = []
  for ir in range(nr):
    line = ''
    for ic in range(nc):
      line += " {:{:d}s} {:s}".format(short_string(strarr[ir][ic],
                                                   colsizes_needed[ic],
                                                   what2keep=what2keep), colsizes_needed[ic], sep)

    # adjust edges
    line = line[1:-1]

    # if print_:
    #   print(line)

    lines.append(line)

  if hline_at_index is not None:
    nof_lines = len(lines)
    hline = hline_marker*(min(maxlen, total_size_needed) + len(sep))
    hline_at_indices_lst = [nof_lines + idx + 1 if idx < 0 else idx for idx in listify(hline_at_index)]
    # get the indices backwards
    hlines_at_indices = np.sort(hline_at_indices_lst)[-1::-1]
    for iline in hlines_at_indices:
      lines.insert(iline, hline)

  if print_:
    for line in lines:
      print(line)

  return lines


def pixels_under_line(abcvec, xvec, yvec, mode='ax+by=c', upscale_factor=4):
  """
  find all the pixels under a line
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
    raise ValueError("The chosen *mode* ({}) is not valid".format(mode))

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

  Ivalid_y = np.argwhere([elm in yvecnr for elm in yfndr]).ravel()
  Ipix_y = np.ravel_multi_index((yfndr[Ivalid_y], xvecnr[Ivalid_y]), size_grid)

  # Check all y positions and find corresponding x positions
  Ivalid_x = np.argwhere([elm in xvecnr for elm in xfndr]).ravel()
  Ipix_x = np.ravel_multi_index((yvecnr[Ivalid_x], xfndr[Ivalid_x]), size_grid)

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
    raise ValueError("The value given for *fmt={}* is not valid.".format(fmt))


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
        print("horizontal coordinate OK")
        if np.fmin(lp1[1], lp2[1]) <= pt[1] <= np.fmax(lp1[1], lp2[1]):
          print("vertical coordinate OK")
          is_on_line = True

  return is_on_line


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
  for name, coefs in linecoefs.items():
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
