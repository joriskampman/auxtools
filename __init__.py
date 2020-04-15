'''
jktools module
'''

# import files
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import numexpr as ne
import re

from scipy.fftpack import fftshift
import os # noqa
import sys # noqa
import warnings as wn # noqa
import warnings
import datetime as dt
import pdb  # noqa
from warnings import warn
from matplotlib.colors import to_rgb

# import all subfunctions
from .coordinate_transforms import *  # noqa
from .cmaps import * # noqa

# constants
lightspeed = 299706720.0
boltzmann = 1.3806485279e-23
r_earth = 6371.0088e3
radius_earth = r_earth
amax_size_inches = (9.82*np.sqrt(2), 9.82)
d2r = np.pi/180
r2d = 180/np.pi
T0 = 290


def split_complex(cplx, sfx=1, sfy=1):
  """
  split a complex point in real and imaginary parts with a scale factor
  """
  return sfx*np.real(cplx), sfy*np.imag(cplx)


def monospace(array, delta=None):
  """
  straighten an array which has rounding errors
  """
  if delta is None:
    delta = np.mean(np.diff(array))

  array = np.r_[:array.size]*delta

  return array


def get_closest_index(value_wanted, values, suppress_warnings=False):
  """
  get the closest index
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
  return make_array_like(input_, 'list')


def tuplify(input_):
  """
  make into a tuple
  """
  return make_array_like(input_, 'tuple')


def arrayify(input_):
  """
  make into array
  """
  return make_array_like(input_, 'np.ndarray')


def make_array_like(input_, array_like):
  """
  make an input into an array like
  """
  output = np.copy(input_)
  # from single elements to array of 1 element
  if np.ndim(input_) == 0:
    output = [input_,]

  # convert to the right type
  if array_like == 'list' or array_like == 'np.ndarray':
    output = [*output,]
    if array_like == 'np.ndarray':
      output = np.array(output)
  elif array_like == 'tuple':
    output = (*output,)

  return output


def color_vector(nof_points, base_color, os=0.25):
  """
  determine a color vector from dark to light around a center color
  """
  base_color = np.array(to_rgb(base_color))

  start_color = np.fmax(0, base_color - base_color.max() + os)
  end_color = np.fmin(1., base_color - base_color.min() + 1 - os)
  # make base vector
  icenter = np.int((nof_points + 0.5)//2)

  cvec = np.zeros((nof_points, 3), dtype=float)
  for iax in range(3):
    cvec[:icenter, iax] = np.linspace(start_color[iax], base_color[iax], icenter, endpoint=False)
    cvec[icenter:, iax] = np.linspace(base_color[iax], end_color[iax], nof_points - icenter)

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
                    check_exists=True):
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

  # while True:
  filename = filedialog.asksaveasfilename(defaultextension=defaultextension,
                                          initialdir=initialdir, title=title,
                                          initialfile=initialfile)

  # if check_exists:
  #   if os.path.exists(filename):
  #     answer = jk.dinput('The file already exists. Overwrite? [y/n]', 'y')
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
  root = Tk()
  root.withdraw()

  dirname = filedialog.askdirectory(**options)

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
  if center_zero:
    taps = np.arange(-nof_taps//2, nof_taps//2)
  else:
    taps = np.arange(nof_taps)

  freqs = taps*(fs/nof_taps)

  return freqs


def plot_spectrum(signal, plotspec='b.-', fs=1., nof_taps=None, mode='default', center_zero=True,
                  ax=None, **plot_kwargs):
  '''
  Plot the spectrum of an input signal (or a filter spec)

  Positional arguments:
  ---------------------
  ... to be filled in an continued ....continued
  '''

  signal = signal.reshape(-1)
  if nof_taps is None:
    nof_taps = signal.size

  freqs = calc_frequencies(nof_taps, fs=fs, center_zero=center_zero)

  spect = np.fft.fft(signal, n=nof_taps)
  if center_zero:
    spect = fftshift(spect)

  if mode == 'default':
    pass
  elif mode == 'per_sample':
    spect /= nof_taps
  elif mode == 'normalize':
    spect /= np.abs(Y).max()

  # plot the stuff
  if ax is None:
    fig = plt.figure(figname('{:d}-point spectrum'.format(nof_taps)))
    ax = fig.add_subplot(111)

  ax.plot(freqs, logmod(spect), plotspec, **plot_kwargs)
  plt.show(block=False)

  return (ax, spect)


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
  ._print_struct_array_compact : prints the array content with flag *verbose=False*
  ._print_struct_array_full : prints the array content with flag *verbose=True*
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

  dtype = type(data)
  if dtype in [list, tuple]:
    data = np.array(data)

  data -= data.min()
  data /= data.max()

  if dtype is list:
    out = list(data)
  elif dtype is tuple:
    out = tuple(data)
  else:
    out = data

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
    master.destroy()

    return None

  if type(strings) is list:
    nof_rows = len(strings)
    if defaults is None:
      defaults = [None] * nof_rows
    if types is None:
      types = [str] * nof_rows
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
    if types[irow] is float:
      tkvar.append(tk.DoubleVar(master, value=defaults[irow]))

    elif types[irow] is int:
      tkvar.append(tk.IntVar(master, value=defaults[irow]))

    elif types[irow] is str:
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
    return out[0]
  else:
    return out


def rms(signal, axis=-1):
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

  return None


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

  return x.min(), x.max()


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
            jktools.amax_size_inches()
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


def tighten(fig=None, orientation='landscape', forward=True):
  '''
  *tighten* is equivalent to numpy.tight_layout() but defined specifically for figures laid-out
  as A-format landscape (via jktools.resize_figure() for instance). The subplots are placed
  such that there is room for a suptitle, labels and plottitles.

  keyword arguments:
  ------------------
  fig [handle] is the figure handle to tighten. In case fig=None, the current figure will be taken
  forward [bool] set to True will (re)draw the figure immediately. set to False will hold drawing

  *tighten* will return None

  See also: jktools.resize_figure(),
            jktools.amax_size_inches,
            matplotlib.pyplot.tight_layout()

  Author: Joris Kampman, Thales NL, 2017
  '''

  if fig is None:
    fig = plt.gcf()

  if orientation == 'landscape':
    top = 0.927
    bottom = 0.063
    left = 0.054
    right = 0.987
    hspace = 0.348
    wspace = 0.297
  elif orientation == 'portrait':
    left = 0.084
    right = 0.973
    top = 0.939
    bottom = 0.044
    wspace = 0.105
    hspace = 0.265

  fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)

  if forward is True:
    plt.draw()

  return None


def resize_figure(fig=None, size=None, orientation='landscape', tight_layout=True):
  '''
  resize_figure sets the figure size such that the ratio for the A-format is kept, while maximizing
  the display on the screen.None

  positional arguments:
  ---------------------
  <none>

  keyword arguments:
  ------------------
  fig         [handle] figure handle. If None is given, the current figure handle will be taken
  size        [None/list(float)] either None or a list of 2 elements containing the width and
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

  See also: jktools.tighten(), jktools.amax_size_inches
  '''

  if fig is None:
    fig = plt.gcf()

  if size is None:
    if orientation == 'landscape':
      fig.set_size_inches(amax_size_inches, forward=True)
    elif orientation == 'portrait':
      fig.set_size_inches((amax_size_inches[1]/np.sqrt(2), amax_size_inches[1]), forward=True)
    else:
      raise ValueError('keyword argument orientation="{}" is not valid.'
                       ' Only "landscape" and "portrait" are.'.format(orientation))
  else:
    fig.set_size_inches(*size, forward=True)

  if tight_layout:
    tighten(fig, orientation=orientation)

  return None


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


def F1_Score(nof_true_pos, nof_false_pos, nof_false_neg):
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


def qplot(*args, newfig=True, **kwargs):
  """
  a quicklook plot
  """
  # set kwargs if no display spec is given
  if isinstance(args[0], str):
    if args[0].startswith(('-n', '--new')):
      fig, ax = plt.subplots(1)
      args = args[1:]

  if not isinstance(args[-1], str):
    kwargs_plot = dict(marker='.', color='b', linestyle='-')
    kwargs.update(**kwargs_plot)

  # plot in current figure
  if newfig:
    fig = plt.figure()
    ax = fig.add_subplot(111)
  else:
    ax = plt.gca()
  ax.plot(*args, **kwargs)
  plt.show(block=False)
  plt.draw()

  return ax


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


def convert_to_list_of_tuples(inpt):
  """
  make an input to a list of tuples
  """
  # make it al into a list of tuple(s)
  if isinstance(inpt, list):
    for item in range(len(inpt)):
      if isinstance(inpt[item], str):
        inpt[item] = (inpt[item],)

  if isinstance(inpt, tuple):
    inpt = [inpt]

  if isinstance(inpt, str):
    inpt = [(inpt,),]

  return inpt.copy()


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


def short_string(str, istart=2, maxlength=8):
  """
  shorten a long string to keep only the start and end parts connected with dots
  """
  if len(str) <= maxlength:
    return str

  return str[:istart] + "..." + str[-(maxlength-istart):]


def find_elm_containing_substrs(substrs, list2search, is_case_sensitive=False, nreq=None,
                                return_strings=False, strmatch='full', raise_except=True,
                                if_multiple_take_shortest=True):
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
  if substrs is None:
    list2search_fnd = list2search.copy()

  else:
    substrs = convert_to_list_of_tuples(substrs)

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
    if len(output) == nreq:
      if nreq == 1:
        output = output[0]
    else:
      raise ValueError("There must be exactly {:d} ouputs. This case found {:d} outputs".
                       format(nreq, len(list2search_fnd)))

  return output


def data_scaling(data, minval=0., maxval=1.):
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
  globs = convert_to_list_of_tuples(globs)
  specs = convert_to_list_of_tuples(specs)

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


