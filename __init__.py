'''
jktools module
'''

# import files
import numpy as np
import tkinter as tk
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os
from scipy.io import loadmat
import numexpr as ne

from scipy.fftpack import fftshift
import os # noqa
import sys # noqa
import warnings as wn # noqa
from functools import reduce
import warnings
import datetime as dt

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


def grazing_angle(h_system, r_slant, earth_model='standard', angle_units='rad'):
  """
  Calculate the grazing angle in radians based on a model for the atmosphere, the system height and
  the slant range (i.e., the measured range)
  """

  if earth_model == 'standard':
    sf = 4/3
  elif earth_model == 'flat':
    sf = np.inf
  elif earth_model == 'no_atmos':
    sf = 1
  else:
    raise ValueError('The *earth_model* keyword argument "{}" is not defined'.format(earth_model),
                     'Valid options are:\n',
                     ' - "standard": earth radius scale factor is 4/3 (standard atmoshpere)\n',
                     ' - "flat": uses INF earth radius, to simulate a flat earth\n',
                     ' - "no_atmos": earth radius scale factor is 1 (no atmosphere)')

  r_earth_c = r_earth*sf

  if np.isinf(r_earth_c):
    phi = np.arcsin(H/r_slant)
  else:
    phi = np.arcsin((h_system**2 + 2*h_system*r_earth_c - r_slant**2)/(2*r_slant*r_earth_c))

  phi[r_slant < h_system] = np.nan
  phi[phi < 0.] = 0.

  if angle_units == 'rad':
    pass
  elif angle_units == 'deg':
    phi *= 180/np.pi
  else:
    raise ValueError('the *angle_units* value "{}" is not valid\n'.format(angle_units),
                     'Valid values are "deg" and "rad"')

  return phi


def slant2ground(h_system, r_slant, earth_model='standard'):
  """
  convert a slant range to a ground range. Very usefull in sea-clutter analysis
  """
  if earth_model == 'standard':
    sf = 4/3
  elif earth_model == 'flat':
    sf = np.inf
  elif earth_model == 'no_atmos':
    sf = 1
  else:
    raise ValueError('The *earth_model* keyword argument "{}" is not defined'.format(earth_model),
                     'Valid options are:\n',
                     ' - "standard": earth radius scale factor is 4/3 (standard atmoshpere)\n',
                     ' - "flat": uses INF earth radius, to simulate a flat earth\n',
                     ' - "no_atmos": earth radius scale factor is 1 (no atmosphere)')

  r_earth_c = r_earth*sf

  if np.isinf(r_earth_c):
    r_ground = r_slant
  else:
    Y = (h_system**2 - r_slant**2)/(2*(h_system + r_earth_c))
    X = np.sqrt(r_earth_c**2 - (Y + r_earth_c)**2)

    angle_earth = np.arctan(X/(r_earth_c + Y))
    angle_earth[r_slant < h_system] = np.nan

    r_ground = angle_earth*r_earth_c/sf

  r_ground[r_slant < h_system] = np.nan

  return r_ground


def ground2slant(h_system, r_ground, earth_mode='standard'):
  """
  ground range to slant range
  """
  if earth_model == 'standard':
    sf = 4/3
  elif earth_model == 'flat':
    sf = np.inf
  elif earth_model == 'no_atmos':
    sf = 1
  else:
    raise ValueError('The *earth_model* keyword argument "{}" is not defined'.format(earth_model),
                     'Valid options are:\n',
                     ' - "standard": earth radius scale factor is 4/3 (standard atmoshpere)\n',
                     ' - "flat": uses INF earth radius, to simulate a flat earth\n',
                     ' - "no_atmos": earth radius scale factor is 1 (no atmosphere)')

  r_earth_c = r_earth*sf

  if np.isinf(r_earth_c):
    r_slant = np.sqrt(h_system**2 + r_ground**2)
  else:
    angle_earth = sf*r_ground/r_earth_c

    X = r_earth_c*np.sin(angle_earth)
    Y = r_earth_c*np.cos(angle_earth) - r_earth

    r_slant = np.sqrt(X**2 + (h_system + Y)**2)

  return r_slant


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


def get_psys(code_or_name=None, squeeze_me=True, must_make_choice=True):
  '''
  Get the Psys data from the matlab .mat file as created by System Engineering (Hugo Kleijer)

  Keyword argument:
  -----------------
  code_or_name : str or None, default=None
                 If *None* the function will print all available Psys files defined here.
                 If a string, it must be a key in the table for which a Psys is defined
  squeeze_me : bool, default=True
               If keyword argument will be copied to underlying *scipy.io.loadmat* keyword
               argument with the same name.
  must_make_choice : bool, default=True
                     Whether or not to be forced to make a choice

  Return:
  -------
  psys : dict
         Contains the Psys data from the matlab .mat file

  '''

  def _provide_dict_key():

    print('The following `psys\'` are available:\n')
    for idx, (key, pth) in enumerate(sorted(lut.items())):
      # check if it has an alias
      if key in lut.keys():
        print('  [{:2d}] {:s} ("{:s}")-> {:s}'.format(idx, key, lut[key], pth))
      else:
        print('  [{:2d}] {:s} -> {:s}'.format(idx, key, pth))

    print('\n<base> = "{}"\n'.format(base))

    if must_make_choice:
      answer = dinput("Please select a project/system by index, *None* is no choice: ", None,
                      include_default_in_question=True)

      # get key from answer
      if answer is not None:
        key = list(sorted(lut.keys()))[int(answer)]
      else:
        key = None

    return key

  base = os.path.join(os.environ['HOME'], 'mydoc', 'thales')
  lut = {'smart_l_mm': os.path.join('<base>', 'smart_l_mm', 'psys', 'Psys_SL_elr.mat'),
         'xpar': os.path.join('<base>', 'xpar', 'psys', 'Psys_Xpar.mat')}
  lut['004ne'] = lut['smart_l_mm']
  lut['029ne'] = lut['smart_l_mm']

  if code_or_name is None:  # show contents of dict
    key = _provide_dict_key()

  else:  # subject is not None
    if code_or_name in lut.keys():
      key = code_or_name
    else:
      print('The system/project "{}" is not valid key or alias\n'
            'Press <enter> to continue...'.format(code_or_name))
      input()  # pseudo-pause
      key = _provide_dict_key()

  if key is not None:
    lut_item_w_base = lut[key]
    lut_item = lut_item_w_base.replace('<base>', base)

    # check if exists
    if os.path.exists(lut_item):

      psys = loadmat(lut_item, squeeze_me=squeeze_me)['Psys'][()]

    else:
      raise ValueError('The psys file "{}" does not exist'.format(lut_item))

  return psys


def grep_struct_array(arr, patterns, verbose=False, match_case=False, and_or='and'):

  # get output list
  output_list = print_struct_array(arr, verbose=verbose, output_in_list=True, flat=True)

  # make a list of patterns if it is a single item
  if type(patterns) in [list, tuple]:
    pass
  else:
    patterns = [patterns]

  # loop over all patterns and make decision based on argument *and_or*
  i_pttns_fnd = []
  for pattern in patterns:
    if match_case:
      isub_pttn_fnd1 = np.array([entry.find(pattern) for entry in output_list])
    else:
      isub_pttn_fnd1 = np.array([entry.lower().find(pattern.lower()) for entry in output_list])
    
    i_pttns_fnd.append(np.argwhere(isub_pttn_fnd1 != -1).flatten().tolist())

  if and_or.lower() == 'and':
    i_pttn_fnd = reduce(np.intersect1d, i_pttns_fnd)
  elif and_or.lower() == 'or':
    i_pttn_fnd = reduce(np.union1d, i_pttns_fnd)
  else:
    raise ValueError('The keyword argument *and_or={}* is not valid. Choices are `and` or `or`'.
                     format(and_or))

  # loop and print the found pattern entries
  for ientry in i_pttn_fnd:
    print(output_list[ientry])

  return None


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


def fd2r(deg):
  '''
  Quick conversion of degrees to radians

  Arguments:
  ----------
  deg : ndarray of floats or float
        the angles in degrees

  Returns:
  --------
  rad : ndarray of floats or float
        The angles in radians

  See Also:
  ---------
  jktools.r2d : converts radians to degrees
  '''

  return deg * np.pi / 180


def fr2d(rad):
  '''
  Quick conversion of radians to degrees

  Arguments:
  ----------
  rad : float, ndarray of floats
        The angles in radians

  Returns:
  --------
  deg : float, ndarray of floats
        the angles in degrees

  See Also:
  ---------
  jktools.d2r : convert degrees to radians
  '''

  return rad * 180 / np.pi


def primes(n_max, n_min=2):
  '''
  Calculates the primes between n_max and n_min

  Arguments:
  ----------
  n_max : int
          The upper boundary
  n_min : int [optional]
          The lower boundary

  Returns:
  --------
  primes_list : ndarray
                A 1D ndarray containing all primes between and including n_min and n_max

  See Also:
  jktools.isprime : checks if an integer value is a prime value
  '''

  tfs = np.ones(n_max + 1, dtype=np.bool)

  tfs[:2] = False
  for n in range(2, np.ceil(n_max / 2).astype(np.int)):
    tfs[2 * n::n] = False

  primes_list = np.where(tfs)[0]

  if n_min > 2:
    tfs_min = np.ones(n_min + 1, dtype=np.bool)
    tfs_min[:2] = False
    for n in range(2, np.ceil(n_min / 2).astype(np.int)):
      tfs_min[2 * n::n] = False

    primes_list_min = np.where(tfs_min)[0]

    primes_list = np.setdiff1d(primes_list, primes_list_min, assume_unique=True)

  return primes_list


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


def angled(x):
  '''
  *angled* gives the angle in degrees of a (set of) complex float(s).

  positional argument:
  --------------------
  x [ndarray of complex floats] is a single complex-valued float or a ndarray of complex-valued
                                floats for which the angle in degrees must be calculated

  *angled* returns the angles in degees for all values in the ndarray "x"

  Author: Joris Kampman, Thales NL, 2017
  '''

  return np.angle(x)*180/np.pi


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


def ind2sub(indices, size, order='c', origin='top'):
  '''
  *ind2sub* is a function which return the subscripted row and column indices for a global
  index given a specific size.specific

  positional arguments:
  ---------------------
  indices [ndarray of ints] is the argument holding a set of global indices from 0 to <number of
                            elements - 1> of the array.
  size [2-list or ndarray] is the defined size of the 2D array

  keyword arguments:
  ------------------
  order='c' [str] is the order in which the global index is defined. Two options are available:
                  'c' and 'f'. The default 'c' is the C-order in which the counter moves SIDEWAYS,
                  implying that in memory the elements of a row reside next to each other:
                  "row-major" order.

                  The FORTRAN-order indicated with 'f' is defined to run DOWNWARD the matrix. This
                  means that elements in the same column reside next to each other in memory:
                  "column-major" order.

  origin='top' [str] defines the starting point of the data. May be either 'top' or 'bottom'.
                     When 'top' is given, the TOP-LEFT point is the starting point.
                     When 'bottom' is given, the BOTTOM-LEFT point is the starting point

  *ind2sub* returns a 2-tuple, containing the row and column indices having the same shape as
  the input argument "indices", which may be an integer or a numpy array of integers.

  Author: Joris Kampman, Thales NL, 2017
  '''

  if np.alltrue(indices >= size[0] * size[1]):
    raise ValueError('At leas one of the indices is outside the maximum value')

  if order == 'c':
    row = np.floor(indices / size[1]).astype(int)
    col = indices - row * size[1]
    if origin == 'top':
      pass
    elif origin == 'bottom':
      row = size[0] - row - 1
    else:
      raise ValueError('The value for origin can only be either "top" (default), or "bottom"')

  elif order == 'f':
    col = np.floor(indices / size[0]).astype(int)
    row = indices - col * size[0]
  else:
    raise ValueError('The order "{0}" given is not valid. Only "c" and "f" are valid'.
                     format(order))

  return (row, col)


def sub2ind(rows, cols, size, order='c'):
  '''
  *sub2ind* is a function which returns the global index for a 2D matrix based on the given row
  and column indices

  positional arguments:
  ---------------------
  rows [ndarray of ints] is the argument holding a set of row indices from 0 to <number of
                         rows - 1> of the array.
  cols [ndarray of ints] is the argument holding a set of row indices from 0 to <number of
                         columns - 1> of the array.
  size [2-list or ndarray] is the defined size of the 2D array

  keyword arguments:
  ------------------
  order='c' [str] is the order in which the global index is defined. Two options are available:
                  'c' and 'f'. The default 'c' is the C-order in which the counter moves SIDEWAYS,
                  implying that in memory the elements of a row reside next to each other:
                  "row-major" order.

                  The FORTRAN-order indicated with 'f' is defined to run DOWNWARD the matrix. This
                  means that elements in the same column reside next to each other in memory:
                  "column-major" order.

  *sub2ind* returns a variable of the same shape as both positional arguments "rows" and "cols"
  containing the global indices.

  Author: Joris Kampman, Thales NL, 2017
  '''

  if order == 'c':
    indices = rows * size[1] + cols
  elif order == 'f':
    indices = cols * size[0] + rows
  else:
    raise ValueError('The order "{0}" given is not valid. Only "c" and "f" are.'.format(order))

  return indices


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


def save_animation(anim, filename, fps=30, metadata={'artist': 'Joris Kampman, Thales NL'},
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


def horizon(radar_altitude, sf_prop=4/3, elevation_angle_units='rad'):
  """
  Find the range and elevation of the horizon

  When ndarrays are used, note that the dimensions should allow broadcasting

  Arguments:
  ----------
  radar_altitude : float
            The altitude/height of the radar system above the earth
  sf_prop : float, default=4/3
            Scaling factor used to correct for propagation ray bending effects
  elevation_angle_units : ['rad' | 'deg']
                          The units in which the elevation must be returned:
                           - 'deg' is in degrees
                           - 'rad' is in radians

  Returns:
  --------
  range, elevation : 2-tuple
                     The range in meters and elevation in radians of the horizon

  """

  re = sf_prop*r_earth
  reh = re + radar_altitude

  # find horizon range
  rhz = np.sqrt(reh**2 - re**2)

  # find horizon elevation
  elev_hz = -np.arctan2(rhz, re)

  if elevation_angle_units == 'rad':
    pass
  elif elevation_angle_units == 'deg':
    elev_hz *= 180/np.pi
  else:
    raise ValueError('The *elevation_angle_units* value "{}" is not valid\n'.
                     format(elevation_angle_units), 'Valid options are "rad" (default) and "deg"')

  return rhz, elev_hz


def target_altitude(radar_altitude, target_range, target_elev, sf_prop=4/3):
  """
  Calculates the target altitude based on range and elevation

  When using ndarray values, note that the dimensions must allow broadcasting

  Arguments:
  ----------
  radar_altitude : (ndarray of) floats or ints
                   A set of radar altitudes in meters
  target_range : (ndarray of) floats or ints
                 A set of target ranges in meters
  target_elev : (ndarray of) floats or ints
                A set of target elevations in radians
  sf_prop : float, default=4/3
            A scale factor to correct for the propagation effects

  Returns:
  --------
  target_altitude : (ndarray of) floats
                    A set of target altitudes in meters

  See also:
  ---------
  target_range : calculates the range based on altitude and elevation
  target_elevation : calculates the elevation based on range and altitude
  horizon : calculates the location of the horizon
  """
  re = r_earth*sf_prop
  a = re + radar_altitude
  b = target_range

  c = np.sqrt(a**2 + b**2 - 2*a*b*np.cos(np.pi/2 + target_elev))

  return c - re


def target_range(radar_altitude, target_altitude, target_elevation, sf_prop=4/3):
  """
  Calculates the target range based on altitude and elevation

  When using ndarrays, note that the dimensions must allow broadcasting

  Arguments:
  ----------
  radar_altitude : (ndarray of) floats or ints
                   A set of radar altitudes in meters
  target_altitude : (ndarray of) floats or ints
                    A set of target local altitudes in meters
  target_elev : (ndarray of) floats or ints
                A set of target elevations in radians
  sf_prop : float, default=4/3
            A scale factor to correct for the propagation effects

  Returns:
  --------
  target_range : (ndarray of) floats
                 A set of target ranges in meters

  See also:
  ---------
  target_altitude : calculates the altitude based on altitude and elevation
  target_elevation : calculates the elevation based on range and altitude
  horizon : calculates the location of the horizon
  """
  re = r_earth*sf_prop

  x = re + radar_altitude
  z = re + target_altitude

  # abc rule
  y_candidates = np.array(abc(1., -2*x*np.cos(np.pi/2 + target_elevation), x**2 - z**2))

  # remove candidates with negative ranges (not possible)
  y_candidates[y_candidates < 0.] = np.nan

  return np.nanmin(y_candidates, axis=0)


def target_elevation(radar_altitude, target_altitude, target_range, sf_prop=4/3,
                     angle_units='rad'):
  """
  Calculates the target elevation based on altitude and range

  When using ndarrays, note that the dimensions must allow broadcasting

  Arguments:
  ----------
  radar_altitude : (ndarray of) floats or ints
                   A set of radar altitudes in meters
  target_altitude : (ndarray of) floats or ints
                    A set of target local altitudes in meters
  target_range : (ndarray of) floats or ints
                A set of target ranges in meters
  sf_prop : float, default=4/3
            A scale factor to correct for the propagation effects
  angle_units : ['deg' | 'rad'], default='rad'
                The units of the output elevation angle
                 - 'rad' is in radians
                 - 'deg' is in degrees

  Returns:
  --------
  target_elev : (ndarray of) floats
                A set of target elevations in radians ("rad") or degrees ("deg")

  See also:
  ---------
  target_altitude : calculates the altitude based on altitude and elevation
  target_elevation : calculates the elevation based on range and altitude
  horizon : calculates the location of the horizon
  """

  re = r_earth*sf_prop

  a = re + radar_altitude
  b = re + target_altitude
  c = target_range

  target_elev = -(np.arccos((b**2 - (a**2 + c**2))/(2*a*c)) - np.pi/2)

  if angle_units == 'rad':
    pass
  elif angle_units == 'deg':
    target_elev *= 180/np.pi
  else:
    raise ValueError('the *angle_units* keyword value "{}" is not valid.'.format(angle_units),
                     'Only "rad" and "deg" are.')

  return target_elev


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


def doppler_filter_bank(nof_taps, nof_filters=None, taper_desig='050DCxx', scalingpars=None):
  """
  Creates a standard equidistant doppler filter bank with a tapering

  Arguments:
  ----------
  nof_taps : int
             The number of points in the filter
  nof_filters : None or int, default: None
                If not None, this are the number of filters in the filter bank. Default will result
                in a square filter bank (nof_filters == nof_taps)
  taper_desig : str
                A string designating the weighting to use. Refer to *jktools.beamforming.\
                create_beam* for the syntax options
  scalingpars : dict or None
                Has the keywords necessary for the function *scale_filter_coefs*.

  Returns:
  --------
  dfbank : (<nof_filters>, <nof_taps>) ndarray of complex floats
           The coefficients  for all filters in the doppler filter bank
  """
  if nof_filters is None:
    nof_filters = nof_taps

  coefs0 = create_taper(np.r_[:nof_taps], taperdesig=taper_desig)

  dfbank = np.zeros((nof_filters, nof_taps), dtype=np.complex)
  tap_ratios = np.r_[:nof_taps]
  for ifilt in range(nof_filters):
    inc = ifilt*nof_taps/nof_filters
    E = coefs0*np.exp(-1j*2*np.pi*inc*tap_ratios/nof_taps)
    dfbank[ifilt, :] = E

  if scalingpars is not None:
    dfbank = scale_filter_coefs(dfbank, **scalingpars)

  return dfbank


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


def angle2nf(ang, dpos, lambda_, angle_units='rad', cosang=0.):
  """
  Convert and angle to the corresponding normalized frequency

  Note:
  -----
  The nf calculated is along the `u` coordinate in sine space. Since `u = sin(azi)*cos(elev)`, the
  elevation might be given via the keyword argument *cosang*. The elevation is simply `v=sin(elev)`
  which is equal to `u` with *cosang=0.* which is the default value

  Positional Arguments:
  ---------------------
  ang : float or ndarray of floats
        the angle
  dpos : float
         the spacing between elements in meters
  lambda_ : float
            The wavelength in meters

  Keyword arguments:
  ------------------
  angle_units : [ 'rad' | 'mrad' | 'deg'], default='rad'
                The angle units. 'rad' is radians
                                 'mrad' is milliradians
                                 'deg' is degrees
  cosang : float or ndarray of floats
           The cosine angle used in conversion to sine-space. In case a ndarray is given it should
           have the same size as *ang*

  Returns:
  --------
  nf : ndarray of floats or float
       The normalized frequencies corresponding to the angles provided

  See also:
  ---------
  nf2angle : converts an angle to the normalized frequency on the `u` axis
  """

  uv = angle2uv(ang, cosang=cosang, angle_units=angle_units)
  nf = uv2nf(uv, dpos, lambda_)

  return nf


def nf2angle(nf, dpos, lambda_, angle_units='rad', cosang=0.):
  """
  convert normalized frequencies to angle.

  Note:
  -----
  The angle calculated is the `u` coordinate in sine space. Since `u = sin(azi)*cos(elev)`, the
  elevation might be given via the keyword argument *cosang*. The elevation is simply `v=sin(elev)`
  which is equal to `u` with *cosang=0.* which is the default value

  Positional Arguments:
  ---------------------
  nf : float or ndarray of floats
       the normalized frequencies or frequency
  dpos : float
         the spacing between elements in meters
  lambda_ : float
            The wavelength in meters

  Keyword arguments:
  ------------------
  angle_units : [ 'rad' | 'mrad' | 'deg'], default='rad'
                The angle units. 'rad' is radians
                                 'mrad' is milliradians
                                 'deg' is degrees
  cosang : float or ndarray of floats
           The cosine angle used in conversion to sine-space. In case a ndarray is given it should
           have the same size as *nf*

  Returns:
  --------
  ang : ndarray of floats or float
        The angle corresponding to the input normalized frequencies

  See also:
  ---------
  angle2nf : converts an angle to the normalized frequency on the `u` axis
  uv2angle : converts uv coordinates to angles in radians
  """

  uv = nf2uv(nf, dpos, lambda_)
  angle = uv2angle(uv, angle_units=angle_units, cosang=cosang)

  return angle


def angle2uv(ang, cosang=0., angle_units='rad'):
  """
  Convert an angle to the sine-space representation
  """

  angrad = None
  if angle_units == 'rad':
    angrad = ang
  elif angle_units == 'mrad':
    angrad = ang*1e-3
  elif angle_units == 'deg':
    angrad = ang*np.pi/180

  uv = np.sin(angrad)*np.cos(cosang)

  return uv


def uv2angle(uv, angle_units='rad', cosang=0.):
  """
  Convert sine-space angles to AE angles

  :param uv: sine space coordinate (2-tuple)
  :param angle_units: ['rad', 'deg']
  :param cosang: cosine angle (float)
  :return: angle coordinates (2-tuple)
  """
  angrad = np.arcsin(uv/np.cos(cosang))
  ang = None
  if angle_units == 'rad':
    ang = angrad
  elif angle_units == 'mrad':
    ang = angrad*1e3
  elif angle_units == 'deg':
    ang = angrad*180/np.pi

  return ang


def uv2nf(uv, dpos, lambda_):
  """
  convert uv coordinate to normalized frequency
  """

  nf = (dpos/lambda_)*uv

  return nf


def nf2uv(nf, dpos, lambda_):
  """
  convert normalized frequency to uv coordinate
  """

  uv = (lambda_/dpos)*nf

  return uv


def qplot(*args, **kwargs):
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

