'''
Module which contains the user defined colormaps

Contains:
---------
show_all_colormaps : print all colormaps in a figure
jetmod : colormap which contains no yellow
jetext : extension to jet which starts at black and ends at white
binary : red (False) and green (True) colormap without intermediate colors
traffic_light : red, yellow, green colormap

# private functions
_gen_cmap_output : support function which creates an actual Colormap object.

See Also:
---------
matplotlib.colors : a submodule containing stuff on colors and colormaps
matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a 3x1 matrix of values
'''

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from numpy import ma

__all__ = ['PercNorm', 'MaxContrastNorm', 'show_all_colormaps', 'jetmod', 'jetext',
           'traffic_light', 'binary', 'bgr', 'jetgray']


# generate output based on inputs.. colormaps are defined per marker_array
class PercNorm(Normalize):
  '''
  Normalize a given value to the 0-1 range depending on a percentile bracket

  That is, the vmin and vmax values are calculated based on the percentiles of the data content

  *PercNorm* is a child of *matplotlib.colors.Normalize*, and is a small wrapper around this to
  calculate the vmin and vmax based on the percentiles of the data content.
  '''

  def __init__(self, pmin=0, pmax=100, clip=False):
    '''
    Keyword Arguments:
    ------------------
    pmin : (0) int or float
           The bottom of the percentile brackets in percentages (from 0 to 100)
    pmax : (100) int or float
           The top of the percentile bracket in percentages (from 0 to 100)
    clip : (False) bool
           is passed to *matplotlib.colors.Normalize*. Check this for more information

    '''
    self.pmin = pmin
    self.pmax = pmax

    # call init in superclass
    super().__init__(vmin=None, vmax=None, clip=clip)

  def __call__(self, value, clip=None):
    '''
    A call to calculate the colormap based on data/value

    Argument:
    ---------
    value : ndarray of scalars, or scalar
            the data of which the image is made up. Generally for colormaps this is a 2D set
    clip : See *matplotlib.colors.Normalize* for more information

    Returns:
    --------
    result : see *matplotlib.colors.Normalize* for more information
    '''

    result, is_scalar = self.process_value(value)

    # set vmin, vmax based on pmin and pmax
    self.vmin = np.nanpercentile(result.ravel(), self.pmin)
    self.vmax = np.nanpercentile(result.ravel(), self.pmax)

    return super().__call__(value, clip=clip)


class MaxContrastNorm(Normalize):
  '''
  normalize a given value to the 0-1 range depending on data densities
  '''

  # matched normalization
  def __call__(self, value, clip=None):
    """
    Normalize *value* data in the ``[vmin, vmax]`` interval into
    the ``[0.0, 1.0]`` interval and return it.  *clip* defaults
    to *self.clip* (which defaults to *False*).  If not already
    initialized, *vmin* and *vmax* are initialized using
    *autoscale_None(value)*.
    """
    if clip is None:
      clip = self.clip

    result, is_scalar = self.process_value(value)

    # set vmin, vmax in case they are still None
    self.autoscale_None(result)
    vmin, vmax = self.vmin, self.vmax
    if vmin == vmax:
      result.fill(0)   # Or should it be all masked?  Or 0.5?
    elif vmin > vmax:
      raise ValueError("minvalue must be less than or equal to maxvalue")
    else:
      vmin = float(vmin)
      vmax = float(vmax)
      if clip:
          mask = ma.getmask(result)
          result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                            mask=mask)
      # ma division is very slow; we can take a shortcut
      # use np.asarray so data passed in as an ndarray subclass are
      # interpreted as an ndarray. See issue #6622.
      resdat = np.asarray(result.data)

      # give sorting index
      isort = np.unique(resdat.ravel(), return_inverse=True)[1]
      print(isort)
      print(isort.min(), isort.max())
      resdat = isort.reshape(*resdat.shape).astype(float)
      resdat -= resdat.min()
      resdat /= resdat.max()
      print(resdat.dtype)
      result = np.ma.array(resdat.astype(float), mask=result.mask, copy=False)
      print(resdat.dtype)
    if is_scalar:
      result = result[0]

    print(np.unique(result))
    print(result.dtype)
    return result


def show_all_colormaps():
  '''
  Shows all colormaps in an - unblocked - figure

  Author:
  -------
  Joris Kampman, Thales NL, 2017
  '''

  import matplotlib.pyplot as plt

  nof_steps = 256
  cmaps_to_show = ['jetmod', 'jetgray', 'jetext', 'traffic_light', 'binary', 'bgr']
  # cmaps_to_show = ['jetmod']

  mat = np.linspace(0, 1, nof_steps).reshape((1, -1))
  mat = np.vstack((mat, mat))

  fig, axs = plt.subplots(len(cmaps_to_show), 4, num='All defined subplots',
                          subplot_kw=dict(xticks=[], yticks=[]))

  for iax, cmap_str in enumerate(cmaps_to_show):
    # axs.append(plt.subplot(gs[iax, 0]))
    cmap = eval('{:s}(nof_steps=nof_steps, interpolation="linear")'.format(cmap_str))
    cmapnnb = eval('{:s}(nof_steps=nof_steps, interpolation="nearest")'.format(cmap_str))
    cmapi = eval('{:s}(nof_steps=nof_steps, interpolation="linear", invert=True)'.format(cmap_str))
    cmapn = eval('{:s}(nof_steps=nof_steps, invert=False, negative=True, interpolation="linear")'
                 .format(cmap_str))

    axs[iax, 0].imshow(mat, cmap=cmap, aspect='auto')
    axs[iax, 0].set_title(cmap_str)
    axs[iax, 1].imshow(mat, cmap=cmapi, aspect='auto')
    axs[iax, 1].set_title(cmap_str + '(inverted)')
    axs[iax, 2].imshow(mat, cmap=cmapn, aspect='auto')
    axs[iax, 2].set_title(cmap_str + '(negative)')
    axs[iax, 3].imshow(mat, cmap=cmapnnb, aspect='auto')
    axs[iax, 3].set_title(cmap_str + '(markers only)')

  plt.show(block=False)

  return None


def _gen_cmap_output(marker_array, nof_steps, istep, invert=False, negative=False,
                     interpolation='linear', name=None):
  '''
  Creates an actual colormap object via matplotlib.colors.LinearSegmentedColormap

  Arguments:
  ----------
  marker_array : ndarray of float 3-tuples
                 contains the rgb tuples of the markers
  nof_steps : int
              Number of steps in the colormap
  istep : {int, None, 'vector'}
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors
  name : str or None [optional]
         The name of the colormap to create

  Returns:
  --------
  In case istep=None, the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep" is an integer the color at step "istep" is returned as an 3-tuple of rgb values

  In case istep='vector', the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''

  # check if interpolation is nearest, then create double markers to prevent interpolation
  if interpolation == 'nearest':
    nof_markers = marker_array.shape[0]
    marker_array_new = np.zeros(((nof_markers - 1)*3 + 1, 4), dtype=np.float)
    marker_array_new[[0, -1], :] = marker_array[[0, -1], :]

    for imarker in range(1, nof_markers):
      inew = imarker*3 + np.array([-2, -1, 0], dtype=np.int)
      rel_pos = (marker_array[imarker, 0] + marker_array[imarker-1, 0])/2
      marker_array_new[inew[0], 0] = rel_pos - 0.0001
      marker_array_new[inew[1], 0] = rel_pos + 0.0001
      marker_array_new[inew[0], 1:] = marker_array[imarker-1, 1:]
      marker_array_new[inew[1], 1:] = marker_array[imarker, 1:]
      marker_array_new[inew[2], :] = marker_array[imarker, :]

    marker_array = marker_array_new

  # if linear -> no action
  elif interpolation == 'linear':
    pass

  # else: unknown value for keyword "interpolation"
  else:
    raise ValueError('[keyword error] "interpolation=\'{}\'" is not a valid value'
                     .format(interpolation))

  # invert colormap order (e.g., b -> w becomes w -> b)
  if invert:
    # flip up/down
    marker_array = np.flipud(marker_array)
    # change first column
    marker_array[:, 0] = 1 - marker_array[:, 0]

  # take the negative of the colors (e.g., [1, 1, 0] becomes [0, 0, 1])
  if negative:
    marker_array[:, 1:] = 1.0 - marker_array[:, 1:]

  # =================================================
  # switch based on what keyword "istep" is ...

  # istep = None -> return colormap object
  if istep is None:  # create in colormap format
    cdict = dict(red=[], green=[], blue=[])
    for row in marker_array:
      cdict['red'].append((row[0], row[1], row[1]))
      cdict['green'].append((row[0], row[2], row[2]))
      cdict['blue'].append((row[0], row[3], row[3]))

    return LinearSegmentedColormap(name, cdict, nof_steps)

  # istep = 'vector' -> create Nx3 matrix with RGB 3-tuples
  elif istep == 'vector':
    return_vector = np.zeros((nof_steps, 3), dtype=np.float)
    for istep in range(nof_steps):
      posi = np.linspace(0, 1, nof_steps)[istep]
      redi = np.interp(posi, marker_array[:, 0], marker_array[:, 1])
      greeni = np.interp(posi, marker_array[:, 0], marker_array[:, 2])
      bluei = np.interp(posi, marker_array[:, 0], marker_array[:, 3])

      return_vector[istep, :] = [redi, greeni, bluei]

    return return_vector

  # istep is an integer -> give single RGB 3-tuple
  elif type(istep) == int:
    # do interpolation and pick index
    posi = np.linspace(0, 1, nof_steps)[istep]
    redi = np.interp(posi, marker_array[:, 0], marker_array[:, 1])
    greeni = np.interp(posi, marker_array[:, 0], marker_array[:, 2])
    bluei = np.interp(posi, marker_array[:, 0], marker_array[:, 3])

    return (redi, greeni, bluei)

  # else: exception
  else:
    raise ValueError('[keyword error] The value "istep"=\'{}\' is not valid'.format(istep))


def jetmod(nof_steps=256, istep=None, bright=False, invert=False, negative=False,
           interpolation='linear'):
  '''
  Modified 'jet' colormap. Yellow is removed for types like Geert Onstenk ...

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  bright : bool [optional]
           Whether to have bright varieties of red and blue at the endings or the default dark ones
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''

  marker_array = np.array([[0, 0, 0, 143],
                           [32, 0, 0, 255],
                           [76, 0, 255, 255],
                           [128, 0, 255, 0],
                           [159, 255, 153, 0],
                           [223, 255, 0, 0],
                           [255, 128, 0, 0]], dtype='float')/255

  # if bright then ditch darker edge markers (reduce list by 1 on both sides)
  if bright:
    marker_array = marker_array[slice(1, -1), :]
    # stretch positions
    positions = marker_array[:, 0]
    positions -= positions[0]
    positions /= positions[-1]
    marker_array[:, 0] = positions

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='jetmod')


def jetgray(nof_steps=256, istep=None, bright=False, invert=False, negative=False,
             interpolation='linear'):
  '''
  Modified 'jet' colormap. Yellow is removed for types like Geert Onstenk ...

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  bright : bool [optional]
           Whether to have bright varieties of red and blue at the endings or the default dark ones
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''
  marker_array = np.array([[0, 0, 0, 127.5],  # dark blue
                           [32, 0, 0, 255],  # blue
                           [64, 0, 255, 255],  # cyan
                           [112, 0, 255, 0],  # green
                           [128, 200, 200, 200],
                           [140, 255, 128, 40],  # orange
                           [192, 255, 0, 0],  # red
                           [255, 128, 0, 0]], dtype='float')/255  # dark red

  # if bright then ditch darker edge markers (reduce list by 1 on both sides)
  if bright:
    marker_array = marker_array[slice(1, -1), :]
    # stretch positions
    positions = marker_array[:, 0]
    positions -= positions[0]
    positions /= positions[-1]
    marker_array[:, 0] = positions

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='jetmod')


def jetext(nof_steps=256, istep=None, invert=False, negative=False, interpolation='linear'):
  '''
  Modified 'jet' colormap which extends the lower values to black, and the upper values to white

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''

  marker_array = np.array([[0, 0, 0, 0],
                           [79, 0, 0, 255],
                           [142, 0, 255, 0],
                           [173, 255, 255, 0],
                           [236, 255, 0, 0],
                           [255, 255, 255, 255]], dtype='float')/255

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='jetext')


def traffic_light(nof_steps=256, istep=None, invert=False, negative=False, interpolation='linear'):
  '''
  colormap consisting of colors with markers red, yellow and green (like a traffic light)

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''

  marker_array = np.array([[0.0, 1, 0, 0],
                           [0.5, 1, 1, 0],
                           [1.0, 0, 1, 0]], dtype=np.float)

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='traffic_light')


def binary(nof_steps=256, istep=None, invert=False, negative=False, interpolation='linear'):
  '''
  Colormap for displaying true or false images. consists of two colors and no interpolation done

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''
  marker_array = np.array([[0.0, 1, 0, 0],
                           [1.0, 0, 1, 0]], dtype=np.float)

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='binary')


def bgr(nof_steps=256, istep=None, invert=False, negative=False, interpolation='linear'):
  '''
  Colormap for displaying blue/green/red. consists of 3 colors

  Arguments:
  ----------
  nof_steps : int [optional]
              The number of colors in the colormap
  istep : {int, None, 'vector'} [optional]
          Main switch to indicate what to return:
           - an integer, results in a single RGB 3-tuple to be returned
           - None, results in the creation of a colormap object which is returned
           - 'vector', results in the return of a Nx3 ndarray of all RGB colors in the colormap
  invert : bool [optional]
           Whether to invert the colormap
  negative : bool [optional]
             Whether to take the negative of the specified colors
  interpolation : {'linear', 'nearest'}
                  'linear' implies linear interpolation between markers
                  'nearest' implies no interpolation and only the marker colors are valid colors

  Returns:
  --------
  In case "istep=None", the function returns a colormapobject via the function matplotlib.colors.
  LinearSegmentedColormap function.

  In case "istep=<int>", the color at step "istep" is returned as an 3-tuple of rgb values

  In case "istep='vector'", the 3xN ndarray with rgb values is returned, giving the full set of
  colors, with N is "nof_steps"

  See Also:
  ---------
  matplotlib.colors : submodule containing information on colors and colormaps
  matplotlib.colors.LinearSegmentedColormap : creates a colormap object from a RGB input
  '''
  marker_array = np.array([[0, 0, 0, 1],
                           [0.4, 0, 0, 1],
                           [0.5, 0, 1, 0],
                           [0.6, 1, 0, 0],
                           [1, 1, 0, 0]], dtype=np.float)

  return _gen_cmap_output(marker_array, nof_steps, istep, invert, negative, interpolation,
                          name='bgr')
