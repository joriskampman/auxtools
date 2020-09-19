# README for _auxtools_ python module

## 1. general information

This _auxtools_ module contains a large set of smaller and medium-sized python functions which can
come in handy when working in python. They are _auxiliary_ tools, hence the name _auxtools_.

This module can be expanded by anyone with smal, tools, wrappers or functions at will. However, try
not to clutter to module and keep an eye out for useless or superfluous function that can be
removed at some point.

These auxtools module is stored in [github](https://github.com/joriskampman/auxtools.git) and are
a personal repository, however to be used without any type of warranty by anyone who feels like it.

## 2. contents

The repository contains many functions which all should contain the docstring allowing the user
to find out the syntax and purpose of all.

### 2.1 Main module

The is all contained in the *\__init__.py* file. 

The full list of basis functions is shown below. all _helper_ functions are not included. These
generally start with an underscore (_):

- _nanplot_ : creates a plot for which the y-values are NaN's
- _split\_complex_ : splits a complex value in a real and imaginary part
- _monospace_ : create monospaced grid from an almost-monospaced grid
- _get\_closest\_index_ : the the index of the wanted value in an array. or the closest
- _substr2index_ : find the indices of a substring
- _listify_ : turn the input in a list
- _tuplify_ : turn the input in a tuple
- _arrayify_ : turn the input in an array
- _color\_vector_ : make a color vector based on a single color (from dark to light)
- _strip\_all\_spaces_ : strip all spaces from the strings in an array-like
- _goto_ : thales function -> goto a certain project and set all paths
- _select\_file_ : wrapper to select a file
- _select\_savefile_ : wrapper to select a savefile
- _select\_folder_ : wrapper to select a directory/folder (sorry for the name!!)
- _val2ind_ : the index where a value would end up if inserted
- _calc\_frequencies_ : calculate an array of frequencies for a spectrum
- _spectrum_ : calculate and plot a spectrum
- _find\_dominant_frequencie_s: find the dominant frequencies in a signal
- _print\_struct_arra_y: function that can display a structured array in readable form
- _subset\_str_ : print a string containing only a subset of an array (prevent to long printouts)
- _extent_ : creates and extent list of 4 elements from xlims and  ylims
- _normalize_ : normalize a set of data
- _inputdlg_ : dialog window to give inputs in with defaults
- _rms_ : calculate the root-mean-square
- _phase2snr_ : calculates the phase error from a signal-to-noise value
- _snr2phase_ : calculates the signal-to-noise from a phase error
- _ndprint_ : print a ndarray in readable form
- _dinput_ : ask for an input that has a default value
- _round\_to\_values_ : round an input to a value in a list
- _figname_ : creates a figure name, takes action if already exists
- _db_ : calculate decibel for input
- _lin_ : calculate linear value from db
- _logmod_ : calculate log modulus from linear input
- _bracket_ : return the data bracket
- _datarange_ : returns the range of the data values
- _filter\_gains_ : gives ghe gains (noise and signal) of a filter
- _subplot\_layout_ : calculates the optimum layout for a number of subplots
- _save\_animation_ : saves an animation with all default properties
- _tighten_ : tighten the subplots in a figures
- _resize\_figure_ : resize a figure within constraints (e.g., A3 format)
- _paper\_A\_dimensions_ : get the size of A<x> paper dimensions
- _abc_ : quadratic equation solver via the ABC-algorithm
- _f1\_score_ : calculates the F1 metric (for neural nets)
- _scale\_filter\_coefs_ : scales filter coefficients
- _power2amp_ : convert the power to the amplitude values
- _power2rms_ : the power to root-mean-squared value
- _exp\_fast_ : calculates the exponent fast (via numexpr module)
- _qplot_ : quick plot
- _subtr\_angles_ : subtract angles the correct way
- _add\_angles_ : add angles the correct way
- _mean\_angle_ : calculate the mean of angles the correct way
- _str2int_ : string to integer (might be useless)
- _str2timedelta_ : string that indicates a time delta
- _ind2rgba_ : convert indexed image to RGBA image
- _short\_short_ : creates a short string from a long string. various options
- _find\_elm\_containing\_substrs_ : find elements in a list of string that contain a substring
- _data\_scaling_ : scale data according to some minimum and maximum value
- _modify\_strings_ : modify strings in a list either globally or specifically
- _improvedshow_ : extension to _matplotlib.pyplot.imshow_ that can add text and stuff
- _get\_file_ : don't know what this adds over _select\_file_
- _wrap\_string_ : wrap a long string across multiple lines, some intelligence to it
- _eval\_string\_of\_indices_ : from a string that contains indices, create the list
- _select\_from\_list_ : select items from a list of items
- _markerline_ : create/print a line with markers and text to use as header

### 2.2 submodules
The module _auxtools_ contains 2 submodules:
- cmaps : contains functions used to make and define various colormaps
- coordinate_transforms : contains functions that convert coordinate systems and more

Both submodules are imported plainly in auxtools, however the breakdown is in parts below

#### 2.2.1 _coordinate\_transforms_ submodule

The full list of functions, plus a oneline description is given below for the submodule
_coordinate_transforms_:

__generic__ functions
- _ctform\_mat_ : gives the rotation matrix for a sequence of rotations around the principal axes
- _ctform_ : coordinate transformation for a set of points and a sequence of rotations
- _rotx_ : rotation around the x-axis
- _roty_ : rotation around the y-axis
- _rotz_ : rotation around the z-axis
- _rot3_ : rotation around xyz axes

__thales specific__ functions (see _EAR coordinate transformations_ document)
- _csinfo_ : provide information of specific (thales) defined coordinate systems
- _enu2p_ : ENU to PCS coordinate transformation
- _p2enu_ : PCS to ENU coordinate transformation
- _p2ar_ : PCS to ARCS coordinate transformation
- _p2am_ : PCS to AMCS coordinate transformation
- _am2p_ : AMCS to PCS coordinate transformation
- _ar2p_ : ARCS to PCS coordinate transformation
- _ar2am_ : ARCS to AMCS coordinate transformation
- _am2ar_ : AMCS to ARCS coordinate transformation
- _am2aa_ : AMCS to AACS coordinate transformation
- _aa2am_ : AACS to AMCS coordinate transformation
- _aa2af_ : AACS to AFCS coordinate transformation
- _aa2af_ : AACS to AFCS coordinate transformation
- _af2aa_ : AFCS to AACS coordinate transformation
- _af2as_ : AFCS to ASCS coordinate transformation
- _aa2as_ : AACS to ASCS coordinate transformation

__earth fixed__ coordinate systems and transforms:
- _lla2ecef_ : GPS coordinates to earth-centered, earth-fixed coordinate transformation
- _ecef2lla_ : Earth-Centered, Earth-fixed to GPS coordinate transformation
- _enu2ecef_ : East-North-Up to Earth-Centered Earth-Fixed coordinate transformation
- _ecef2enu_ : Earth-Centered Earth-Fixed to East-North-Up coordinate transformation
- _enu2lla_ : East-North-Up to GPS coordinate transformation
- _lla2enu_ : GPS to East-North-Up coordinate transformation
- _cart2sph_ : cartesian to spherical coordinate transformation
- _sph2cart_ : spherical to cartesian coordinate transformation

some wrappers for people who use __NHCS__ instead of __ENU__
- _nhcs\_is\_enu\_warning_ : warning that is issued when NHCS functions are used instead of ENU
- _nh2p_ : equal to _enu2p_ with an printed warning that ENU is preferred over NHCS
- _p2nh_ : equal to _p2enu_ with an printed warning that ENU is preferred over NHCS
- _nh2ecef_ : equal to _enu2ecef_ with an printed warning that ENU is preferred over NHCS
- _ecef2nh_ : equal to _ecef2enu_ with an printed warning that ENU is preferred over NHCS
- _nh2lla_ : equal to _enu2lla_ with an printed warning that ENU is preferred over NHCS
- _lla2nh_ : equal to _lla2enu_ with an printed warning that ENU is preferred over NHCS

#### 2.2.2 _cmaps_ submodule

The full set of callable user functions is given below. Note that the functions that are not
meant to be called are omitted here:

- _show\_all\_colormaps_ : creates a plot showing all defined colormaps
- _jetmod_ : colormap that takes _jet_ as a basis, but has no yellow
- _jetext_ : colormap from black to white with _jet_ in the middle
- _traffic\_light_ : red, yellow and green colormap
- _binary_ : red and green discrete colormap (usefull for true/false matrices for instance)

Classes for __normalization__ for colormaps:
- _PercNorm_ : normalize depending on a percentile bracket
- _MaxContrastNorm_ : normalize depending on data density (beta! lot of printouts!)
