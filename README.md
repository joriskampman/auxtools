# README for _auxtools_ python module

## 1. general information

This *auxtools* module contains a large set of smaller and medium-sized python functions which can come in handy when working in python. They are _auxiliary_ tools, hence the name *auxtools*.

This module can be expanded by anyone with smal, tools, wrappers or functions at will. However, try not to clutter to module and keep an eye out for useless or superfluous function that can be removed at some point.

These auxtools module is stored in [github](https://github.com/joriskampman/auxtools.git) and are a personal repository, however to be used without any type of warranty by anyone who feels like it.

## 2. contents

The repository contains many functions which all should contain the docstring allowing the user to find out the syntax and purpose of all.

### 2.1 Main module

The is all contained in the `__init__.py` file. 

The full list of basis functions is shown below. all _helper_ functions are not included. These
generally start with an underscore (_):
- `nanplot` creates a plot for which the y-values are NaN's
- `split_complex` splits a complex value in a real and imaginary part
- `monospace` create monospaced grid from an almost-monospaced grid
- `get_closest_index` the the index of the wanted value in an array. or the closest
- `substr2index` find the indices of a substring
- `listify` turn the input in a list
- `tuplify` turn the input in a tuple
- `arrayify` turn the input in an array
- `color_vector` make a color vector based on a single color (from dark to light)
- `strip_all_spaces` strip all spaces from the strings in an array-like
- `goto` thales function -> goto a certain project and set all paths
- `select_file` wrapper to select a file
- `select_savefile` wrapper to select a savefile
- `select_folder` wrapper to select a directory/folder (sorry for the name!!)
- `val2ind` the index where a value would end up if inserted
- `calc_frequencies` calculate an array of frequencies for a spectrum
- `spectrum` calculate and plot a spectrum
- `find_dominant_frequencie_`find the dominant frequencies in a signal
- `print_struct_arra_`function that can display a structured array in readable form
- `subset_str` print a string containing only a subset of an array (prevent to long printouts)
- `extent` creates and extent list of 4 elements from xlims and  ylims
- `normalize` normalize a set of data
- `inputdlg` dialog window to give inputs in with defaults
- `rms` calculate the root-mean-square
- `phase2snr` calculates the phase error from a signal-to-noise value
- `snr2phase` calculates the signal-to-noise from a phase error
- `ndprint` print a ndarray in readable form
- `dinput` ask for an input that has a default value
- `round_to_values` round an input to a value in a list
- `figname` creates a figure name, takes action if already exists
- `db` calculate decibel for input
- `lin` calculate linear value from db
- `logmod` calculate log modulus from linear input
- `bracket` return the data bracket
- `datarange` returns the range of the data values
- `filter_gains` gives ghe gains (noise and signal) of a filter
- `subplot_layout` calculates the optimum layout for a number of subplots
- `save_animation` saves an animation with all default properties
- `tighten` tighten the subplots in a figures
- `resize_figure` resize a figure within constraints (e.g., A3 format)
- `paper_A_dimensions` get the size of A<x> paper dimensions
- `abc` quadratic equation solver via the ABC-algorithm
- `f1_score` calculates the F1 metric (for neural nets)
- `scale_filter_coefs` scales filter coefficients
- `power2amp` convert the power to the amplitude values
- `power2rms` the power to root-mean-squared value
- `exp_fast` calculates the exponent fast (via numexpr module)
- `qplot` quick plot
- `subtr_angles` subtract angles the correct way
- `add_angles` add angles the correct way
- `mean_angle` calculate the mean of angles the correct way
- `str2int` string to integer (might be useless)
- `str2timedelta` string that indicates a time delta
- `ind2rgba` convert indexed image to RGBA image
- `short_short` creates a short string from a long string. various options
- `find_elm_containing_substrs` find elements in a list of string that contain a substring
- `data_scaling` scale data according to some minimum and maximum value
- `modify_strings` modify strings in a list either globally or specifically
- `improvedshow` extension to _matplotlib.pyplot.imshow_ that can add text and stuff
- `get_file` don't know what this adds over _select_file_
- `wrap_string` wrap a long string across multiple lines, some intelligence to it
- `eval_string_of_indices` from a string that contains indices, create the list
- `select_from_list` select items from a list of items
- `markerline` create/print a line with markers and text to use as header

### 2.2 submodules
The module _auxtools_ contains 2 submodules:
- cmaps : contains functions used to make and define various colormaps
- coordinate_transforms : contains functions that convert coordinate systems and more

Both submodules are imported plainly in auxtools, however the breakdown is in parts below

#### 2.2.1 _coordinate_transforms_ submodule

The full list of functions, plus a oneline description is given below for the submodule
_coordinate_transforms_:

__generic__ functions
- `ctform_mat` gives the rotation matrix for a sequence of rotations around the principal axes
- `ctform` coordinate transformation for a set of points and a sequence of rotations
- `rotx` rotation around the x-axis
- `roty` rotation around the y-axis
- `rotz` rotation around the z-axis
- `rot3` rotation around xyz axes

__thales specific__ functions (see _EAR coordinate transformations_ document)
- `csinfo` provide information of specific (thales) defined coordinate systems
- `enu2p` ENU to PCS coordinate transformation
- `p2enu` PCS to ENU coordinate transformation
- `p2ar` PCS to ARCS coordinate transformation
- `p2am` PCS to AMCS coordinate transformation
- `am2p` AMCS to PCS coordinate transformation
- `ar2p` ARCS to PCS coordinate transformation
- `ar2am` ARCS to AMCS coordinate transformation
- `am2ar` AMCS to ARCS coordinate transformation
- `am2aa` AMCS to AACS coordinate transformation
- `aa2am` AACS to AMCS coordinate transformation
- `aa2af` AACS to AFCS coordinate transformation
- `aa2af` AACS to AFCS coordinate transformation
- `af2aa` AFCS to AACS coordinate transformation
- `af2as` AFCS to ASCS coordinate transformation
- `aa2as` AACS to ASCS coordinate transformation

__earth fixed__ coordinate systems and transforms:
- `lla2ecef` GPS coordinates to earth-centered, earth-fixed coordinate transformation
- `ecef2lla` Earth-Centered, Earth-fixed to GPS coordinate transformation
- `enu2ecef` East-North-Up to Earth-Centered Earth-Fixed coordinate transformation
- `ecef2enu` Earth-Centered Earth-Fixed to East-North-Up coordinate transformation
- `enu2lla` East-North-Up to GPS coordinate transformation
- `lla2enu` GPS to East-North-Up coordinate transformation
- `cart2sph` cartesian to spherical coordinate transformation
- `sph2cart` spherical to cartesian coordinate transformation

some wrappers for people who use __NHCS__ instead of __ENU__
- `nhcs_is_enu_warning` warning that is issued when NHCS functions are used instead of ENU
- `nh2p` equal to _enu2p_ with an printed warning that ENU is preferred over NHCS
- `p2nh` equal to _p2enu_ with an printed warning that ENU is preferred over NHCS
- `nh2ecef` equal to _enu2ecef_ with an printed warning that ENU is preferred over NHCS
- `ecef2nh` equal to _ecef2enu_ with an printed warning that ENU is preferred over NHCS
- `nh2lla` equal to _enu2lla_ with an printed warning that ENU is preferred over NHCS
- `lla2nh` equal to _lla2enu_ with an printed warning that ENU is preferred over NHCS

#### 2.2.2 _cmaps_ submodule

The full set of callable user functions is given below. Note that the functions that are not
meant to be called are omitted here:

- `show_all_colormaps` creates a plot showing all defined colormaps
- `jetmod` colormap that takes _jet_ as a basis, but has no yellow
- `jetext` colormap from black to white with _jet_ in the middle
- `traffic_light` red, yellow and green colormap
- `binary` red and green discrete colormap (usefull for true/false matrices for instance)

Classes for __normalization__ for colormaps:
- `PercNorm` normalize depending on a percentile bracket
- `MaxContrastNorm` normalize depending on data density (beta! lot of printouts!)
