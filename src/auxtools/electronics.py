"""
electro toolbox of ools.

Anything related to electronics

author: E.J.J. (Joris) Kampman, fall 2026
"""
from . import colorvec, format_as_si, qplot, logmod
import matplotlib.pyplot as plt
import numpy as np


# ----------------- FUNCTIONS --------------------------------------------------------
def calc_group_impedances(group, makeplot=False, ax=None):
  """ check if this is a set to process, or redo if it is another branch

  arguments:
  ----------
  group : indication on how the system is grouped
          np.ndarray -> this is a single element in series
          list -> series of elements (maximum impedance counts)
          tuple -> parallel elements (minimum counts)

  makeplot : bool, default=False
             plot the impedance list
  ax : axes, default=None
       when makeplot=True these can be the axes

  returns:
  --------
  impedances: np.ndarray
              an array of impedance (complex) values
  """
  if makeplot:
    ax = qplot(ax)
  combine_list = []
  for elm in group:
    if isinstance(elm, np.ndarray):
      combine_list.append(elm)
    elif isinstance(elm, (list, tuple)):
      impedances = calc_group_impedances(elm)
      combine_list.append(impedances)
    else:
      raise TypeError("The type {type(elm)} is not implemented")

  # all is in, combine the groups to a group impedance
  if isinstance(group, list):
    impedances = np.max(np.asarray(combine_list), axis=0)
  elif isinstance(group, tuple):
    impedances = np.min(np.asarray(combine_list), axis=0)
  else:
    raise TypeError(f"The group type ({type(group)}) is not expected!")

  return impedances


def bode_zero_asymp(freqs, fz, order=1, Q=0.5, inverted=False):
  """ create a bode plot in asymptotes

  """
  ws = 2*np.pi*freqs
  amps = np.zeros_like(ws)
  sign = 1 - 2*float(inverted)

  if inverted:
    ws_valid = freqs < fz
    start_phase_rad = -np.pi/2
  else:
    ws_valid = freqs >= fz
    start_phase_rad = 0.0

  amps[ws_valid] = order*sign*logmod(ws[ws_valid]/(6.2832*fz))

  # phases
  fbelow = 10**(-1/(2*Q))*fz
  fabove = 10**(1/(2*Q))*fz
  total_phase_change_rad = order*np.pi/2
  slope_rad_p_dec = total_phase_change_rad/np.log10(fabove/fbelow)

  phases = (start_phase_rad
            + slope_rad_p_dec*np.log10(freqs/fbelow))
  phases[freqs < fbelow] = start_phase_rad
  phases[freqs > fabove] = start_phase_rad + total_phase_change_rad

  return amps, phases


def bode_pole_asymp(freqs, fp, order=1, Q=0.5, inverted=False):
  """ asymptotic approximation of a pole

  """
  ws = 2*np.pi*freqs
  amps = np.zeros_like(ws)
  sign = 1 - 2*float(inverted)

  # dB
  if inverted:
    ws_valid = freqs < fp
    start_phase_rad = np.pi/2
  else:
    ws_valid = freqs >= fp
    start_phase_rad = 0.0

  # amplitudes
  amps[ws_valid] = order*sign*logmod(6.2832*fp/ws[ws_valid])

  # phase
  fbelow = 10**(-1/(2*Q))*fp
  fabove = 10**(1/(2*Q))*fp
  total_phase_change_rad = -(order*np.pi/2)
  slope_rad_p_dec = total_phase_change_rad/np.log10(fabove/fbelow)

  print(fbelow, fp, fabove, np.rad2deg(total_phase_change_rad), np.rad2deg(slope_rad_p_dec))

  phases = (start_phase_rad + slope_rad_p_dec*np.log10(freqs/fbelow))  # full slope
  phases[freqs < fbelow] = start_phase_rad
  phases[freqs > fabove] = start_phase_rad + total_phase_change_rad

  return (amps, phases)


def bode_asymp(freqs, fpoles=None, fzeros=None, gain=1):
  """ a combined bode plot of poles, zeros and gains in dB

  using ONLY ASYMPTOTES!!

  """
  amps_final = logmod(gain*np.ones_like(freqs))
  phases_final = np.zeros_like(amps_final)

  if fpoles is not None:
    for fpole_ in fpoles:
      if isinstance(fpole_, (np.ndarray, list, tuple)):
        fpole, order, Q = fpole_
      else:
        fpole = fpole_
        Q = 0.5
        order=1
      inverted = fpole < 0
      amps_this, phases_this = bode_pole_asymp(freqs, np.abs(fpole), order=order, Q=Q,
                                               inverted=inverted)
      amps_final += amps_this
      phases_final += phases_this

  if fzeros is not None:
    for fzero_ in fzeros:
      if isinstance(fzero_, (np.ndarray, list, tuple)):
        fzero, order, Q = fzero_
      else:
        fzero = fzero_
        Q = 0.5
        order = 1
      inverted = fzero < 0
      amps_this, phases_this = bode_zero_asymp(freqs, np.abs(fzero), order=order, Q=Q,
                                               inverted=inverted)
      amps_final += amps_this
      phases_final += phases_this

  return amps_final, phases_final


def plot_bode_single_freq(freqs, btype, fc, order=1, Q=0.5, inverted=False, gain=1.0,
                          axs=None, label=None, **plotkwargs):
  """ plot a dual bode plot of amplitudes and phases

  """
  # ------------ create plot (if axs is None) ---------------------------------------------
  # predefined bode (may be otherwise calculated)
  if btype == 'pole':
    amps, phases = bode_pole_asymp(freqs, fc, order=order, Q=Q, inverted=inverted)
  elif btype == 'zero':
    amps, phases = bode_zero_asymp(freqs, fc, order=order, Q=Q, inverted=inverted)
  else:
    raise TypeError("The btype variable value given {btype} is incorrect. 'pole' or 'zero' are OK")

  print(f"label is: {label}")
  if label is None:
    label=f"{btype} @ {format_as_si(fc)}Hz"
  plotkwargs['label'] = label

  # add the indication of the Q
  if order == 2:
    Qdb = logmod(Q)
    ipk = np.argmin(np.abs(freqs - fc)).ravel()
    if btype == 'pole':
      amps[ipk] += Qdb
    else:
      amps[ipk] -= Qdb

  # compensate for gain
  amps += logmod(np.abs(gain))
  phases += np.arctan2(np.imag(gain), np.real(gain))

  freqs2plot = [fc]
  axs = plot_bode_precalc(freqs, amps, phases, axs=axs, freqs2plot=freqs2plot, **plotkwargs)

  return axs


def plot_bode_precalc(freqs, amps, phss, axs=None, freqs2plot=None, phss2plot=None,
                      amps2plot=None, color='r', **plotkwargs):
  """ calculate bode plots that are already precalculated; the amplitudes and phases are already
  known. Particularly useful for compound bode plots


  """
  plotkwargs['marker'] = 'none'
  if phss2plot is None:
    phss2plot = [-180.0]
  else:
    phss2plot = list(phss2plot).append(-180.0)

  if amps2plot is None:
    amps2plot = [0.0]
  else:
    amps2plot = list(amps2plot).append(0.0)

  if axs is None:
    _, axs = plt.subplots(2, 1, sharex=True, sharey=False)
    axs[0].set_title("Gain")
    axs[0].set_ylabel("Amplitude [dB]")
    axs[1].set_xlabel("freqs [Hz]")
    for ax in axs:
      ax.grid(visible=True, which='both', axis='x', alpha=0.2)
      ax.grid(visible=True, which='major', axis='y', alpha=0.2)

  if freqs2plot is not None:
    add_critical_freqs(axs, freqs2plot, colors=color)

  for phs_to_plot in phss2plot:
    axs[1].axhline(y=phs_to_plot, ls='--', color='k')
    axs[1].annotate(f"{phs_to_plot:0.1f}",
                    xy=(0.0, phs_to_plot),
                    xytext=(0.0, phs_to_plot),
                    xycoords=('axes fraction', 'data'),
                    ha='left', va='center', backgroundcolor='white')

  for amp_to_plot in amps2plot:
    axs[0].axhline(y=amp_to_plot, ls='--', color='k')
    axs[0].annotate(f"{amp_to_plot:0.1f}",
                    xy=(0.0, amp_to_plot),
                    xytext=(0.0, amp_to_plot),
                    xycoords=('axes fraction', 'data'),
                    ha='left', va='center', backgroundcolor='white')
  ax = axs[0]
  plotkwargs.update({'legend': True, 'legend_loc': 'upper right'})
  qplot(ax, freqs, amps, color=color, **plotkwargs)

  ax = axs[1]
  plotkwargs.update({'legend': False})
  qplot(ax, freqs, np.rad2deg(phss), color=color, **plotkwargs)
  plt.draw()
  plt.pause(1e-2)

  return axs


def break_frequencies(fcs, span=10):
  """ find the break frequencies

  """
  breaks_list = []
  if np.isscalar(fcs):
    fcs = np.asarray([fcs])

  print(fcs)
  for fc in fcs:
    if np.isscalar(fc):
      fbreaks = (fc/span, fc*span)
    else:
      fc, order, q = fc
      if order == 2:
        fbreaks = (10**(-1/(2*q))*fc, 10**(1/(2*q))*fc)
      else:
        raise NotImplementedError(f"the case for order = {order} is not implemented (yet)")

    breaks_list.append(fbreaks)

  return breaks_list


def add_critical_freqs(axs, fcs, colors='k', shade_color='auto', shade_alpha=0.1):
  """ add the critical frequencies to a bode plot (both of them)

  """
  # plot the vlines for frequencies in both plots
  if isinstance(fcs, (np.integer, np.floating)):
    fcs = [fcs]

  nof_freqs = len(fcs)
  if isinstance(colors, str):
    colors = np.asarray([colorvec(colors).tolist()]*nof_freqs)
  elif isinstance(colors, (tuple, list)):
    colors = np.asarray(colors)

  if colors.ndim == 1:
    colors = colors.reshape(1, -1)
    colors = np.tile(colors, [nof_freqs, 1])

  if colors.shape != (nof_freqs, 3):
    raise ValueError(f"The defined colors array shape ({colors.shape}) is not the expected "
                     f"shape ({(nof_freqs, 3)})")

  # -----------
  fbreaks = break_frequencies(fcs, span=10)

  # do it to both axes
  for ax in axs:
    for ifreq, (fc, fbreak) in enumerate(zip(fcs, fbreaks)):
      if isinstance(fc, (tuple, list, np.ndarray)):
        fc = fc[0]
      if shade_color == 'auto':
        shade_color_this = colors[ifreq]
      else:
        shade_color_this = shade_color
      ax.axvspan(xmin=fbreak[0], xmax=fbreak[1], color=shade_color_this,
                  alpha=shade_alpha)
      ax.axvline(x=fc, ls='--', lw=1, color=colors[ifreq])

      ax.annotate(format_as_si(fc),
                  xy=(fc, 0),
                  xytext=(fc, 0.05),
                  xycoords=('data', 'axes fraction'),
                  color='k',
                  ha='center', va='bottom', backgroundcolor='white',
                  bbox=dict(boxstyle='square, pad=0', ec='none', fc='white', alpha=0.5))

  plt.draw()
  plt.pause(1e-2)

  return axs


def add_phase_lines(ax, phases, colors='k'):
  """ add lines indicating a certain phase

  """
  if isinstance(phases, (np.floating, np.integer)):
    phases = [phases]

  nof_freqs = len(phases)
  if isinstance(colors, str):
    colors = np.asarray([colorvec(colors).tolist()]*nof_freqs)
  elif isinstance(colors, (tuple, list)):
    colors = np.asarray(colors)

  if colors.ndim == 1:
    colors = colors.reshape(1, -1)
    colors = np.tile(colors, [nof_freqs, 1])

  if colors.shape != (nof_freqs, 3):
    raise ValueError(f"The defined colors array shape ({colors.shape}) is not the expected "
                     f"shape ({(nof_freqs, 3)})")

  for phs, color in zip(phases, colors):
    ax.axhline(y=phs, ls='--', color=color)
    ax.annotate(f"{phs:0.1f}", xy=(0.05, phs), xytext=(0.0, phs),
                xycoords=('axes fraction', 'data'),
                ha='left', va='center', backgroundcolor='white',
                bbox=dict(boxstyle='square, pad=0', ec='none', fc='white', alpha=0.8))
