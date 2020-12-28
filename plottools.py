"""
plottools

this module contains some preset plotting functions to easier create standard plots

author: Joris Kampman, Thales NL
"""

import matplotlib.pyplot as plt
import numpy as np
import common_tools as ct
from common_tools.cmaps import jetext
import matplotlib.gridspec as gridspec

pi = np.pi

__all__ = ['gain_phase_plot', 'plot_2D_beam_pattern']


def plot_2D_beam_pattern(px, py, coefs, freq,
                         xtaper='050DCxx', ytaper='040DCxx', predef_taper=None,
                         azi_bracket=[-pi/2, pi/2], elev_bracket=[-pi/2, pi/2],
                         vmin_dbc=-60.0, azi_steer=0.0, elev_steer=0.0,
                         nof_points_per_dim=[301, 301], nof_points_1D=5001, suptitle=None,
                         add_ref=True, center_mainlobe=False, tx_or_rx='rx', cmap=jetext(),
                         show_colorbar=True, figtitle=None):
  '''
  Gives a comprehensive 2D beam pattern plot plus cross-sections  in both dimensions through the
  mainlobe.

  Positional arguments:
  ---------------------
  px : ndarray of floats
       The horizontal element positions
  py : ndarray of floats
       The vertical element positions
  coefs : ndarray of complex floats
          the coefficients of the elements
  freq : float
         The frequency in Hz

  Keyword arguments (optional):
  -----------------------------
  xtaper : str or None, default : None
           Taper designation for horizontal direction (see common_tools.create_taper)
  ytaper : str or None, default : None
           Taper designation for vertical direction (see common_tools.create_taper)
  predef_taper : None or ndarray of complex floats, default=None
                 if not None, this array is the taper to be applied to the coefficients
  azi_bracket : 2-list of floats, default : [-pi/2, pi/2]
                The brackets in azimuth between which the patterns must be created
  elev_bracket : 2-list of floats, default : [-pi/2, pi/2]
                 The brackets in elevation between which the patterns must be created
  vmin_dbc : float or int, default : -60.0
             The minimum colormapping value in dBc (thus amount below the mainlobe)
  azi_steer : float, default : 0.0
              The amount of azimuth steering for the entire pattern in radians
  elev_steer : float, default : 0.0
               The amount of elevation steering for the entire pattern in radians
  nof_points_per_dim : 2-list of floats, default : [301, 301]
                       The number of points in each dimension.
  suptitle : None or str, default : None
             if not *None*, the string is placed on the top of the figure
  add_ref : boolean, default : True
            Flag indicating whether to add a reference in both the cross sections of the ideal
            taper
  center_mainlobe : boolean, default : False
                    Flag indicating whether to center the mainlobe. Setting this to *True* will
                    render keyword arguments *azi_steer* and *elev_steer* moot. The position of the
                    mainlobe will first be determined and the pattern will be subsequently shifted
                    to center the mainlobe.
  tx_or_rx : ['tx' | 'rx'], default : 'rx'
             Switch stating if the beam pattern is a tx or rx beam pattern.
  cmap : colormap instance, default; ct.cmaps.jetmod()

  Returns:
  --------
  fig : figure handle
        the figure handle

  See Also:
  ---------
  .find_mainlobe : Find the position (in angle coordinates or uv-coordinates) of the mainlobe
                   before having plotted the beam pattern
  common_tools.CreateBeam : Creates a beam, and calculates the pattern
  common_tools.figname : checks existence of chosen name
  common_tools.tighten : tighten the whitespace around the axes
  common_tools.amax_size_inches : 2-tuple of the dimensions to A-like, fitting on screen

  '''

  if nof_points_1D is None:
    nof_points_1D = nof_points_per_dim

  if type(nof_points_1D) is int:
    nof_points_1D = [nof_points_1D]*2

  # create beam pattern
  hcb = ct.beamforming.CreateBeam(coefs, px, py, xtaper=xtaper, ytaper=ytaper,
                                  predef_taper=predef_taper, cosine_decay=False, tx_or_rx=tx_or_rx,
                                  ae_or_uv='ae', nof_points_per_dim=nof_points_per_dim,
                                  azi_bracket=azi_bracket, elev_bracket=elev_bracket, freq=freq,
                                  normalize=True)

  azi_ml, elev_ml, max_power = hcb.find_mainlobe()

  print('Mainlobe at:\n',
        '  {:0.1f} deg azimuth\n'.format(azi_ml*180/pi),
        '  {:0.1f} deg elevation\n'.format(elev_ml*180/pi))

  azis_hr = np.linspace(*hcb.azi_bracket, nof_points_1D[0])
  elevs_hr = np.linspace(*hcb.elev_bracket, nof_points_1D[1])

  if center_mainlobe:
    azi_steer = azi_ml
    elev_steer = elev_ml

  # azimuth pattern
  hcb.azi_vector_predef = azis_hr
  hcb.elev_vector_predef = np.array([elev_ml])
  azipatt = hcb.calculate_beam_pattern(scale='logmod', azi_steer=azi_steer, elev_steer=elev_steer)

  # elevation pattern
  hcb.azi_vector_predef = np.array([azi_ml])
  hcb.elev_vector_predef = elevs_hr
  elevpatt = hcb.calculate_beam_pattern(scale='logmod', azi_steer=azi_steer, elev_steer=elev_steer)

  hcb.azi_vector_predef = None
  hcb.elev_vector_predef = None

  # 2D pattern
  pttn = hcb.calculate_beam_pattern(scale='logmod', azi_steer=azi_steer, elev_steer=elev_steer)

  # add reference or ideal pattern
  if add_ref:
    # init createbeam object
    hcbi = ct.beamforming.CreateBeam(np.ones_like(px, dtype='complex'), px, py,
                                     cosine_decay=False, tx_or_rx='rx', ae_or_uv='ae',
                                     freq=freq, normalize=True, xtaper=xtaper, ytaper=ytaper,
                                     predef_taper=predef_taper)

    # azimuth pattern
    hcbi.azi_vector_predef = azis_hr
    hcbi.elev_vector_predef = np.array([elev_ml])
    azipatt_ref = hcbi.calculate_beam_pattern(scale='logmod', azi_steer=-azi_ml,
                                              elev_steer=-elev_ml)

    # elevation pattern
    hcbi.azi_vector_predef = np.array([azi_ml])
    hcbi.elev_vector_predef = elevs_hr
    elevpatt_ref = hcbi.calculate_beam_pattern(scale='logmod', azi_steer=-azi_ml,
                                               elev_steer=-elev_ml)

  if figtitle is None:
    figtitle = 'beam_patterns_{:d}_MHz'.format(np.round(np.abs(freq)*1e-6).astype(int))

  fig = plt.figure(ct.figname(figtitle))
  if suptitle is None:
    suptitle = 'Beam patterns @ {:d} MHz'.format(np.round(freq*1e-6).astype(int))

  fig.suptitle(suptitle, fontsize=14, fontweight='bold')
  fig.set_size_inches(ct.amax_size_inches, forward=True)

  # create axes
  gs = gridspec.GridSpec(2, 2, width_ratios=[2*np.sqrt(2), 1], height_ratios=[2, 1])

  # ax00 : 2D pattern
  ax00 = plt.subplot(gs[0, 0])
  # ax00.set_title('2D beam pattern')
  ax00.set_xlabel('azimuth [deg]')
  ax00.set_ylabel('elevation [deg]')
  mp = ax00.imshow(pttn, cmap=cmap, extent=np.array(azi_bracket + elev_bracket)*180/pi,
                   vmin=pttn.max() + vmin_dbc, vmax=pttn.max(), origin='lower', aspect='auto',
                   interpolation='nearest')
  ax00.axhline(y=elev_ml*180/pi, ls='--', c='r', linewidth=3)
  ax00.axvline(x=azi_ml*180/pi, ls='--', c='r', linewidth=3)
  ax00.text(hcb._azis_u[0]*180/pi, elev_ml*180/pi,
            '{:0.2f} deg'.format(elev_ml*180/pi),
            fontweight='bold', fontsize=12, color='r', horizontalalignment='left',
            verticalalignment='bottom')
  ax00.text(azi_ml*180/pi, hcb._elevs_v[0]*180/pi,
            '{:0.2f} deg'.format(azi_ml*180/pi), fontweight='bold', fontsize=12, color='r',
            horizontalalignment='right',
            verticalalignment='bottom', rotation=90)
  if show_colorbar:
    plt.colorbar(mp, ax=ax00)

  # ax01: vertical cut
  ax01 = plt.subplot(gs[0, 1])
  if add_ref:
    ax01.plot(elevpatt_ref.reshape(-1), elevs_hr*180/pi, 'b-')
  ax01.plot(elevpatt.reshape(-1), elevs_hr*180/pi, 'r.-')

  ax01.grid(True)
  ax01.set_title('vertical cut')
  ax01.set_xlabel('power [dBc]')
  ax01.set_ylabel('elevation [deg]')
  ax01.set_ylim(np.array(elev_bracket)*180/pi)
  ax01.yaxis.set_ticks(np.arange(elev_bracket[0]*180/pi, elev_bracket[1]*180/pi + 0.1, 5))
  ax01.set_xlim(vmin_dbc, 0)
  ax01.xaxis.set_ticks(np.arange(vmin_dbc, 1, 5))

  # ax01: horizontal cut
  ax10 = plt.subplot(gs[1, 0])
  if add_ref:
    ax10.plot(azis_hr*180/pi, azipatt_ref.reshape(-1), 'b-')
  ax10.plot(azis_hr*180/pi, azipatt.reshape(-1), 'r.-')
  ax10.grid(True)
  ax10.set_title('Horizontal cut')
  ax10.set_xlabel('azimuth [deg]')
  ax10.set_ylabel('power [dBc]')
  ax10.set_xlim(np.array(azi_bracket)*180/pi)
  ax10.xaxis.set_ticks(np.arange(azi_bracket[0]*180/pi, azi_bracket[1]*180/pi + 0.1, 5))
  ax10.yaxis.set_ticks(np.arange(-vmin_dbc, 1, 5))
  ax10.set_ylim(vmin_dbc, 0)
  ax10.yaxis.set_ticks(np.arange(vmin_dbc, 1, 5))

  # wrap up
  ct.tighten(fig)
  plt.show(block=False)
  plt.pause(0.1)

  return fig


def _calc_phase_offset(signals, type='rad'):

  nof_elms = signals.size
  signalsn = np.exp(1j*np.angle(signals))

  phase_offset = np.angle((signalsn**(1/nof_elms)).prod())

  if type == 'rad':
    return phase_offset
  elif type == 'deg':
    return phase_offset*180/pi


def gain_phase_plot(px, py, signals, title=None, axs=None, display_means=True, amp_scale='db',
                    **kwargs_new):
  '''
  plot 2 figures hodling the gains and the phases for complex signals

  Arguments:
  ----------
  px : ndarray of floats
       horizontal positions
  py : ndarray of floats
       vertical positions
  signals : ndarray of complex floats
            signals to be plotted. Complex valued

  title : (None) None or str
          The title of the figure
  axs : (None) None or 2-list/2-tuple of axes objects
        The axes in which the plots must be made
  display_means : (True) bool
                  Whether to display the mean value on the image in text
  **kwargs_new : dict of kwargs
                  keyword arguments given to the scatter plot function for both amplitude and phase
                  plots
 '''

  # default keyword arguments
  kwargs = dict(edgecolors='face', s=20, marker='s', cmap=ct.cmaps.jetmod(bright=True))

  for key, value in kwargs_new.items():
    kwargs[key] = value

  pkwargs = kwargs.copy()
  akwargs = kwargs.copy()
  if 'vmin' not in kwargs.keys():
    pkwargs['vmin'] = ct.angled(signals).min()
  else:
    akwargs.pop('vmin')

  if 'vmax' not in kwargs.keys():
    pkwargs['vmax'] = ct.angled(signals).max()
  else:
    akwargs.pop('vmax')

  if pkwargs['vmax'] - pkwargs['vmin'] < 0.02:
    vmean = (pkwargs['vmax'] + pkwargs['vmin'])/2
    pkwargs['vmin'] = vmean - 0.01
    pkwargs['vmax'] = vmean + 0.01

  # plot full TX array
  if axs is None:
    title = ct.figname(title)
    fig, axs = plt.subplots(1, 2, num=title, subplot_kw=dict(xticks=[], yticks=[]))

  if amp_scale == 'lin':
    spa = axs[0].scatter(px, py, c=np.abs(signals), **akwargs)
  elif amp_scale == 'db':
    spa = axs[0].scatter(px, py, c=20*np.log10(np.abs(signals)), **akwargs)
  else:
    raise ValueError('[amp_scale="{}"] The value for keyword "amp_scale" is not valid'.
                     format(amp_scale))

  spp = axs[1].scatter(px, py, c=ct.angled(signals), **pkwargs)

  if display_means:
    txt = np.abs(signals).mean()
    if amp_scale == 'db':
      txt = 20*np.log10(txt)

    axs[0].text(px.mean(), py.mean(), '{:0.2f}'.format(txt),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='w',
                fontweight='bold')
    axs[1].text(px.mean(), py.mean(),
                '{:0.2f}/{:0.2f}'.format(ct.angled(signals.mean()),
                                         _calc_phase_offset(signals)*180/pi),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='w',
                fontweight='bold')

  plt.colorbar(spa, ax=axs[0])
  plt.colorbar(spp, ax=axs[1])

  plt.show(block=False)

  return axs, spa, spp
