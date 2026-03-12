'''
contains functions used for coordinate transformations
'''

import numpy as np
import warnings

upstream = 'yxz'
downstream = 'zxy'

angle_signs = np.zeros((3,), dtype=[('p2enu', float),
                                    ('ar2p', float),
                                    ('am2ar', float),
                                    ('aa2am', float)])
angle_signs['p2enu'] = (-1, 1, -1)
angle_signs['ar2p'] = (0, 0, -1)
angle_signs['am2ar'] = (1, 1, 0)
angle_signs['aa2am'] = (1, 1, -1)

# signs depending on if it is (cw) clockwise or counterclockwise (ccw)
cstypes = dict(left={'cw': +1, 'ccw': -1},
               right={'cw': -1, 'ccw': +1})


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


# some Thales specific prescribed coordinate transformations
def csinfo(name):
  '''
  provide information on a specific predefined coordinate system

  Argument:
  ---------
  name : str
         The name of a coordinate system, such as 'ENU' or 'pcs'
  print : bool, default: True
          Whether to print the data

  Returns:
  --------
  data : dict
         A dictionary holding information on this system
  '''

  # init data dictionary
  data = dict(name=dict(full='', short=''),
              info='',
              handed='right',
              origin=dict(full='',
                          short=''),
              axis=dict(x='',
                        y='',
                        z=''),
              angles=dict(x=dict(name='n.a.', posrot='n.a.', extra='n.a.'),
                          y=dict(name='n.a.', posrot='n.a.', extra='n.a.'),
                          z=dict(name='n.a.', posrot='n.a.', extra='n.a.')))

  if name.lower() == 'nhcs':
    data['name'] = dict(full='North-Horizontal Coordinate System', short='NHCS')
    data['info'] = ('The North-Horizontal Coordinate System (NHCS) is an earth-fixed system.\n' +
                    'It is equal to the more commonly known name East-North-Up (ENU) system. ' +
                    'NHCS is a Thales NL-only term.\n\n' +
                    'All targets are defined in this system to allow handing over designations' +
                    ' becoming possible between systems/platforms')
    data['origin'] = dict(full='Ships Reference Point', short='SRP')
    data['axis'] = dict(x='points east',
                        y='points north',
                        z='points upward to zenith')
    data['angles'] = dict(x=dict(name='pitch', posrot='cw', extra='bow dow'),
                          y=dict(name='roll', posrot='ccw', extra='starboard down'),
                          z=dict(name='heading', posrot='cw', extra='n.a.'))

  if name.lower() == 'enu':
    data['name'] = dict(full='East, North, Up coordinate system', short='ENU')
    data['info'] = ('The East, North, Up Coordinate System (ENU) is an earth-fixed system.\n' +
                    'It is equal to the Thales-only name North Horizontal Coordinate System' +
                    ' (NHCS)\n')
    data['origin'] = dict(full='Ships Reference Point', short='SRP')
    data['axis'] = dict(x='points east',
                        y='points north',
                        z='points upward to zenith')
    data['angles'] = dict(x=dict(name='pitch', posrot='cw', extra='bow dow'),
                          y=dict(name='roll', posrot='ccw', extra='starboard down'),
                          z=dict(name='heading', posrot='cw', extra='n.a.'))

  elif name.lower() == 'pcs':
    data['name'] = dict(full='Platform Coordinate System', short='PCS')
    data['info'] = ('The Platform Coordinate System (PCS) is a ship-fixed system.\n' +
                    'The PCS rotates with the ship in the ENU, but has the same reference point')
    data['origin'] = dict(full='Ships Reference Point', short='SRP')
    data['axis'] = dict(x='points to starboard',
                        y='points to the bow',
                        z='points upward from the ship''s deck')
    data['angles']['z'] = dict(name='bearing', posrot='cw', extra='from bow to starboard')

  elif name.lower() == 'arcs':
    data['name'] = dict(full='Antenna Rotating Coordinate System (ARCS)', short='ARCS')
    data['info'] = ('The Antenna Rotating Coordinate System (ARCS) is a translated PCS ' +
                    'placeholder which is only applicable in rotating antenna systems.\n' +
                    'The ARCS origin is the point aroud which the antenna rotates and thus the' +
                    'reference point is termed `ant_rot`. The z-axis is upward through this' +
                    ' and is defined to be the rotation axis.\n' +
                    'The arm connecting the ant_rot to the antenna face is the positive y-axis')
    data['origin'] = dict(full='Antenna rotation reference point', short='ant_rot')
    data['axis'] = \
        dict(x='points right along the IDEAL antenna face (viewed from positive y-axis)',
             y='points along the normal of the IDEAL antenna face outward/upstream',
             z='points upward along the IDEAL antenna face (viewed from positive y-axis)')
    data['angles']['x'] = dict(name='tilt', posrot='ccw', extra='face turned upward')
    data['angles']['y'] = dict(name='queer', posrot='ccw', extra='n.a.')

  elif name.lower() == 'amcs':
    data['name'] = dict(full='Antenna Mounting Coordinate System', short='AMCS')
    data['info'] = ('The Antenna Mounting Coordinate System (AMCS) is the coordinate system in ' +
                    'which the IDEAL antenna face would be situated. This is thus before adding' +
                    'the misalignment angles\n\n' +
                    'The real antenna face is misaligned via the angles defined in this CS')
    data['origin'] = dict(full='Face Reference Point', short='FRP')
    data['axis'] = \
        dict(x='points right along the IDEAL antenna face (viewed from positive y-axis)',
             y='points along the normal of the IDEAL antenna face outward/upstream',
             z='points upward along the IDEAL antenna face (viewed from positive y-axis)')
    data['angles']['x'] = dict(name='local_tilt', posrot='ccw', extra='face turned upward')
    data['angles']['y'] = dict(name='local_queer', posrot='ccw', extra='n.a.')
    data['angles']['z'] = dict(name='local_bearing', posrot='cw', extra='from bow to starboard')

  elif name.lower() == 'aacs':
    data['name'] = dict(full='Antenna Aperture Coordinate System', short='AACS')
    data['info'] = ('The Antenna Aperture Coordinate System (AACS) is the coordinate system that' +
                    ' is actually aligned with the physical/electrical antenna face.\n' +
                    'It is slightly misaligned regarding the AMCS via the `local_<xx>` angles\n' +
                    'All targets are measured in this system.')
    data['origin'] = dict(full='Face Reference Point', short='FRP')
    data['axis'] = \
        dict(x='points right along the REAL antenna face (viewed from positive y-axis)',
             y='points along the normal of the REAL antenna face outward/upstream',
             z='points upward along the REAL antenna face (viewed from positive y-axis)')

  elif name.lower() == 'afcs':
    data['name'] = dict(full='Antenna Farmer Coordinate System', short='AFCS')
    data['info'] = ('The Antenna Farmer Coordinate System (AFCS) is the coordinate system in ' +
                    'which the antenna `farmers` (a friendly nickname) or experts define ' +
                    'the antenna. It is a 3D representation of a 2D plane in which the axes ' +
                    'are x is to the right, and y is upward. This implies that the viewpoint' +
                    ' is located at the designers eye')
    data['origin'] = dict(full='Face Reference Point', short='FRP')
    data['axis'] = \
        dict(x='points right along the REAL antenna face (viewed from positive z-axis)',
             y='points upward along the REAL antenna face (viewed from positive z-axis)',
             z='points along the normal of the REAL antenna face outward/upstream')

  elif name.lower() == 'ascs':
    data['name'] = dict(full='Antenna Steering Coordinate System', short='ASCS')
    data['info'] = ('The Antenna Steering Coordinate System (ASCS) is not really a coordinate ' +
                    'system. All points in ASCS are points on a unit sphere and have ' +
                    'coordinates u, v and w. ASCS is also termed `sine space` for the general ' +
                    'public, while ASCS is Thales only (sigh!).\n' +
                    'A noteworthy property is the constraint u**2 + v**2 + w**2 = 1, which ' +
                    'allows for the definition only of u and v.\n' +
                    'Note also that due to the fact that ASCS is generally based on AFCS and not' +
                    ' AACS, some intermediate actions must be performed to transform AACS to ' +
                    'ASCS')
    data['origin'] = dict(full='Face Reference Point', short='FRP')
    data['axis'] = \
        dict(x='(u-axis) points right along the REAL antenna face (viewed from positive w-axis)',
             y='(v-axis) points upward along the REAL antenna face (viewed from positive w-axis)',
             z='(w-axis) points along the normal of the REAL antenna face outward/upstream')

  else:
    raise NotImplementedError('The data for coordinate system `{}` is not implemented yet'.
                              format(name.upper()))

  # print this data
  print('{} ({})\n'.format(data['name']['full'], data['name']['short']) +
        '-----------------------------------------------------------------\n\n' +
        data['info'] +
        '\n\n' +
        'Reference point:\n' +
        '----------------\n' +
        ' {} : {}\n\n'.format(data['origin']['short'], data['origin']['full']) +
        'Axes info:\n' +
        '----------\n' +
        ' X-axis : {}\n'.format(data['axis']['x']) +
        ' Y-axis : {}\n'.format(data['axis']['y']) +
        ' Z-axis : {}\n\n'.format(data['axis']['z']) +
        'Angle definitions:\n' +
        '------------------\n' +
        ' X-axis : {} ({}). Extra info: {}\n'.format(data['angles']['x']['name'],
                                                     data['angles']['x']['posrot'],
                                                     data['angles']['x']['extra']) +
        ' Y-axis : {} ({}). Extra info: {}\n'.format(data['angles']['y']['name'],
                                                     data['angles']['y']['posrot'],
                                                     data['angles']['y']['extra']) +
        ' Z-axis : {} ({}). Extra info: {}\n'.format(data['angles']['z']['name'],
                                                     data['angles']['z']['posrot'],
                                                     data['angles']['z']['extra']))

  return data


def enu2p(pin, pitch=0., roll=0., heading=0.):
  '''
  ENU to PCS (upstream) transformation.

  Note that both CS share the origin (in the SRP)

  Positional argument:
  --------------------
  pin : ndarray of floats
        The input points in ENU to be transformed to PCS

  Keyword arguments:
  ------------------
  pitch : float
          The pitch of the platform in radians
  roll : float
         The roll of the platform in radians
  heading : float
            The heading of the platform in radians

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in PCS
  '''
  # print(pin)
  angles_xyz = -angle_signs['p2enu']*np.array([pitch, roll, heading])
  pout = ctform(pin, angles_xyz, None, upstream)
  return pout


def p2enu(pin, pitch=0., roll=0., heading=0.):
  '''
  PCS to ENU (downstream) transformation.

  Note that both CS share the origin (in the SRP)

  Positional argument:
  --------------------
  pin : ndarray of floats
        The input points in PCS to be transformed to ENU

  Keyword arguments:
  ------------------
  pitch : float, default: 0.
          The pitch of the platform in radians
  roll : float, default: 0.
         The roll of the platform in radians
  heading : float, default: 0.
            The heading of the platform in radians

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in ENU

  '''
  angles_xyz = angle_signs['p2enu']*np.array([pitch, roll, heading])
  pout = ctform(pin, angles_xyz, None, downstream)
  return pout


def p2ar(pin, bearing=0., tx=0., ty=0., tz=0.):
  '''
  transformation from PCS to ARCS (upstream)

  Note that a arm may be defined here, both CS do not share an origin principally

  Positional Argument:
  --------------------
  pin : 3xN ndarray of floats
        An array of PCS points

  Keyword arguments:
  ------------------
  bearing : float, default: 0.
            The bearing position of the antenna normal relative to the ship's bow in radians
  tx : float, default: 0.
       translation from SRP to ant_rot (x-axis) in meters
  ty : float, default: 0.
       the translation from SRP to ant_rot (y-axis) in meters
  tz : float, default: 0.
       the translation from SRP to ant_rot (z-axis) in meters

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in ARCS
  '''
  angles_xyz = -angle_signs['ar2p']*np.array([0, 0, bearing])
  pout = ctform(pin, angles_xyz, [tx, ty, tz], order_rotations=upstream)
  return pout


def p2am(pin, tilt=0., queer=0., bearing=0., tx=0., ty=0., tz=0., is_rotating=False, arm=0.):
  '''
  transformation from PCS to AMCS (upstream)

  Note that a arm may be defined here, both CS do not share an origin principally

  Positional Argument:
  --------------------
  pin : 3xN ndarray of floats
        An array of PCS points

  Keyword arguments:
  ------------------
  tilt : float, default: 0.
         The antenna tilt
  queer : float, default: 0.
          The antenna float
  bearing : float, default: 0.
            The bearing position of the antenna normal relative to the ship's bow in radians
  tx : float, default: 0.
       translation from SRP to AMCS (x-axis) in meters
  ty : float, default: 0.
       the translation from SRP to AMCS (y-axis) in meters
  tz : float, default: 0.
       the translation from SRP to AMCS (z-axis) in meters
  is_rotating : bool, default: False
                Flag stating if the antenna is rotating
  arm : float, default: 0.
        The arm between the ant_rot and AMCS (only valid in case *is_rotating* is True)

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in AMCS
  '''
  if is_rotating:
    # PCS -> ARCS
    angles_xyz = -angle_signs['ar2p']*np.array([0, 0, bearing])
    par = ctform(pin, angles_xyz, [tx, ty-arm, tz], order_rotations=upstream)

    # ARCS -> AMCS
    angles_xyz = -angle_signs['am2ar']*np.array([tilt, queer, 0.])
    pout = ctform(par, angles_xyz, [0., arm, 0.], order_rotations=upstream)
  else:
    angles_xyz = (-angle_signs['ar2p'] + -angle_signs['am2ar'])*np.array([tilt, queer, bearing])
    par = ctform(pin, angles_xyz, [tx, ty, tz], order_rotations=upstream)

  return pout


def am2p(pin, tilt=0., queer=0., bearing=0., tx=0., ty=0., tz=0., is_rotating=False, arm=0.):
  '''
  transformation from AMCS to PCS (downstream)

  Note that a arm may be defined here, both CS do not share an origin principally

  Positional Argument:
  --------------------
  pin : 3xN ndarray of floats
        An array of PCS points

  Keyword arguments:
  ------------------
  tilt : float, default: 0.
         The antenna tilt
  queer : float, default: 0.
          The antenna float
  bearing : float, default: 0.
            The bearing position of the antenna normal relative to the ship's bow in radians
  tx : float, default: 0.
       translation from SRP to AMCS (x-axis) in meters
  ty : float, default: 0.
       the translation from SRP to AMCS (y-axis) in meters
  tz : float, default: 0.
       the translation from SRP to AMCS (z-axis) in meters
  is_rotating : bool, default: False
                Flag stating if the antenna is rotating
  arm : float, default: 0.
        The arm between the ant_rot and AMCS (only valid in case *is_rotating* is True)

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in AMCS
  '''

  # arm is always positive!!
  arm = np.abs(arm)

  if is_rotating:
    # PCS -> ARCS
    angles_xyz = angle_signs['ar2p']*np.array([0, 0, bearing])
    par = ctform(pin, angles_xyz, [tx, ty+arm, tz], order_rotations=downstream)

    # ARCS -> AMCS
    angles_xyz = angle_signs['am2ar']*np.array([tilt, queer, 0.])
    pout = ctform(par, angles_xyz, [0., -arm, 0.], order_rotations=downstream)
  else:
    angles_xyz = (angle_signs['ar2p'] + angle_signs['am2ar'])*np.array([tilt, queer, bearing])
    par = ctform(pin, angles_xyz, [tx, ty, tz], order_rotations=downstream)

  return pout


def ar2p(pin, bearing=0., tx=0., ty=0., tz=0.):
  '''
  transformation from ARCS to PCS (downstream)

  Note that a arm may be defined here, both CS do not share an origin principally

  Positional Argument:
  --------------------
  pin : 3xN ndarray of floats
        An array of ARCS points

  Keyword arguments:
  ------------------
  bearing : float, default: 0.
            The bearing position of the antenna normal relative to the ship's bow in radians
  tx : float, default: 0.
       translation from ant_rot to SRP (x-axis) in meters
  ty : float, default: 0.
       the translation from ant_rot to SRP (y-axis) in meters
  tz : float, default: 0.
       the translation from ant_rot to SRP (z-axis) in meters

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in PCS
  '''
  angles_xyz = angle_signs['ar2p']*np.array([0, 0, bearing])
  pout = ctform(pin, angles_xyz, [tx, ty, tz], order_rotations=downstream)
  return pout


def ar2am(pin, tilt=0., queer=0., ty=0.):
  '''
  transformation from ARCS to AMCS (upstream)

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in ARCS

  Keyword arguments:
  ------------------
  tilt : float, default: 0.
         The tilt of the antenna in radians
  queer : float, default: 0.
          The queer of the antenna in radians
  ty : float, default: 0.
       The distance between ant_rot and the center of the face (FRP) in meters

  Returns:
  --------
  pout : 3xN ndarray of floats
         The set of points in AMCS
  '''
  angles_xyz = -angle_signs['am2ar']*np.array([tilt, queer, 0.])
  pout = ctform(pin, angles_xyz, [0., ty, 0.], order_rotations=upstream)
  return pout


def am2ar(pin, tilt=0., queer=0., ty=0.):
  '''
  transformation from AMCS to ARCS (downstream)

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in AMCS

  Keyword arguments:
  ------------------
  tilt : float, default: 0.
         The tilt of the antenna in radians
  queer : float, default: 0.
          The queer of the antenna in radians
  ty : float, default: 0.
       The distance between the FRP and the ant_rot in m

  Returns:
  --------
  pout : 3xN ndarray of floats
         The set of points in ARCS
  '''
  angles_xyz = angle_signs['am2ar']*np.array([tilt, queer, 0.])
  pout = ctform(pin, angles_xyz, [0., ty, 0.], order_rotations=downstream)
  return pout


def am2aa(pin, local_tilt=0., local_queer=0., local_bearing=0.):
  '''
  Transformation from AMCS to AACS (upstream)

  Note: Both CS share an origin (FRP), and is thus only rotations

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in AMCS

  keyword arguments:
  ------------------
  local_tilt : float, default: 0.
               The local_tilt angle in radians
  local_queer : float, default: 0.
                The local_queer angle in radians
  local_bearing : float, default: 0.
                  The local_bearing angle in radians

  Returns:
  --------
  pout : 3xN ndarray of floats
         the set of points in AACS
  '''
  angles_xyz = -angle_signs['aa2am']*np.array([local_tilt, local_queer, local_bearing])
  pout = ctform(pin, angles_xyz, None, order_rotations=upstream)
  return pout


def aa2am(pin, local_tilt=0., local_queer=0., local_bearing=0.):
  '''
  Transformation from AACS to AMCS (downstream)

  Note: Both CS share an origin (FRP), and is thus only rotations

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in AACS

  keyword arguments:
  ------------------
  local_tilt : float, default: 0.
               The local_tilt angle in radians
  local_queer : float, default: 0.
                The local_queer angle in radians
  local_bearing : float, default: 0.
                  The local_bearing angle in radians

  Returns:
  --------
  pout : 3xN ndarray of floats
         the set of points in AMCS
  '''
  angles_xyz = angle_signs['aa2am']*np.array([local_tilt, local_queer, local_bearing])
  pout = ctform(pin, angles_xyz, None, order_rotations=downstream)
  return pout


def aa2af(pin):
  '''
  Transforms points in AACS to AFCS (Antenna Farmer Coordinate System).

  Note: This AFCS is the coordinate system used mainly by antenna engineers ("antenna farmers")
  and uses a front-view, 2D coordinate system in which the x-axis points to the right,
  and the y-axis points upward

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The points in AACS

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in AFCS
  '''
  rots = [-np.pi/2, 0., np.pi]
  pout = ctform(pin, rots, [0., 0., 0.], order_rotations=upstream)
  return pout


def af2aa(pin):
  '''
  Transforms points in AFCS (Antenna Farmer Coordinate System) to AACS.

  Note: This AFCS is the coordinate system used mainly by antenna engineers ("antenna farmers")
  and uses a front-view, 2D coordinate system in which the x-axis points to the right,
  and the y-axis points upward

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The points in AFCS

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in AACS
  '''
  rots = [np.pi/2, 0., np.pi]
  pout = ctform(pin, rots, [0., 0., 0.], order_rotations=downstream)
  return pout


def af2as(pin):
  '''
  Convert coordiante in AFCS to ASCS  or `sine space` (upstream)

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in AFCS

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in ASCS. All positions are normalized
  '''
  pout = pin/np.linalg.norm(pin, 2)
  return pout


def aa2as(pin):
  '''
  convert coordinates in AACS to ASCS (via AFCS) (upstream)

  Positional argument:
  --------------------
  pin : 3xN ndarray of floats
        The positions in AACS

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions/directions in sine-space or ASCS
  '''
  pout = af2as(aa2af(pin))
  return pout


# some earth fixed stuff
def lla2ecef(lla):
  '''
  Convert Latitude, Longitude, Altitude (LLA) coordinates to Earth-Centered, Earth-Fixed (ECEF)
  coordinates

  Arguments:
  ----------
  lla : array-like or 3xN ndarray of floats
        The lla data

  Returns:
  --------
  ecef : 3xN (M>= 1) ndarray of floats
         The coordinates in ECEF
  '''

  # WGS84 constants
  a = 6378137.
  e = 8.1819190842622e-2

  # split in parts
  lat, lon, alt = lla

  # intermediate calculation (prime vertical radius of curvature)
  N = a/np.sqrt(1 - e**2*np.sin(lat)**2)

  # results
  x = (N + alt)*np.cos(lat)*np.cos(lon)
  y = (N + alt)*np.cos(lat)*np.sin(lon)
  z = ((1 - (e**2))*N + alt)*np.sin(lat)

  ecef = np.vstack((x, y, z))

  return ecef


def ecef2lla(pin):
  '''
  Convert ECEF coordinates to LLA coordinates

  Arguments:
  ----------
  pin : 3xN ndarray of floats
        The positions in ECEF

  Returns:
  --------
  pout : 3xN ndarray of floats
         The GPS coordinates (LLA)

  See also:
  ---------
  .lla2ecef : convert to ecef from lla
  .lla2nh : convert to ENU from LLA
  '''

  a = 6378137.
  e = 8.1819190842622e-2

  x = pin[0, :]
  y = pin[1, :]
  z = pin[2, :]

  # intermediate values
  b = np.sqrt(a**2*(1 - e**2))
  ep = np.sqrt((a**2 - b**2)/b**2)
  p = np.sqrt(x**2 + y**2)
  th = np.arctan2(a*z, b*p)
  lon = np.arctan2(y, x)
  lat = np.arctan2((z + ep**2*b*np.sin(th)**3), (p - e**2*a*np.cos(th)**3))
  N = a/np.sqrt(1 - e**2*np.sin(lat)**2)
  alt = p/np.cos(lat)-N

  lon = np.mod(lon, 2*np.pi)

  k = np.where(np.logical_and(np.abs(x) < 1, np.abs(y) < 1))[0]
  alt[k] = np.abs(z[k]) - b

  lla = np.vstack((lat, lon, alt))

  return lla


def enu2ecef(pin, lla_ref):
  '''
  convert a point in ENU to the Earth-Centered, Earth-Fixed (ECEF) coordinates.

  Arguments:
  ----------
  pin : 3xN ndarray of floats
        The positions in ENU
  lla_ref : 3xN ndarray of floats
            if shape is 3x1, the SRP is stated to have remained constant
            if shape is equal to *pin*, the measurements have a different LLA from point to point

  Returns:
  --------
  pout : 3xN ndarray of floats
         The ECEF coordinates
  '''
  def _conversion(lat, lon, alt):
    convmat = np.array([[-np.sin(lat), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                        [np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                        [0., np.cos(lat), np.sin(lat)]], dtype=float)
    return convmat

  if pin.shape == lla_ref.shape:
    pout = np.empty(pin.shape)
    for ipt, (pt, lla) in enumerate(zip(pin.T, lla_ref.T)):
      ecef_srp = lla2ecef(lla.T)  # may be more than 1 point
      lat, lon, alt = lla.T
      x, y, z = pt.T

      convmat = _conversion(lat, lon, alt)
      pout1 = convmat.dot(pt) + ecef_srp
      pout[:, ipt] = pout1

  else:
    lat, lon, alt = lla
    ecef_srp = lla2ecef(lla)  # may be more than 1 point
    convmat = _conversion(lat, lon, alt)
    pout = convmat.dot(pin) + ecef_srp

  return pout


def ecef2enu(pin, lla_ref):
  '''
  Convert an ECEF point to how it is viewed in ENU

  Arguments:
  ----------
  pin : 3xN ndarray of floats
        The ECEF positions
  lla_ref : 3-arrary-like or 3xN ndarray of floats
            The position or positions of the SRP in LLA

  Returns:
  --------
  pout : 3xN ndarray of floats
         The position in ENU
  '''
  def _conversion(lat, lon, alt):
    convmat = np.array([[-np.sin(lon), np.cos(lon), 0],
                        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                        [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]],
                       dtype=float)
    return convmat

  if pin.shape == lla_ref.shape:
    pout = np.empty(pin.shape)
    for ipt, (pt, lla) in enumerate(zip(pin.T, lla_ref.T)):
      ecef_ref = lla2ecef(lla.T)  # may be more than 1 point
      lat, lon, alt = lla.T
      x, y, z = pt.T

      convmat = _conversion(lat, lon, alt)
      pout1 = convmat.dot(pt - ecef_ref)
      pout[:, ipt] = pout1

  else:
    lat, lon, alt = lla
    ecef_ref = lla2ecef(lla)  # may be more than 1 point
    convmat = _conversion(lat, lon, alt)
    pout = convmat.dot(pin + ecef_ref)

  return pout


def enu2lla(pin, lla_ref):
  '''
  Convert ENU coordinates to LLA coordinates via the ships LLA position

  Note:
  Since adding angles together is not allowed, the ENU position is first converted to ECEF and
  subsequently transformed to LLA!!

  Arguments:
  ----------
  pin : 3xN ndarray of floats
        The ENU positions
  lla_ref : 3xN ndarray or 3-array-like of floats
            The LLA position of the SRP, may be a single position or equal to number of *pin*

  Returns:
  --------
  pout : 3xN ndarray of floats
         The positions in LLA
  '''

  pecef = nh2ecef(pin, lla_ref)
  plla = ecef2lla(pecef)

  return plla


def lla2enu(pin, lla_ref):
  '''
  Convert a LLA position to a ENU position (via ECEF)

  Arguments:
  ----------
  pin : 3xN ndarray of floats
        The positions in LLA
  lla_ref : 3xN ndarray or 3-array-like of floats
            The LLA position of the SRP, may be a single position or equal to number of *pin*

  Returns:
  --------
  pout : 3xN ndarray of floats
         The ENU positions
  '''
  pecef = lla2ecef(pin)
  pout = ecef2enu(pecef, lla_ref)

  return pout


def cart2sph(pcart, angle_units='rad'):
  '''
  convert a Cartesian point (xyz) to a spherical coordinate (range, azimuth, elevation)

  Argument:
  ---------
  pcart : 3xN ndarray of floats
          The Cartesian positions
  angle_units : ['rad' | 'deg' | 'unit'], default: 'rad'
               The unit of the angles in the polar coordinates

  Returns:
  --------
  pout : 3xN ndarray of floats
         The spherical positions range, azimuth and elevation
  '''
  if angle_units == 'rad':
    sfa = 1.
  elif angle_units == 'deg':
    sfa = 180/np.pi
  elif angle_units == 'unit':
    sfa = 1/(2*np.pi)
  else:
    raise ValueError('The *angle_mode=`{}`* is not valid'.format(angle_units))

  rngxy = np.hypot(pcart[0, :], pcart[1, :])
  rng = np.hypot(rngxy, pcart[2, :])
  azi = np.arctan2(pcart[0, :], pcart[1, :])*sfa
  elev = np.arctan2(pcart[2, :], rngxy)*sfa

  pout = np.vstack((rng, azi, elev))

  return pout


def sph2cart(psph, angle_units='rad'):
  '''
  convert spherical coordinates to Cartesian coordinates

  Arguments:
  ----------
  psph : 3xN ndarray of floats
         The polar coordinates. Order range, azimuth, elevation along the 0th axis
  angle_units : ['rad' | 'deg' | 'unit'], default: 'rad'
               The unit of the angles in the polar coordinates

  Returns:
  --------
  pout : 3xN ndarray of floats
         The points in Cartesian coordinates
  '''
  if angle_units == 'rad':
    sfa = 1.
  elif angle_units == 'deg':
    sfa = 180/np.pi
  elif angle_units == 'unit':
    sfa = 1/(2*np.pi)
  else:
    raise ValueError('The *angle_mode=`{}`* is not valid'.format(angle_units))

  rng = psph[0, :]
  azi = psph[1, :]/sfa
  elev = psph[2, :]/sfa

  z = rng*np.sin(elev)
  rngcose = rng*np.cos(elev)
  x = rngcose*np.sin(azi)
  y = rngcose*np.cos(azi)

  pout = np.vstack((x, y, z))
  return pout


# wrappers for ENU == ENU
# define wrapper decorator
def nhcs_is_enu_warning(func):
  def wrapper(*args, **kwargs):
    warnings.warn('\n\nThe coordinate system NHCS is equal to the more commonly known ENU.\n' +
                  'Please use the ENU to avoid this warning\n', UserWarning, stacklevel=1)
    pout = func(*args, **kwargs)
    return pout

  return wrapper


nh2p = nhcs_is_enu_warning(enu2p)
p2nh = nhcs_is_enu_warning(p2enu)
nh2ecef = nhcs_is_enu_warning(enu2ecef)
ecef2nf = nhcs_is_enu_warning(ecef2enu)
nh2lla = nhcs_is_enu_warning(enu2lla)
lla2nh = nhcs_is_enu_warning(lla2enu)
