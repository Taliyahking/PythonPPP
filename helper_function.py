import numpy as np
import json

def cal2jul(year, mon, day, sec):
    if not (np.isscalar(year) and np.isscalar(mon) and np.isscalar(day)):
        raise ValueError('Year, Month and Day should be scalar.')

    if mon < 1 or mon > 12:
        raise ValueError('Month should be between 1 and 12.')

    if day < 1 or day > 31:
        raise ValueError('Day should be between 1 and 31.')

    sec = sec / 3600

    if mon <= 2:
        m = mon + 12
        y = year - 1
    else:
        m = mon
        y = year

    jd = np.floor(365.25 * y) + np.floor(30.6001 * (m + 1)) + day + (sec / 24) + 1720981.5
    mjd = jd - 2400000.5

    return jd, mjd


def clc_doy(year, mon, day):
    if year % 400 == 0:
        sit = 1
    elif year % 300 == 0:
        sit = 0
    elif year % 200 == 0:
        sit = 0
    elif year % 100 == 0:
        sit = 0
    elif year % 4 == 0:
        sit = 1
    else:
        sit = 0

    if sit == 0:
        dom = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dom = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if mon - 1 < 1:
        doy = day
    else:
        doy = sum(dom[0:(mon - 1)]) + day

    return doy

def frequencies():

    c = 299792458
    freq = np.zeros((105, 2))
    wavl = np.zeros((105, 2))
    glok = [1, -4, 5, 6, 1, -4, 5, 6, -2, -7, 0, -1, -2, -7, 0, -1, 4, -3, 3, 2, 4, -3, 3, 2, 0, 0, 0]

    for i in range(1, 106):
        idx = i - 1
        if i < 33:  # GPS
            freq[idx, 0] = 10.23 * 10 ** 6 * 154  # Hz
            wavl[idx, 0] = c / (10.23 * 10 ** 6 * 154)  # m
            freq[idx, 1] = 10.23 * 10 ** 6 * 120  # Hz
            wavl[idx, 1] = c / (10.23 * 10 ** 6 * 120)  # m
        elif i < 60:  # GLONASS
            glok_idx = i - 33
            if glok_idx < len(glok):  # 确保不会超出glok数组范围
                freq[idx, 0] = (1602 + 0.5625 * glok[glok_idx]) * 10 ** 6  # Hz
                wavl[idx, 0] = c / ((1602 + 0.5625 * glok[glok_idx]) * 10 ** 6)  # m
                freq[idx, 1] = (1246 + 0.4375 * glok[glok_idx]) * 10 ** 6  # Hz
                wavl[idx, 1] = c / ((1246 + 0.4375 * glok[glok_idx]) * 10 ** 6)  # m
            else:
                # 对于超出glok范围的GLONASS卫星，使用默认值
                freq[idx, 0] = (1602) * 10 ** 6  # Hz
                wavl[idx, 0] = c / ((1602) * 10 ** 6)  # m
                freq[idx, 1] = (1246) * 10 ** 6  # Hz
                wavl[idx, 1] = c / ((1246) * 10 ** 6)  # m
        elif i < 96:  # GALILEO
            freq[idx, 0] = 10.23 * 10 ** 6 * 154  # Hz
            wavl[idx, 0] = c / (10.23 * 10 ** 6 * 154)  # m
            freq[idx, 1] = 10.23 * 10 ** 6 * 115  # Hz
            wavl[idx, 1] = c / (10.23 * 10 ** 6 * 115)  # m
        else:  # BEIDOU
            freq[idx, 0] = 10.23 * 10 ** 6 * 152.6  # Hz
            wavl[idx, 0] = c / (10.23 * 10 ** 6 * 152.6)  # m
            freq[idx, 1] = 10.23 * 10 ** 6 * 118  # Hz
            wavl[idx, 1] = c / (10.23 * 10 ** 6 * 118)  # m

    return freq, wavl

def xyz2plh(cart, dopt=0):
    cart = np.array(cart).flatten()  # 确保是一维数组
    if len(cart) != 3:
        raise ValueError(f'Input matrix must have exactly 3 components, got {len(cart)}')

    # WGS84 ellipsoid parameters
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f**2  # eccentricity squared

    # Calculate longitude
    lam = np.arctan2(cart[1], cart[0])
    lam = lam % (2 * np.pi)  # normalize to [0, 2π)

    # Calculate distance from Earth's axis
    p = np.sqrt(cart[0]**2 + cart[1]**2)

    # Initial estimate of latitude
    phi0 = np.arctan(cart[2] / (p * (1 - e2)))

    # Iterative calculation of latitude and height
    while True:
        N = a / np.sqrt(1 - (e2 * (np.sin(phi0)**2)))
        h = p / np.cos(phi0) - N
        phi = np.arctan((cart[2] / p) / (1 - (N / (N + h) * e2)))
        dphi = abs(phi - phi0)

        if dphi > 10**-12:
            phi0 = phi
        else:
            break

    # Return result in requested format
    if dopt == 0:
        return np.array([phi, lam, h])
    elif dopt == 1:
        t = (180 / np.pi)
        return np.array([phi * t, lam * t, h])

def rotation(position, angle, axis):
    """
    旋转坐标变换函数

    参数:
    position (array): 3D位置向量
    angle (float): 旋转角度(度)
    axis (int): 旋转轴(1=X, 2=Y, 3=Z)

    返回:
    array: 旋转后的位置向量
    """
    # 检查输入维度
    position = np.array(position).flatten()
    if len(position) != 3:
        raise ValueError('Matrix dimension should be 3xN or Nx3.')

    if not (np.isscalar(angle) and np.isscalar(axis)):
        raise ValueError('Angle and axis should be scalar.')

    # 创建旋转矩阵 - 注意使用角度制
    if axis == 1:  # X轴
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), np.sin(np.radians(angle))],
            [0, -np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
    elif axis == 2:  # Y轴
        rot = np.array([
            [np.cos(np.radians(angle)), 0, -np.sin(np.radians(angle))],
            [0, 1, 0],
            [np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ])
    elif axis == 3:  # Z轴
        rot = np.array([
            [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0],
            [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError('Axis must be 1, 2, or 3.')

    # 应用旋转
    xout = rot @ position

    return xout

def local(rec, sat, dopt):
    """
    Convert ECEF to local coordinates and calculate azimuth and elevation.

    Parameters:
    rec (array): Receiver position in ECEF
    sat (array): Satellite position in ECEF
    dopt (int): Output option (0=radians, 1=degrees)

    Returns:
    tuple: (azimuth, elevation)
    """
    if len(rec) != 3 or len(sat) != 3:
        raise ValueError('Receiver and satellite position vectors must include X,Y,Z')

    rec = np.array(rec)
    sat = np.array(sat)

    # Ensure consistent dimensions
    if rec.shape[0] != sat.shape[0]:
        if rec.shape[0] == 3:
            rec = rec.reshape(1, -1)
        elif sat.shape[0] == 3:
            sat = sat.reshape(1, -1)

    # Line of sight vector
    los = sat - rec

    # Normalized LOS vector
    p = los / np.linalg.norm(los)

    # Get geodetic coordinates
    ellp = xyz2plh(rec, 0)
    lat = ellp[0]
    lon = ellp[1]

    # Local coordinate system vectors (ENU)
    e = np.array([-np.sin(lon), np.cos(lon), 0])
    n = np.array([-np.cos(lon) * np.sin(lat), -np.sin(lon) * np.sin(lat), np.cos(lat)])
    u = np.array([np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)])

    # Calculate elevation and azimuth
    elev = np.arcsin(np.dot(p, u))
    azim = np.arctan2(np.dot(p, e), np.dot(p, n))
    azim = azim % (2 * np.pi)

    # Convert to degrees if requested
    if dopt == 1:
        elev = np.degrees(elev)
        azim = np.degrees(azim)

    return azim, elev
def i_free(o1, o2, opt):
    """
    Calculate ionosphere-free combination.

    Parameters:
    o1 (array): First observable
    o2 (array): Second observable
    opt (int): System option (0=GPS, 1=GLONASS, 2=GALILEO, 3=BEIDOU)

    Returns:
    array: Ionosphere-free combination
    """
    if opt == 0:
        ifr = o1 * 2.545727780163160 - o2 * 1.545727780163160
    elif opt == 1:
        ifr = o1 * 2.53125 - o2 * 1.53125
    elif opt == 2:
        ifr = o1 * 2.260604327518826 - o2 * 1.260604327518826
    elif opt == 3:
        ifr = o1 * 2.487168313616925 - o2 * 1.487168313616925

    return ifr
def decimation(data, options):
    """
    根据options截取需要计算的历元 (Trim observation data to specified time range)

    Parameters:
    data (dict): Data structure containing observations
    options (dict): Processing options with 'from' and 'to' time limits

    Returns:
    dict: Updated data structure with trimmed observations
    """
    if (options['from'] < data['obsd']['epoch'][0, 0]) or (options['to'] > data['obsd']['epoch'][-1, 0]):
        raise ValueError('Observation file may not contain data between the chosen interval.')

    f = (options['from'] - data['obsd']['epoch'][0, 0])
    l = (options['to'] - data['obsd']['epoch'][0, 0])
    fe = int(np.round(f / data['obsh']['time']['obsinterval'])) + 1  # first epoch
    le = int(np.round(l / data['obsh']['time']['obsinterval'])) + 1  # last epoch

    # Trim end of data if needed
    if le < data['obsd']['st'].shape[0]:
        data['obsd']['p1'] = data['obsd']['p1'][:le, :]
        data['obsd']['p2'] = data['obsd']['p2'][:le, :]
        data['obsd']['l1'] = data['obsd']['l1'][:le, :]
        data['obsd']['l2'] = data['obsd']['l2'][:le, :]
        data['obsd']['epoch'] = data['obsd']['epoch'][:le, :]
        data['obsd']['st'] = data['obsd']['st'][:le, :]

    # Trim beginning of data if needed
    if fe != 1:
        data['obsd']['p1'] = data['obsd']['p1'][fe - 1:, :]
        data['obsd']['p2'] = data['obsd']['p2'][fe - 1:, :]
        data['obsd']['l1'] = data['obsd']['l1'][fe - 1:, :]
        data['obsd']['l2'] = data['obsd']['l2'][fe - 1:, :]
        data['obsd']['epoch'] = data['obsd']['epoch'][fe - 1:, :]
        data['obsd']['st'] = data['obsd']['st'][fe - 1:, :]

    return data
def velo(t, y, gap):
    vel = (((t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (-362880)) * y[0] + \
          (((t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (40320)) * y[1] + \
          (((t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (-10080)) * y[2] + \
          (((t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (4320)) * y[3] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (-2880)) * y[4] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 7) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9)) / (2880)) * y[5] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 8) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9)) / (-4320)) * y[6] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 9) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9)) / (10080)) * y[7] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 10) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8)) / (-40320)) * y[8] + \
          (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) +
            (t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8)) / (362880)) * y[9]

    vel = vel / gap
    return vel


# Lagrange interpolation, using ten points to fit polynomial curve
def lagrange(t, y):
    pos = (((t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (-362880)) * y[
        0] + \
          (((t - 1) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (40320)) * y[
              1] + \
          (((t - 1) * (t - 2) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (-10080)) * y[
              2] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (4320)) * y[3] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 6) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (-2880)) * y[
              4] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 7) * (t - 8) * (t - 9) * (t - 10)) / (2880)) * y[5] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 8) * (t - 9) * (t - 10)) / (-4320)) * y[
              6] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 9) * (t - 10)) / (10080)) * y[
              7] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 10)) / (-40320)) * y[
              8] + \
          (((t - 1) * (t - 2) * (t - 3) * (t - 4) * (t - 5) * (t - 6) * (t - 7) * (t - 8) * (t - 9)) / (362880)) * y[9]

    return pos


def entrp(nep, gap, dat):
    """
    改进版插值函数

    参数:
    nep (float): 插值时刻
    gap (float): 时间间隔
    dat (array): 数据数组

    返回:
    tuple: (插值结果, 速度)
    """
    min_idx = 1
    max_idx = int(86400 / gap)
    n = (nep / gap) + 1

    # 检查数据维度
    if dat.ndim > 1:
        data_column = dat[:, 0]
    else:
        data_column = dat

    # 数据长度检查
    if len(data_column) < 10:
        print(f"数据长度不足: {len(data_column)}")
        return np.nan, np.nan

    # 处理起始边界情况
    if n < (min_idx + 5):
        # 如果太靠近数据开始
        kern = data_column[min_idx - 1:min_idx + 9]
        if len(kern) < 10:
            # 尝试使用多项式拟合
            print(f"起始边界数据不足: {len(kern)}")
            return np.nan, np.nan

        nt = n - (min_idx - 1)  # 调整索引位置
        out1 = lagrange(nt, kern)
        out2 = velo(nt, kern, gap)

    # 处理结束边界情况
    elif n > (max_idx - 5):
        # 如果太靠近数据结束
        if max_idx <= 10:
            # 数据总量不足
            print("数据总量不足 10 个点")
            return np.nan, np.nan

        kern = data_column[max_idx - 10:max_idx]
        if len(kern) < 10:
            print(f"结束边界数据不足: {len(kern)}")
            return np.nan, np.nan

        nt = n - (max_idx - 10)  # 调整索引位置
        out1 = lagrange(nt, kern)
        out2 = velo(nt, kern, gap)

    # 处理正常情况
    else:
        st = int(np.floor(n)) - 5
        fn = int(np.ceil(n)) + 4

        # 边界调整
        st = max(0, st)
        fn = min(len(data_column), fn)

        # 数据检查
        if fn - st < 10:
            print(f"中间区域数据不足: {fn - st}")
            return np.nan, np.nan

        kern = data_column[st:fn]
        nt = n - st  # 调整索引位置

        try:
            out1 = lagrange(nt, kern)
            out2 = velo(nt, kern, gap)
        except Exception as e:
            print(f"插值计算异常: {str(e)}")
            return np.nan, np.nan

    return out1, out2


def entrp_orbt(nep, gap, dat):
    """
    改进版轨道插值函数

    参数:
    nep (float): 插值时刻
    gap (float): 时间间隔
    dat (array): 数据数组

    返回:
    tuple: (插值结果, 速度)
    """
    n = (nep / gap) + 6

    # 确保数据维度正确
    if dat.ndim > 1:
        data_column = dat[:, 0]
    else:
        data_column = dat

    # 计算数据范围
    st = int(np.floor(n)) - 5
    fn = int(np.ceil(n)) + 5

    # 边界调整
    st = max(0, st)
    fn = min(len(data_column), fn)

    # 数据检查
    if fn - st < 10:
        # 数据不足10个点
        print(f"轨道数据不足: st={st}, fn={fn}, 长度={fn - st}")
        return np.nan, np.nan

    kern = data_column[st:fn]
    nt = n - st  # 调整到局部坐标

    try:
        out1 = lagrange(nt, kern)
        out2 = velo(nt, kern, gap)
    except Exception as e:
        print(f"轨道插值异常: {str(e)}")
        return np.nan, np.nan

    return out1, out2
def arc_dtr(obs):
    """
    Detect continuous observation arcs in satellite data.

    Parameters:
    obs (dict): Observation data structure

    Returns:
    list: List of arrays containing start and end epoch indices for each continuous arc
    """
    sn = obs['st'].shape[1]  # Changed from obs.st to obs['st']
    arc = [None] * sn

    for k in range(sn):
        # Find indices where p1 is NaN and st is 1, then set st to 0
        dc = np.where((np.isnan(obs['p1'][:, k])) & (obs['st'][:, k] == 1))[0]  # Changed from obs.p1, obs.st
        if len(dc) > 0:
            obs['st'][dc, k] = 0  # Changed from obs.st

        # Find indices where st equals 1
        row = np.where(obs['st'][:, k] == 1)[0]  # Changed from obs.st

        if len(row) == 0:
            arc[k] = np.array([])
            continue

        # Find breaks in continuous observations
        brk = np.where(np.diff(row) != 1)[0]

        # Create arrays for start and end indices
        frs = np.concatenate(([0], brk + 1))
        lst = np.concatenate((brk, [len(row) - 1]))
        als = np.column_stack((frs, lst))

        # Remove arcs with less than 10 continuous observations
        if als.shape[0] > 1:
            i = als.shape[0] - 1
            while i >= 0:
                if (als[i, 1] - als[i, 0]) < 10:
                    als = np.delete(als, i, axis=0)
                i -= 1
        elif als.shape[0] == 1:
            if (als[0, 1] - als[0, 0]) < 10:
                als = np.array([])

        # Create output array with actual epoch indices
        if len(als) > 0:
            arn = np.full((als.shape[0], 2), np.nan)
            for i in range(als.shape[0]):
                arn[i, 0] = row[int(als[i, 0])]
                arn[i, 1] = row[int(als[i, 1])]
            arc[k] = arn
        else:
            arc[k] = np.array([])

    return arc
def rotation(position, angle, axis):
    """
    旋转坐标变换函数

    参数:
    position (array): 3D位置向量
    angle (float): 旋转角度(度)
    axis (int): 旋转轴(1=X, 2=Y, 3=Z)

    返回:
    array: 旋转后的位置向量
    """
    # 检查输入维度
    position = np.array(position).flatten()
    if len(position) != 3:
        raise ValueError('Matrix dimension should be 3xN or Nx3.')

    if not (np.isscalar(angle) and np.isscalar(axis)):
        raise ValueError('Angle and axis should be scalar.')

    # 创建旋转矩阵 - 注意使用角度制
    if axis == 1:  # X轴
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), np.sin(np.radians(angle))],
            [0, -np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
    elif axis == 2:  # Y轴
        rot = np.array([
            [np.cos(np.radians(angle)), 0, -np.sin(np.radians(angle))],
            [0, 1, 0],
            [np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ])
    elif axis == 3:  # Z轴
        rot = np.array([
            [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0],
            [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError('Axis must be 1, 2, or 3.')

    # 应用旋转
    xout = rot @ position

    return xout
def dtr_satno(obs):
    """
    确定每个历元可观测的卫星个数

    参数:
    obs (dict): 观测数据字典

    返回:
    numpy.ndarray: 每个历元可观测的卫星数量
    """
    m = obs['st'].shape[0]  # 历元数
    satno = np.zeros((m, 1))

    for i in range(m):
        satno[i, 0] = np.sum(obs['st'][i, :])

    return satno


def entrp_clkf(nep, gap, dat):
    """
    基于多项式拟合的钟差插值函数

    参数:
    nep (float): 需要插值的时刻
    gap (float): 时间间隔
    dat (numpy.ndarray): 数据数组

    返回:
    float: 插值结果
    """
    min_idx = 1
    max_idx = int(86400 / gap)
    n = (nep / gap) + 1
    ns = int((10 * 60) / gap)  # 10分钟的点数
    nt = int((20 * 60) / gap)  # 20分钟的点数

    if (n - ns) <= min_idx:
        st = min_idx
        fn = min_idx + (nt - 1)
        t = np.arange(st, fn + 1)
        kern = dat[st - 1:fn, 0] if dat.ndim > 1 else dat[st - 1:fn]
        # 使用2阶多项式拟合
        p = np.polyfit(t, kern, 2)
        out1 = np.polyval(p, n)
    elif (n + ns) > max_idx:
        st = max_idx - (nt - 1)
        fn = max_idx
        t = np.arange(st, fn + 1)
        kern = dat[st - 1:fn, 0] if dat.ndim > 1 else dat[st - 1:fn]
        p = np.polyfit(t, kern, 2)
        out1 = np.polyval(p, n)
    else:
        st = int(np.floor(n)) - (ns - 1)
        fn = int(np.ceil(n)) + (ns - 1)
        t = np.arange(st, fn + 1)
        kern = dat[st - 1:fn, 0] if dat.ndim > 1 else dat[st - 1:fn]
        p = np.polyfit(t, kern, 2)
        out1 = np.polyval(p, n)

    return out1
def moon(mjd):
    """
    计算地月矢量（月球相对地心位置）

    参数:
    mjd (float): 约化儒略日

    返回:
    numpy.ndarray: 月球在地固系下的位置向量（单位：m）
    """
    T = (mjd - 51544.5) / 36525
    L0 = (218.31617 + 481267.88088 * T - 1.3972 * T) % 360  # 度
    l = (134.96292 + 477198.86753 * T) % 360  # 度
    lp = (357.52543 + 35999.04944 * T) % 360  # 度
    F = (93.27283 + 483202.01873 * T) % 360  # 度
    D = (297.85027 + 445267.11135 * T) % 360  # 度
    obl = 23.43929111  # 度

    # 计算月球位置
    long = (L0 + (22640 * np.sin(np.radians(l)) + 769 * np.sin(np.radians(2 * l))
                  - 4586 * np.sin(np.radians(l - 2 * D)) + 2370 * np.sin(np.radians(2 * D))
                  - 668 * np.sin(np.radians(lp)) - 412 * np.sin(np.radians(2 * F))
                  - 212 * np.sin(np.radians(2 * l - 2 * D)) - 206 * np.sin(np.radians(l + lp - 2 * D))
                  + 192 * np.sin(np.radians(l + 2 * D)) - 165 * np.sin(np.radians(lp - 2 * D))
                  + 148 * np.sin(np.radians(l - lp)) - 125 * np.sin(np.radians(D))
                  - 110 * np.sin(np.radians(l + lp)) - 55 * np.sin(np.radians(2 * F - 2 * D))) / 3600) % 360

    lat = ((18520 * np.sin(
        np.radians(F + long - L0 + (412 * np.sin(np.radians(2 * F)) + 541 * np.sin(np.radians(lp))) / 3600))
            - 526 * np.sin(np.radians(F - 2 * D)) + 44 * np.sin(np.radians(l + F - 2 * D))
            - 31 * np.sin(np.radians(-l + F - 2 * D)) - 25 * np.sin(np.radians(-2 * l + F))
            - 23 * np.sin(np.radians(lp + F - 2 * D)) + 21 * np.sin(np.radians(-l + F))
            + 11 * np.sin(np.radians(-lp + F - 2 * D))) / 3600) % 360

    dist = (385000 - 20905 * np.cos(np.radians(l)) - 3699 * np.cos(np.radians(2 * D - l))
            - 2956 * np.cos(np.radians(2 * D)) - 570 * np.cos(np.radians(2 * l)) + 246 * np.cos(
                np.radians(2 * l - 2 * D))
            - 205 * np.cos(np.radians(lp - 2 * D)) - 171 * np.cos(np.radians(l + 2 * D))
            - 152 * np.cos(np.radians(l + lp - 2 * D)))  # km

    # 计算月球在黄道坐标系下的位置
    m_pos = np.array([
        dist * np.cos(np.radians(long)) * np.cos(np.radians(lat)),
        dist * np.sin(np.radians(long)) * np.cos(np.radians(lat)),
        dist * np.sin(np.radians(lat))
    ])

    # 转换到赤道坐标系
    m_pos = rotation(m_pos, -obl, 1)

    # 转换到地固坐标系（ECI to ECEF）
    fday = mjd - np.floor(mjd)
    JDN = mjd - 15019.5
    gstr = (279.690983 + 0.9856473354 * JDN + 360 * fday + 180) % 360
    m_pos = rotation(m_pos, gstr, 3)

    # 转换单位从km到m
    m_pos = m_pos * 1000

    return m_pos
def sun(mjd):
    """
    计算太阳位置相对于地心

    参数:
    mjd (float): 约化儒略日

    返回:
    numpy.ndarray: 太阳在地固系的位置向量（单位：m）
    """
    AU = 149597870700  # 天文单位，单位：米
    d2r = np.pi / 180  # 度转弧度

    fday = mjd - np.floor(mjd)  # 一天中的小数部分
    JDN = mjd - 15019.5

    # 太阳位置计算参数
    v1 = (279.696678 + 0.9856473354 * JDN) % 360  # 度
    gstr = (279.690983 + 0.9856473354 * JDN + 360 * fday + 180) % 360  # 度
    g = np.radians((358.475845 + 0.9856002670 * JDN) % 360)  # 弧度

    # 计算太阳黄经
    slong = v1 + (1.91946 - 0.004789 * JDN / 36525) * np.sin(g) + 0.020094 * np.sin(2 * g)  # 度
    obliq = np.radians(23.45229 - 0.0130125 * JDN / 36525)  # 黄道倾角，弧度

    # 计算太阳位置
    slp = np.radians(slong - 0.005686)
    snd = np.sin(obliq) * np.sin(slp)
    csd = np.sqrt(1 - snd ** 2)
    sdec = np.degrees(np.arctan2(snd, csd))  # 太阳赤纬，度

    sra = 180 - np.degrees(np.arctan2((snd / csd / np.tan(obliq)), (-np.cos(slp) / csd)))  # 太阳赤经，度

    # 太阳在赤道坐标系中的位置（单位：米）
    s_pos = np.array([
        np.cos(np.radians(sdec)) * np.cos(np.radians(sra)) * AU,
        np.cos(np.radians(sdec)) * np.sin(np.radians(sra)) * AU,
        np.sin(np.radians(sdec)) * AU
    ])

    # 转换到地固系
    s_pos = rotation(s_pos, gstr, 3)

    return s_pos
def dtr_satlist(obs):
    """
    创建每个历元可见卫星的列表

    参数:
    obs (dict): 观测数据字典，包含st字段

    返回:
    list: 每个历元可见卫星的索引列表
    """
    m = obs['st'].shape[0]  # 历元数
    satlist = [[] for _ in range(m)]

    for i in range(m):
        # 找到所有st值为1的卫星索引
        satlist[i] = np.where(obs['st'][i, :] == 1)[0] + 1

    return satlist


def dtr_sys(options):
    """
    根据启用的卫星系统确定系统参数

    参数:
    options (dict): 计算选项字典

    返回:
    tuple: (bp, ap, sit) 基本参数数量，总参数数量，卫星系统组合情况代码
    """
    # 对流层梯度选择
    if options['TroGrad'] == 0:
        base = 5
    elif options['TroGrad'] == 1:
        base = 7

    if options['system']['gps'] == 1:
        if options['system']['glo'] == 1:
            if options['system']['gal'] == 1:
                if options['system']['bds'] == 1:
                    bp = base + 3
                    ap = 92
                    sit = 12
                else:
                    bp = base + 2
                    ap = 78
                    sit = 8
            elif options['system']['bds'] == 1:
                bp = base + 2
                ap = 72
                sit = 9
            else:
                bp = base + 1
                ap = 58
                sit = 3
        elif options['system']['gal'] == 1:
            if options['system']['bds'] == 1:
                bp = base + 2
                ap = 66
                sit = 10
            else:
                bp = base + 1
                ap = 52
                sit = 4
        elif options['system']['bds'] == 1:
            bp = base + 1
            ap = 46
            sit = 5
        else:
            bp = base
            ap = 32
            sit = 1
    elif options['system']['glo'] == 1:
        if options['system']['gal'] == 1:
            if options['system']['bds'] == 1:
                bp = base + 2
                ap = 60
                sit = 11
            else:
                bp = base + 1
                ap = 46
                sit = 6
        elif options['system']['bds'] == 1:
            bp = base + 1
            ap = 40
            sit = 7
        else:
            bp = base
            ap = 26
            sit = 2
    else:
        raise ValueError("Process must include GPS or GLONASS satellites")

    return bp, ap, sit
def save_visualization_data(xs, n, e, u, thrD, rms, CT):
    """
    将GNSS定位结果保存为可视化所需的JSON格式

    参数:
    xs (numpy.ndarray): 定位结果
    n, e, u (numpy.ndarray): 北向、东向、高程误差
    thrD (numpy.ndarray): 三维距离误差
    rms (numpy.ndarray): 各方向RMS误差
    CT (int): 收敛时间
    """
    # 创建数据列表
    viz_data = []

    # 遍历每个历元
    for i in range(thrD.shape[0]):
        epoch_data = {
            "epoch": i + 1,
            "thrD": float(thrD[i]),
            "x": float(xs[0, i]),
            "y": float(xs[1, i]),
            "z": float(xs[2, i]),
            "n": float(n[i]),
            "e": float(e[i]),
            "u": float(u[i])
        }
        viz_data.append(epoch_data)

    # 添加元数据
    metadata = {
        "convergenceTime": int(CT),
        "rmsN": float(rms[0, 0]),
        "rmsE": float(rms[1, 0]),
        "rmsU": float(rms[2, 0]),
        "rms3D": float(np.sqrt(rms[0, 0] ** 2 + rms[1, 0] ** 2 + rms[2, 0] ** 2)),
        "referencePosition": {
            "x": float(np.mean(xs[0, :])),
            "y": float(np.mean(xs[1, :])),
            "z": float(np.mean(xs[2, :]))
        }
    }

    # 创建完整数据对象
    full_data = {
        "metadata": metadata,
        "epochs": viz_data
    }

    # 保存为JSON文件
    with open('./result/gnss_visualization_data.json', 'w') as f:
        json.dump(full_data, f)


