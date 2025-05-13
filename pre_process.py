import numpy as np
from helper_function import frequencies, entrp, entrp_orbt, i_free, arc_dtr,local,rotation,lagrange,velo

def cal_sat(data):
    """
    计算卫星位置、速度和钟差

    参数:
    data (dict): 观测数据字典

    返回:
    dict: 更新后的数据结构
    """
    c = 299792458  # 光速 m/s
    we = 7.2921151467e-5  # 地球自转角速度 rad/s

    clkint = data['obsh']['time']['clkinterval']
    sp3int = data['obsh']['time']['sp3interval']
    rec = np.array(data['obsh']['sta']['pos']).flatten()

    en = data['obsd']['st'].shape[0]  # 历元数
    sn = data['obsd']['st'].shape[1]  # 卫星数
    psat = np.full((en, 7, sn), np.nan)  # 卫星位置、速度和钟差
    tofs = np.full((en, sn), np.nan)  # 信号传播时间

    for i in range(en):
        for k in range(sn):
            if data['obsd']['st'][i, k] == 1:
                try:
                    # 1. 计算飞行时间
                    if not np.isnan(data['obsd']['p1'][i, k]) and data['obsd']['p1'][i, k] != 0:
                        tof = data['obsd']['p1'][i, k] / c
                    elif not np.isnan(data['obsd']['p2'][i, k]) and data['obsd']['p2'][i, k] != 0:
                        tof = data['obsd']['p2'][i, k] / c
                    else:
                        data['obsd']['st'][i, k] = 0
                        continue

                    # 2. 计算卫星发射信号时间（接收时间减去传播时间）
                    nep = data['obsd']['epoch'][i, 0] - tof
                    d_clk = data['clk'][:, k]

                    # 3. 第一次卫星钟差插值
                    try:
                        # 使用多态方式调用插值函数，兼容不同的返回方式
                        dt_result = entrp(nep, clkint, d_clk)
                        if isinstance(dt_result, tuple):
                            dt = dt_result[0]  # 返回元组时取第一个值
                        else:
                            dt = dt_result  # 直接返回值时直接使用

                        if np.isnan(dt):
                            data['obsd']['st'][i, k] = 0
                            continue
                    except Exception:
                        data['obsd']['st'][i, k] = 0
                        continue

                    # 4. 更新发射时间和传播时间
                    nep = nep - dt
                    tofs[i, k] = tof + dt

                    # 5. 获取卫星轨道数据
                    d_x = data['sat']['sp3'][:, 0, k]
                    d_y = data['sat']['sp3'][:, 1, k]
                    d_z = data['sat']['sp3'][:, 2, k]

                    # 6. 插值获取卫星位置和速度
                    try:
                        if data['opt'].get('intrp', 0) == 1:
                            X_result = entrp_orbt(nep, sp3int, d_x)
                            Y_result = entrp_orbt(nep, sp3int, d_y)
                            Z_result = entrp_orbt(nep, sp3int, d_z)
                        else:
                            X_result = entrp(nep, sp3int, d_x)
                            Y_result = entrp(nep, sp3int, d_y)
                            Z_result = entrp(nep, sp3int, d_z)

                        # 处理多种返回类型
                        if isinstance(X_result, tuple):
                            X, VX = X_result
                            Y, VY = Y_result
                            Z, VZ = Z_result
                        else:
                            X = X_result
                            Y = Y_result
                            Z = Z_result
                            VX = VY = VZ = 0  # 如果没有返回速度，则设为0

                        if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
                            data['obsd']['st'][i, k] = 0
                            continue
                    except Exception:
                        data['obsd']['st'][i, k] = 0
                        continue

                    # 7. 再次插值获取卫星钟差
                    try:
                        dt_result = entrp(nep, clkint, d_clk)
                        if isinstance(dt_result, tuple):
                            dt = dt_result[0]
                        else:
                            dt = dt_result

                        if np.isnan(dt):
                            data['obsd']['st'][i, k] = 0
                            continue
                    except Exception:
                        data['obsd']['st'][i, k] = 0
                        continue

                    # 8. 组合卫星位置向量
                    R = np.array([X, Y, Z])

                    # 9. 计算地球自转改正角度
                    tf = np.linalg.norm(R - rec) / c  # 光传播时间
                    er_ang = np.degrees(tf * we)  # 地球自转角度

                    # 10. 应用地球自转改正
                    try:
                        pos = rotation(R, er_ang, 3)

                        # 确保结果为数组并且不含NaN
                        pos = np.array(pos).flatten()
                        if np.any(np.isnan(pos)) or len(pos) != 3:
                            data['obsd']['st'][i, k] = 0
                            continue
                    except Exception:
                        data['obsd']['st'][i, k] = 0
                        continue

                    # 11. 存储卫星位置、速度和钟差
                    psat[i, 0, k] = pos[0]
                    psat[i, 1, k] = pos[1]
                    psat[i, 2, k] = pos[2]
                    psat[i, 3, k] = VX
                    psat[i, 4, k] = VY
                    psat[i, 5, k] = VZ
                    psat[i, 6, k] = dt

                except Exception:
                    # 捕获所有其他异常
                    data['obsd']['st'][i, k] = 0

    # 更新数据结构
    data['psat'] = psat
    data['tofs'] = tofs

    return data


def elm_badclk(data):

    intr = data['obsh']['time']['clkinterval']

    for i in range(data['clk'].shape[1]):
        # Check for satellites with some clock data but containing errors
        if (np.any(~np.isnan(data['clk'][:, i])) and
                (np.any(np.isnan(data['clk'][:, i]) | (data['clk'][:, i] == 999999.999999)))):

            loc = np.where(np.isnan(data['clk'][:, i]))[0]
            for t in loc:
                sod = (t - 1) * intr
                ep = data['obsd']['epoch'][:, 0] == sod
                data['obsd']['st'][ep, i] = 0

        # Mark satellites with no clock data as invalid
        elif np.any(np.isnan(data['clk'][:, i])):
            data['obsd']['st'][:, i] = 0

    return data


def elv_mask(data, options):
    """
    Calculate satellite elevation and azimuth angles, and apply elevation mask.

    Parameters:
    data (dict): Data structure containing observations and satellite positions
    options (dict): Processing options including minimum elevation angle

    Returns:
    dict: Updated data structure with elevation and azimuth data
    """
    en = data['obsd']['st'].shape[0]
    sn = data['obsd']['st'].shape[1]

    r_xyz = data['obsh']['sta']['pos']
    elv = np.full((en, sn), np.nan)  # elevation angle
    azm = np.full((en, sn), np.nan)  # azimuth

    for i in range(en):
        for k in range(sn):
            if data['obsd']['st'][i, k] == 1:
                s_xyz = data['psat'][i, 0:3, k]

                # Convert ECEF coordinates to local ENU coordinates
                az, elev = local(r_xyz, s_xyz, 1)  # option=1 returns degrees
                elv[i, k] = elev
                azm[i, k] = az

                # Apply elevation mask
                if elev < options['elvangle']:
                    data['obsd']['st'][i, k] = 0

    data['obsd']['elv'] = elv
    data['obsd']['azm'] = azm

    return data


def clk_jmp2(data):
    """
    接收机钟跳变检测 (Receiver clock jump detection)

    Parameters:
    data (dict): Data structure containing observations

    Returns:
    dict: Updated data structure with clock jump corrections
    """
    arc = arc_dtr(data['obsd'])

    sn = data['obsd']['st'].shape[1]
    c = 299792458  # m/s

    pot = np.full((data['obsd']['st'].shape[0], sn), np.nan)
    jump = np.zeros((data['obsd']['st'].shape[0], 1))
    lmt = 1 - 50 * 10 ** -6  # millisecond w.r.t dif

    for k in range(sn):
        ark = arc[k]
        if ark.size == 0:
            continue

        for t in range(ark.shape[0]):
            st = int(ark[t, 0])
            fn = int(ark[t, 1])

            if k < 33:  # for GPS
                ifp = i_free(data['obsd']['p1'][st:fn + 1, k], data['obsd']['p2'][st:fn + 1, k], 0)
                ifl = i_free(data['obsd']['l1'][st:fn + 1, k], data['obsd']['l2'][st:fn + 1, k], 0)
                df = ifl - ifp
            elif k < 60:  # for GLONASS
                ifp = i_free(data['obsd']['p1'][st:fn + 1, k], data['obsd']['p2'][st:fn + 1, k], 1)
                ifl = i_free(data['obsd']['l1'][st:fn + 1, k], data['obsd']['l2'][st:fn + 1, k], 1)
                df = ifl - ifp
            elif k < 96:  # for GALILEO
                ifp = i_free(data['obsd']['p1'][st:fn + 1, k], data['obsd']['p2'][st:fn + 1, k], 2)
                ifl = i_free(data['obsd']['l1'][st:fn + 1, k], data['obsd']['l2'][st:fn + 1, k], 2)
                df = ifl - ifp
            elif k < 106:  # for BEIDOU
                ifp = i_free(data['obsd']['p1'][st:fn + 1, k], data['obsd']['p2'][st:fn + 1, k], 3)
                ifl = i_free(data['obsd']['l1'][st:fn + 1, k], data['obsd']['l2'][st:fn + 1, k], 3)
                df = ifl - ifp

            dfi = np.diff(df) / (10 ** -3 * c)
            mask = np.abs(dfi) > lmt
            if np.any(mask):
                m_indices = np.where(mask)[0]
                for i in range(len(m_indices)):
                    if np.abs(np.round(dfi[m_indices[i]]) - dfi[m_indices[i]]) < lmt:
                        pot[st + m_indices[i], k] = dfi[m_indices[i]]

    for i in range(pot.shape[0]):
        if np.any(~np.isnan(pot[i, :])):
            kern = np.where(~np.isnan(pot[i, :]))[0]
            M = np.nansum(pot[i, :]) / len(kern)
            k2 = 10 ** -5  # ms
            if np.abs(M - np.round(M)) <= k2:
                jump[i, 0] = np.round(M)

    kern2 = np.where(jump != 0)[0]
    for k in range(sn):
        ark = arc[k]
        if ark.size == 0:
            continue

        for t in range(ark.shape[0]):
            st = int(ark[t, 0])
            fn = int(ark[t, 1])
            for ke in kern2:
                if (ke > st) and (ke <= fn):
                    data['obsd']['l1'][ke:fn + 1, k] = data['obsd']['l1'][ke:fn + 1, k] + jump[ke, 0] * (c * 10 ** -3)
                    data['obsd']['l2'][ke:fn + 1, k] = data['obsd']['l2'][ke:fn + 1, k] + jump[ke, 0] * (c * 10 ** -3)

    data['jump'] = jump
    return data


def cs_detect(data, options):
    """
    周跳探测 (Cycle slip detection and repair)

    Parameters:
    data (dict): Data structure containing observations
    options (dict): Processing options

    Returns:
    dict: Updated data structure with cycle slip corrections
    """
    arc = arc_dtr(data['obsd'])

    sn = data['obsd']['st'].shape[1]
    c = 299792458  # m/s

    freq, wavl = frequencies()

    dt = data['obsh']['time']['obsinterval']
    sig0 = np.sqrt(2 * (0.0027 ** 2 + 0.0017 ** 2))  # meter
    dl = (0.4) * (dt / 3600)  # meter/hour

    for k in range(sn):
        if k >= len(freq):
            continue

        f1 = freq[k, 0]
        f2 = freq[k, 1]
        lamwl = c / (f1 - f2)

        # 载波宽巷组合(波长较长，更好解算模糊度)
        lwl = (data['obsd']['l1'][:, k] * f1 - data['obsd']['l2'][:, k] * f2) / (f1 - f2)
        # 伪距窄巷组合(波长变短，精度更高)
        pnl = (data['obsd']['p1'][:, k] * f1 + data['obsd']['p2'][:, k] * f2) / (f1 + f2)
        # MW模糊度/wide lane模糊度
        nwl = (lwl - pnl) / lamwl

        # geometry-free载波相位组合(无几何组合)
        gfl = data['obsd']['l1'][:, k] - data['obsd']['l2'][:, k]

        ark = arc[k]
        if ark.size == 0:
            continue

        for t in range(ark.shape[0]):
            st = int(ark[t, 0])
            fn = int(ark[t, 1])

            for i in range(st + 1, fn + 1):
                dmwc = 0
                dgfc = 0

                # MW detection
                if options.get('CSMw', 0) == 1:
                    if i - 2 < st:
                        mmw = np.mean(nwl[st:i])
                        smw = np.std(nwl[st:i + 1])
                    elif i - 30 < st:  # 设置其检测的历元最大间隔为30
                        mmw = np.mean(nwl[st:i])
                        smw = np.std(nwl[st:i])
                    else:
                        mmw = np.mean(nwl[i - 30:i])
                        smw = np.std(nwl[i - 30:i])

                    dmw = mmw - nwl[i]
                    if np.abs(dmw) > (5 * smw):
                        dmwc = 1

                # Geometry-free detection
                if options.get('CSGf', 0) == 1:
                    dgf = gfl[i - 1] - gfl[i]
                    elv = data['obsd']['elv'][i, k]
                    me = 1 + (10 * np.exp(-elv / 10))  # 这里不太能理解，貌似将高度角作为了阈值
                    smg = sig0 * me

                    if np.abs(dgf) > ((4 * smg) + dl):
                        dgfc = 1

                # Fix cycle slips
                if (dmwc == 1 and np.std(nwl[st:fn + 1]) > 0.6) or dgfc == 1:
                    one = nwl[i - 1] - nwl[i]
                    two = gfl[i - 1] - gfl[i]
                    A = np.array([[1, -1], [wavl[k, 0], -wavl[k, 1]]])
                    L = np.array([one, two])
                    Dn = np.linalg.pinv(A) @ L  # pinv:伪逆
                    Dn1 = round(Dn[0])
                    Dn2 = round(Dn[1])  # 舍入至最近的整数,求的是dN1,dN2

                    if Dn1 != 0 and Dn2 != 0:
                        data['obsd']['l1'][i:fn + 1, k] = data['obsd']['l1'][i:fn + 1, k] + Dn1 * wavl[
                            k, 0]  # 将后面的相位观测值加上这个周跳值
                        data['obsd']['l2'][i:fn + 1, k] = data['obsd']['l2'][i:fn + 1, k] + Dn2 * wavl[k, 1]

                    st = i

    return data

def outlier(data):
    """
    异常值处理 - 二阶多项式拟合周跳 (Detect and remove outliers using polynomial fitting)

    Parameters:
    data (dict): Data structure containing observations

    Returns:
    dict: Updated data structure with outliers removed
    """
    c = 299792458  # m/s
    freq, _ = frequencies()

    arc = arc_dtr(data['obsd'])

    sn = data['obsd']['st'].shape[1]

    for k in range(sn):
        if k >= len(freq):
            continue

        f1 = freq[k, 0]
        f2 = freq[k, 1]

        # MW combination
        lwl = (data['obsd']['l1'][:, k] * f1 - data['obsd']['l2'][:, k] * f2) / (f1 - f2)
        pnl = (data['obsd']['p1'][:, k] * f1 + data['obsd']['p2'][:, k] * f2) / (f1 + f2)
        lamwl = c / (f1 - f2)
        nwl = (lwl - pnl) / lamwl  # MW模糊度/wide lane模糊度

        ark = arc[k]
        if ark.size == 0:
            continue

        for n in range(ark.shape[0]):
            st = int(ark[n, 0])
            fn = int(ark[n, 1])

            while True:
                # Find valid indices
                t = np.where(data['obsd']['st'][:, k] == 1)[0]
                t = t[(t >= st) & (t <= fn)]

                if len(t) < 3:  # Need at least 3 points for quadratic fit
                    break

                L = nwl[t]
                ran = len(t)

                # Create design matrix for quadratic fit
                A = np.column_stack((t ** 2, t, np.ones(ran)))

                # Least squares solution
                X = np.linalg.lstsq(A, L, rcond=None)[0]

                # Residuals
                V = L - A @ X

                # Root mean square error
                rmse = np.sqrt(np.sum(V ** 2) / ran)  # root mean square error均方根差

                # Find outliers
                det = np.where(np.abs(V) > (4 * 0.6))[0]

                if rmse > 0.6 and len(det) > 0:
                    # 个人觉得这种处理方法不可靠，只是去掉了最大值，为了保证数据准确性，应将这个卫星去掉，或者循环去掉最大值直到符合阈值
                    data['obsd']['st'][t, k] = 0
                else:
                    break

    return data


def smoothing(data):
    """
    载波相位平滑伪距 - 利用精确、平滑的载波相位测量值来对粗糙但无模糊度的伪距进行不同程度的平滑
    最常见的是hatch filter

    Parameters:
    data (dict): Data structure containing observations

    Returns:
    dict: Updated data structure with smoothed pseudoranges
    """
    freq, _ = frequencies()

    arc = arc_dtr(data['obsd'])

    sn = data['obsd']['st'].shape[1]

    for k in range(sn):
        if k >= len(freq):
            continue

        f1 = freq[k, 0]
        f2 = freq[k, 1]

        ark = arc[k]
        if ark.size == 0:
            continue

        for t in range(ark.shape[0]):
            st = int(ark[t, 0])
            fn = int(ark[t, 1])

            # 计算均值，不包括nan值
            md1 = np.nanmean(data['obsd']['p1'][st:fn + 1, k] - data['obsd']['l1'][st:fn + 1, k])
            md2 = np.nanmean(data['obsd']['p2'][st:fn + 1, k] - data['obsd']['l2'][st:fn + 1, k])
            ml1 = np.nanmean(data['obsd']['l1'][st:fn + 1, k])
            ml2 = np.nanmean(data['obsd']['l2'][st:fn + 1, k])

            # Apply smoothing
            for i in range(st, fn + 1):
                data['obsd']['p1'][i, k] = data['obsd']['l1'][i, k] + md1 + \
                                           (2 * f2 ** 2 * (1 / (f1 ** 2 - f2 ** 2))) * (
                                                       (data['obsd']['l1'][i, k] - ml1) - (
                                                           data['obsd']['l2'][i, k] - ml2))
                data['obsd']['p2'][i, k] = data['obsd']['l2'][i, k] + md2 + \
                                           (2 * f1 ** 2 * (1 / (f1 ** 2 - f2 ** 2))) * (
                                                       (data['obsd']['l1'][i, k] - ml1) - (
                                                           data['obsd']['l2'][i, k] - ml2))

    return data





def entrp_orbtf(nep, gap, dat):
    """
    轨道插值函数

    Parameters:
    nep (float): Epoch time for interpolation
    gap (float): Time interval between data points
    dat (array): Data array to interpolate

    Returns:
    tuple: (interpolated value, velocity)
    """
    n = (nep / gap) + 6

    st = int(np.floor(n)) - 4
    fn = int(np.ceil(n)) + 4

    # Ensure indices are within valid range
    st = max(0, st)
    fn = min(len(dat), fn)

    if dat.ndim > 1:
        kern = dat[st:fn, 0]
    else:
        kern = dat[st:fn]

    if len(kern) < 10:
        return np.nan, np.nan

    nt = np.remainder(n, 1) + 5
    out1 = lagrange(nt, kern)  # Should be using lag instead of lagrange in the original code
    out2 = velo(nt, kern, gap)

    return out1, out2


def preprocess(data, options):
    print(f"Initial observations: {np.sum(data['obsd']['st'] == 1)}")

    data = elm_badclk(data)
    print(f"After elm_badclk: {np.sum(data['obsd']['st'] == 1)}")

    data = cal_sat(data)
    print(f"After cal_sat: {np.sum(data['obsd']['st'] == 1)}")

    data = elv_mask(data, options)
    print(f"After elv_mask: {np.sum(data['obsd']['st'] == 1)}")

    data = cs_detect(data, options)
    print(f"After cs_detect: {np.sum(data['obsd']['st'] == 1)}")

    if options.get('clkjump', 0) == 1:
        data = clk_jmp2(data)
        print(f"After clk_jmp2: {np.sum(data['obsd']['st'] == 1)}")

    data = outlier(data)
    print(f"After outlier: {np.sum(data['obsd']['st'] == 1)}")

    if options.get('codsmth', 0) == 1:
        data = smoothing(data)
        print(f"After smoothing: {np.sum(data['obsd']['st'] == 1)}")

    return data
