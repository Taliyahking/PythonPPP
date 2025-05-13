import numpy as np
from atmo_correct import trop_gmf,ngpt2
from helper_function import frequencies,cal2jul,rotation,xyz2plh,sun,moon,dtr_satno


def nmodel(data, options):
    """
    核心导航模型函数

    参数:
    data (dict): 包含观测数据和卫星信息的字典
    options (dict): 处理选项

    返回:
    numpy.ndarray: 导航模型结果
    """
    c = 299792458  # 光速 m/s

    freq, wavl = frequencies()  # 假设您已经在前面的代码中定义了这个函数

    # 创建模型选项数组，表示要使用的改正项
    mopt = np.zeros(11)
    mopt[0] = options.get('SatClk', 0)  # 卫星钟差改正
    mopt[1] = options.get('SatAPC', 0)  # 卫星天线相位中心改正
    mopt[2] = options.get('RecAPC', 0)  # 接收机天线相位中心改正
    mopt[3] = options.get('RecARP', 0)  # 接收机天线参考点改正
    mopt[4] = options.get('RelClk', 0)  # 相对论钟差改正
    mopt[5] = options.get('SatWind', 0)  # 卫星天线相位缠绕改正
    mopt[6] = options.get('AtmTrop', 0)  # 对流层改正
    mopt[7] = options.get('Iono', 0)  # 电离层改正
    mopt[8] = options.get('RelPath', 0)  # 相对论路径改正
    mopt[10] = options.get('Solid', 0)  # 固体潮汐改正

    # 获取每个历元可观测的卫星数
    satno = dtr_satno(data['obsd'])
    n_sats = np.sum(satno)
    n_model = int(n_sats * 4)  # 每颗卫星有4个观测值：P1, P2, L1, L2
    model = np.zeros((n_model, 31))

    # 确定时间系统偏差
    if data['obsh']['time'].get('system') == 'GPS' or not data['obsh']['time'].get('system'):
        dt = 51.184
    else:
        dt = 32.184 + data['obsh']['time'].get('leap', 0)

    # 接收机位置
    r_xyz = np.array(data['obsh']['sta']['pos']).flatten()  # 确保变成一维数组
    if len(r_xyz) != 3:
        raise ValueError(f"Expected r_xyz to have 3 components, got {len(r_xyz)}")

    # 计算接收机地理坐标
    ellp = xyz2plh(r_xyz, 0)
    dlat = ellp[0]
    dlon = ellp[1]
    hell = ellp[2]

    # 计算约化儒略日期
    year = data['obsh']['time']['first'][0]
    mon = data['obsh']['time']['first'][1]
    day = data['obsh']['time']['first'][2]
    sec = 0
    _, dmjd = cal2jul(year, mon, day, sec)

    # 使用数值天气模型获取压力
    p = ngpt2(dmjd, dlat, dlon, hell, 1, 1)
    if isinstance(p, tuple):
        p = p[0]  # 获取返回元组的第一个元素

    # 天线参考点偏移量
    arp = np.array(data['obsh']['sta']['antdel'])
    if arp.shape[0] != 1:
        arp = arp.reshape(1, -1)

    # 接收机相位中心偏移量
    r_apc = data['atx']['rcv']['pco']

    t = 0  # 模型记录计数器

    # 对每个历元进行处理
    for i in range(data['obsd']['st'].shape[0]):
        year = data['obsh']['time']['first'][0]
        doy = data['obsh']['time']['doy']
        secod = data['obsd']['epoch'][i, 0]

        # 计算TT和UTC的约化儒略日
        _, mjd_tt = cal2jul(year, mon, day, secod + dt)
        _, mjd = cal2jul(year, mon, day, secod)

        # 计算太阳和月球位置
        sun_xyz = sun(mjd_tt)
        mon_xyz = moon(mjd_tt)

        # 获取当前历元可观测的卫星
        sats = np.where(data['obsd']['st'][i, :] == 1)[0]

        # 对每颗卫星进行处理
        for k in sats:
            # 跳过非GPS/GLONASS卫星
            if k >= 59:  # Galileo或其他系统
                continue

            # 检查卫星数据是否包含NaN
            try:
                s_xyz = data['psat'][i, 0:3, k]
                v_xyz = data['psat'][i, 3:6, k]
                s_apc = data['atx']['sat']['pco'][k, :, :]

                if np.any(np.isnan(s_xyz)) or np.any(np.isnan(v_xyz)):
                    continue
            except Exception:
                continue

            # 检查观测值是否都可用
            if (np.isnan(data['obsd']['p1'][i, k]) or
                    np.isnan(data['obsd']['p2'][i, k]) or
                    np.isnan(data['obsd']['l1'][i, k]) or
                    np.isnan(data['obsd']['l2'][i, k])):
                continue

            # 对每种观测量类型进行处理 (P1, P2, L1, L2)
            for u in range(4):
                t += 1  # 增加计数器

                # 记录基本信息
                model[t - 1, 0] = year
                model[t - 1, 1] = doy
                model[t - 1, 2] = secod
                model[t - 1, 3] = k + 1
                model[t - 1, 4] = data['tofs'][i, k]

                # 根据观测类型处理
                if u == 0:  # P1
                    model[t - 1, 5] = data['obsd']['p1'][i, k]

                    # 卫星天线相位中心改正
                    sdt = -sat_apc(s_xyz, r_xyz, sun_xyz, s_apc, 1, k)
                    if not np.isnan(sdt):
                        model[t - 1, 15] = sdt

                    # 接收机天线相位中心改正
                    if k > 32 and k < 59:  # GLONASS
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 3)  # GLO-L1
                    else:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 1)  # GPS-L1

                elif u == 1:  # P2
                    model[t - 1, 5] = data['obsd']['p2'][i, k]

                    sdt = -sat_apc(s_xyz, r_xyz, sun_xyz, s_apc, 2, k)
                    if not np.isnan(sdt):
                        model[t - 1, 15] = sdt

                    if k > 32 and k < 59:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 4)  # GLO-L2
                    else:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 2)  # GPS-L2

                elif u == 2:  # L1
                    model[t - 1, 5] = data['obsd']['l1'][i, k]

                    sdt = -sat_apc(s_xyz, r_xyz, sun_xyz, s_apc, 1, k)
                    if not np.isnan(sdt):
                        model[t - 1, 15] = sdt

                    if k > 32 and k < 59:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 3)
                    else:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 1)

                    # 天线相位缠绕改正
                    if i == 0:
                        prev = 0
                    else:
                        prev = model[t - 5, 19]  # 前一个历元相位缠绕改正值

                    model[t - 1, 19] = (wind_up(r_xyz, s_xyz, sun_xyz, prev) / (2 * np.pi)) * wavl[k, 0]

                elif u == 3:  # L2
                    model[t - 1, 5] = data['obsd']['l2'][i, k]

                    sdt = -sat_apc(s_xyz, r_xyz, sun_xyz, s_apc, 2, k)
                    if not np.isnan(sdt):
                        model[t - 1, 15] = sdt

                    if k > 32 and k < 59:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 4)
                    else:
                        model[t - 1, 16] = rec_apc(s_xyz, r_xyz, r_apc, 2)

                    if i == 0:
                        prev = 0
                    else:
                        prev = model[t - 5, 19]

                    model[t - 1, 19] = (wind_up(r_xyz, s_xyz, sun_xyz, prev) / (2 * np.pi)) * wavl[k, 1]

                # 记录卫星位置和速度
                model[t - 1, 7:10] = s_xyz
                model[t - 1, 10:13] = v_xyz

                # 计算距离
                model[t - 1, 13] = np.linalg.norm(s_xyz - r_xyz)

                # 卫星钟差改正
                model[t - 1, 14] = -(data['psat'][i, 6, k] * c)

                # 接收机天线参考点改正
                model[t - 1, 17] = rec_arp(s_xyz, r_xyz, arp)

                # 相对论钟差改正
                model[t - 1, 18] = -(rel_clk(s_xyz, v_xyz))

                # 对流层改正（Saastamoinen模型）
                Trop, Mwet, Mn, Me, ZHD = trop_gmf(r_xyz, s_xyz, mjd, p)
                model[t - 1, 20] = Trop
                model[t - 1, 27] = Mwet
                model[t - 1, 28] = Mn
                model[t - 1, 29] = Me
                model[t - 1, 30] = ZHD

                # 相对论路径改正
                model[t - 1, 22] = rpath(r_xyz, s_xyz)

                # 固体潮汐改正
                model[t - 1, 24] = solid(r_xyz, s_xyz, sun_xyz, mon_xyz)

                # 记录卫星高度角和方位角
                model[t - 1, 25] = data['obsd']['elv'][i, k]
                model[t - 1, 26] = data['obsd']['azm'][i, k]

                # 汇总所有改正项，生成总改正值
                full = model[t - 1, 14:24]  # 这是10个元素

                # 应用选定的改正项
                mopt_used = np.zeros(10)
                mopt_used[:9] = mopt[:9]  # 复制前9个元素
                mopt_used[9] = mopt[10]  # 将索引10的值放在索引9的位置

                model[t - 1, 6] = np.sum(full[mopt_used == 1])

    return model

def rec_apc(s_xyz, r_xyz, r_apc, opt):
    """
    接收机天线相位中心改正

    参数：
    s_xyz - 卫星坐标
    r_xyz - 接收机坐标
    r_apc - 天线相位中心
    opt - 频率选项: 1=GPS-L1, 2=GPS-L2, 3=GLO-L1, 4=GLO-L2

    返回：
    rapc - 天线相位中心改正值
    """
    # 计算视线向量
    l = r_xyz - s_xyz
    los = l / np.linalg.norm(l)

    # 计算接收机地理位置
    ellp = xyz2plh(r_xyz, 0)
    lat = ellp[0]
    lon = ellp[1]

    # 定义转换矩阵（站心坐标系到地固坐标系）
    ori = np.array([
        [-np.sin(lon), -np.cos(lon) * np.sin(lat), np.cos(lon) * np.cos(lat)],
        [np.cos(lon), -np.sin(lon) * np.sin(lat), np.sin(lon) * np.cos(lat)],
        [0, np.cos(lat), np.sin(lat)]
    ])

    # 获取对应频率的天线相位中心值
    f = r_apc[:, :, opt - 1]

    c = np.array([
        f[0, 1],  # 北向分量
        f[0, 0],  # 东向分量
        f[0, 2]  # 高度分量
    ])

    # 转换到地固坐标系
    p = ori @ c

    # 维度检查
    if p.shape[0] != los.shape[0]:
        los = los.T

    # 计算点积得到改正值
    rapc = np.dot(p, los)

    return rapc


def rec_arp(s_xyz, r_xyz, arp):
    """
    接收机天线参考点改正
    """
    # 视线向量计算
    l = r_xyz - s_xyz
    los = l / np.linalg.norm(l)

    # 计算站心坐标系转换
    ellp = xyz2plh(r_xyz, 0)
    lat = ellp[0]
    lon = ellp[1]

    # 转换矩阵
    ori = np.array([
        [-np.sin(lon), -np.cos(lon) * np.sin(lat), np.cos(lon) * np.cos(lat)],
        [np.cos(lon), -np.sin(lon) * np.sin(lat), np.sin(lon) * np.cos(lat)],
        [0, np.cos(lat), np.sin(lat)]
    ])

    if arp.ndim == 1:
        enu = np.array([arp[1], arp[2], arp[0]])
    elif arp.shape == (1, 3):
        enu = np.array([arp[0, 1], arp[0, 2], arp[0, 0]])
    else:
        enu = np.array([0.0, 0.0, 0.0])

    p = ori @ enu

    # 维度检查
    if p.shape[0] != los.shape[0]:
        los = los.T

    # 计算改正值
    rarp = np.dot(p, los)

    return rarp


def rel_clk(s_xyz, v_xyz):
    """
    相对论钟差改正

    参数:
    s_xyz (numpy.ndarray): 卫星位置
    v_xyz (numpy.ndarray): 卫星速度

    返回:
    float: 相对论钟差改正值
    """
    c = 299792458  # m/s

    # -2*(r·v)/c^2 * c
    rclk = -2 * (np.dot(s_xyz, v_xyz) / (c ** 2)) * c

    return rclk


def rpath(r_xyz, s_xyz):
    """
    相对论路径改正

    参数:
    r_xyz (numpy.ndarray): 接收机位置
    s_xyz (numpy.ndarray): 卫星位置

    返回:
    float: 路径改正值
    """
    # WGS84 参数
    mu = 3986004.418e8  # m^3/s^2
    c = 2.99792458e8  # m/s

    rsat = np.linalg.norm(s_xyz)
    rrec = np.linalg.norm(r_xyz)
    rs = s_xyz - r_xyz
    rrs = np.linalg.norm(rs)

    # 计算相对论路径延迟
    rpath = (2 * mu) / (c ** 2) * np.log((rsat + rrec + rrs) / (rsat + rrec - rrs))

    return rpath


def sat_apc(s_xyz, r_xyz, sun_xyz, s_apc, opt, sno):
    """
    卫星天线相位中心改正

    参数:
    s_xyz (numpy.ndarray): 卫星位置
    r_xyz (numpy.ndarray): 接收机位置
    sun_xyz (numpy.ndarray): 太阳位置
    s_apc (numpy.ndarray): 卫星天线相位中心偏移量
    opt (int): 频率选项 (1=L1, 2=L2)
    sno (int): 卫星编号

    返回:
    float: 改正值
    """
    # 计算视线向量
    l = r_xyz - s_xyz
    los = l / np.linalg.norm(l)

    # 卫星固定坐标系定义
    k = -s_xyz / np.linalg.norm(s_xyz)  # 指向地球
    rs = sun_xyz - s_xyz
    e = rs / np.linalg.norm(rs)  # 指向太阳
    j = np.cross(k, e)
    i = np.cross(j, k)

    # 卫星坐标系转换矩阵
    sf = np.vstack((i, j, k))

    # 检查s_apc的维度并相应地处理
    if s_apc.ndim == 3:
        # 如果是3D数组，按原来的方式处理
        de1 = s_apc[:, :, 0].T
        de2 = s_apc[:, :, 1].T
    elif s_apc.ndim == 2:
        # 如果是2D数组，假设第一列是L1，第二列是L2
        # 这里需要根据实际数据结构进行调整
        de1 = s_apc[:, 0].reshape(-1)
        de2 = s_apc[:, 1].reshape(-1)
    else:
        # 如果是其他维度，使用默认值
        de1 = np.array([0.0, 0.0, 0.0])
        de2 = np.array([0.0, 0.0, 0.0])

    # 对于没有定义的Galileo和BeiDou卫星使用常规值
    if (np.any(np.isnan(de1)) or np.any(np.isnan(de2))) and (sno > 58 and sno < 89):  # GALILEO
        de1 = np.array([0.2, 0.0, 0.6])
        de2 = np.array([0.2, 0.0, 0.6])
    elif (np.any(np.isnan(de1)) or np.any(np.isnan(de2))) and (sno > 88 and sno < 93):  # BEIDOU
        de1 = np.array([0.6, 0.0, 1.1])
        de2 = np.array([0.6, 0.0, 1.1])

    # 选择对应频率的偏移量并转换到地固坐标系
    if opt == 1:
        rk = np.linalg.solve(sf, de1)
    elif opt == 2:
        rk = np.linalg.solve(sf, de2)

    # 计算点积获取改正值
    sapc = np.dot(rk, los)

    return sapc


def solid(r_xyz, s_xyz, sun_xyz, mon_xyz):
    """
    固体潮汐改正
    参数:
    r_xyz (numpy.ndarray): 接收机位置
    s_xyz (numpy.ndarray): 卫星位置
    sun_xyz (numpy.ndarray): 太阳位置
    mon_xyz (numpy.ndarray): 月球位置

    返回:
    float: 固体潮汐改正值
    """
    # 基本参数
    h0, h2, h3 = 0.6078, -0.0006, 0.292
    l0, l2, l3 = 0.0847, 0.0002, 0.015

    MS2E = 332946.0  # 太阳/地球质量比
    MM2E = 0.01230002  # 月球/地球质量比
    re = 6378137  # 地球半径

    # 计算视线向量
    l = r_xyz - s_xyz
    los = l / np.linalg.norm(l)

    # 接收机地理坐标
    ellp = xyz2plh(r_xyz, 1)  # 使用度数
    lat = np.radians(ellp[0])  # 转回弧度

    # 计算潮汐改正参数
    trm = 3 * np.sin(lat) ** 2 - 1
    h = h0 + h2 * trm
    l_val = l0 + l2 * trm

    # 各天体距离
    sunDist = np.linalg.norm(sun_xyz)
    moonDist = np.linalg.norm(mon_xyz)
    recDist = np.linalg.norm(r_xyz)

    # 单位向量
    sunUni = sun_xyz / sunDist
    moonUni = mon_xyz / moonDist
    recUni = r_xyz / recDist

    # 点积
    dotSR = np.dot(sunUni, recUni)
    dotMR = np.dot(moonUni, recUni)

    # 水平分量
    aSun = sunUni - (dotSR * recUni)
    aMoon = moonUni - (dotMR * recUni)

    # 位移因子
    DRS = (re ** 4) / (sunDist ** 3)
    DRM = (re ** 4) / (moonDist ** 3)
    DRM2 = (re ** 5) / (moonDist ** 4)

    # 太阳引起的位移
    s1 = ((3 * dotSR ** 2) - 1) / 2
    d2s = (MS2E * DRS) * (((h * recUni) * s1) + ((3 * l_val * dotSR) * aSun))

    # 月球引起的位移
    s2 = ((3 * dotMR ** 2) - 1) / 2
    d2m = (MM2E * DRM) * (((h * recUni) * s2) + ((3 * l_val * dotMR) * aMoon))

    # 月球高阶项
    s3 = ((5 / 2) * dotMR ** 3) - ((3 / 2) * dotMR)
    s4 = ((15 / 2) * dotMR ** 2) - (3 / 2)
    d3m = (MM2E * DRM2) * (((h3 * recUni) * s3) + ((l3 * s4) * aMoon))

    # 总位移
    stide = d2s + d2m + d3m

    # 改正值（在视线方向上的投影）
    stide_corr = np.dot(stide, los)

    return stide_corr

def wind_up(rec, sat, sun, prev):
    """
    卫星天线相位缠绕改正

    参数:
    rec (numpy.ndarray): 接收机位置
    sat (numpy.ndarray): 卫星位置
    sun (numpy.ndarray): 太阳位置
    prev (float): 上一历元的相位缠绕值

    返回:
    float: 相位缠绕改正值
    """
    # 太阳到卫星单位向量
    esun = sun - sat
    esun = esun / np.linalg.norm(esun)

    # 卫星坐标系
    ez = -sat
    ez = ez / np.linalg.norm(ez)  # Z轴指向地心
    ey = np.cross(ez, esun)  # Y轴垂直于卫星-地心和卫星-太阳平面
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, ez)  # 完成右手系
    ex = ex / np.linalg.norm(ex)

    xs = ex  # 卫星X轴
    ys = ey  # 卫星Y轴

    # 接收机地理坐标
    ellp = xyz2plh(rec, 0)
    phi = ellp[0]  # 纬度
    lam = ellp[1]  # 经度

    # 接收机坐标系（X指北，Y指西）
    xr = np.array([-np.sin(phi) * np.cos(lam), -np.sin(phi) * np.sin(lam), np.cos(phi)])
    yr = np.array([np.sin(lam), -np.cos(lam), 0])

    # 计算视线向量
    k = rec - sat
    k = k / np.linalg.norm(k)

    # 有效偶极子
    Ds = xs - k * np.dot(k, xs) - np.cross(k, ys)
    Dr = xr - k * np.dot(k, xr) + np.cross(k, yr)

    # 计算相位缠绕角
    wup = np.arccos(np.dot(Ds, Dr) / (np.linalg.norm(Ds) * np.linalg.norm(Dr)))

    # 确定符号
    if np.dot(k, np.cross(Ds, Dr)) < 0:
        wup = -wup

    # 增加整周跳变
    wup = (2 * np.pi * np.floor(((prev - wup) / (2 * np.pi)) + 0.5)) + wup

    return wup