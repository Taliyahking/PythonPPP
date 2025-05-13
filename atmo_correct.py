import numpy as np
import math
from helper_function import xyz2plh,local
from scipy.io import loadmat


def ngpt2(dmjd, dlat, dlon, hell, it, opt):
    """
    NGPT (Next Generation Positioning Tool) – Numerical Weather Model (NWM)

    输入参数:
      dmjd : Modified Julian Date（标量）
      dlat : 站点纬度（弧度，标量）
      dlon : 站点经度（弧度，标量）
      hell : 站点高程（米，标量）
      it   : 时间指数（当 it==1 时不考虑年周期项）
      opt  : 选项（opt==1 只计算压力；opt==0 计算全部参数）

    返回值:
      p    : 大气压力（单位：hPa）
      T    : 温度（单位：°C，如果 opt==1 则为 np.nan）
      dT   : 温度梯度（单位：mK/m，如果 opt==1 则为 np.nan）
      e    : 水汽压（hPa，如果 opt==1 则为 np.nan）
      ah   : 干项改正参数（如果 opt==1 则为 np.nan）
      aw   : 湿项改正参数（如果 opt==1 则为 np.nan）
      undu : 地形改正量（米）
    """
    # 时间与物理常数
    dmjd1 = dmjd - 51544.5
    gm = 9.80665
    dMtr = 28.965e-3  # 单位: kg/mol
    Rg = 8.3143

    if it == 1:
        cosfy = 0.0
        coshy = 0.0
        sinfy = 0.0
        sinhy = 0.0
    else:
        cosfy = math.cos(dmjd1 / 365.25 * 2 * math.pi)
        coshy = math.cos(dmjd1 / 365.25 * 4 * math.pi)
        sinfy = math.sin(dmjd1 / 365.25 * 2 * math.pi)
        sinhy = math.sin(dmjd1 / 365.25 * 4 * math.pi)

    # 加载 .mat 数据文件
    mat_data = loadmat('gpt2_5.mat')
    if 'gpt2_5' not in mat_data:
        raise ValueError("无法加载变量 'gpt2_5' 请检查 gpt2_5.mat 文件。")
    gpt2_5 = mat_data['gpt2_5']

    # 根据 opt 分支提取各项网格数据
    if opt == 1 or opt == 0:
        pgrid = gpt2_5[:, 2:7]
        Tgrid = gpt2_5[:, 7:12]
        Qgrid = gpt2_5[:, 12:17] / 1000.0
        dTgrid = gpt2_5[:, 17:22] / 1000.0
        u_vec = gpt2_5[:, 22]
        Hs = gpt2_5[:, 23]
        ahgrid = gpt2_5[:, 24:29] / 1000.0
        awgrid = gpt2_5[:, 29:34] / 1000.0
    else:
        raise ValueError("Option have to be 1 or 0.")

    # 经纬度转换
    if dlon < 0:
        plon = (dlon + 2 * math.pi) * 180 / math.pi
    else:
        plon = dlon * 180 / math.pi
    ppod = (-dlat + math.pi / 2) * 180 / math.pi

    # 计算网格索引
    ipod = math.floor((ppod + 5) / 5)
    ilon = math.floor((plon + 5) / 5)
    diffpod = (ppod - (ipod * 5 - 2.5)) / 5
    difflon = (plon - (ilon * 5 - 2.5)) / 5
    if ipod == 37:
        ipod = 36
    ix = int((ipod - 1) * 72 + ilon) - 1

    # bilinear 标志
    bilinear = 1 if (ppod > 2.5 and ppod < 177.5) else 0

    # 初始化返回变量
    T = np.nan
    dT = np.nan
    e = np.nan
    ah = np.nan
    aw = np.nan

    # 分两种情况处理
    if bilinear == 0:
        # 直接采用最近网格点数据
        undu = u_vec[ix]
        hgt = hell - undu
        T0 = (Tgrid[ix, 0] + Tgrid[ix, 1] * cosfy + Tgrid[ix, 2] * sinfy +
              Tgrid[ix, 3] * coshy + Tgrid[ix, 4] * sinhy)
        p0 = (pgrid[ix, 0] + pgrid[ix, 1] * cosfy + pgrid[ix, 2] * sinfy +
              pgrid[ix, 3] * coshy + pgrid[ix, 4] * sinhy)
        Q_val = (Qgrid[ix, 0] + Qgrid[ix, 1] * cosfy + Qgrid[ix, 2] * sinfy +
                 Qgrid[ix, 3] * coshy + Qgrid[ix, 4] * sinhy)
        dT_val = (dTgrid[ix, 0] + dTgrid[ix, 1] * cosfy + dTgrid[ix, 2] * sinfy +
                  dTgrid[ix, 3] * coshy + dTgrid[ix, 4] * sinhy)

        if opt == 1:
            redh = hgt - Hs[ix]
            Tv = T0 * (1 + 0.6077 * Q_val)
            c_val = gm * dMtr / (Rg * Tv)
            p = (p0 * math.exp(-c_val * redh)) / 100.0
        elif opt == 0:
            redh = hgt - Hs[ix]
            T = T0 + dT_val * redh - 273.15
            dT = dT_val * 1000  # 转换为 mK/m
            Tv = T0 * (1 + 0.6077 * Q_val)
            c_val = gm * dMtr / (Rg * Tv)
            p = (p0 * math.exp(-c_val * redh)) / 100.0
            e = (Q_val * p) / (0.622 + 0.378 * Q_val)
            ah = (ahgrid[ix, 0] + ahgrid[ix, 1] * cosfy + ahgrid[ix, 2] * sinfy +
                  ahgrid[ix, 3] * coshy + ahgrid[ix, 4] * sinhy)
            aw = (awgrid[ix, 0] + awgrid[ix, 1] * cosfy + awgrid[ix, 2] * sinfy +
                  awgrid[ix, 3] * coshy + awgrid[ix, 4] * sinhy)
        else:
            raise ValueError("opt 必须为 0 或 1")
    else:
        # 双线性插值处理
        ipod1 = ipod + int(np.sign(diffpod))
        ilon1 = ilon + int(np.sign(difflon))
        if ilon1 == 73:
            ilon1 = 1
        if ilon1 == 0:
            ilon1 = 72
        # 计算 4 个网格点的索引
        ix1 = ix  # (ipod, ilon)
        ix2 = int((ipod1 - 1) * 72 + ilon) - 1
        ix3 = int((ipod - 1) * 72 + ilon1) - 1
        ix4 = int((ipod1 - 1) * 72 + ilon1) - 1

        undul = []
        Ql = []
        dTl = []
        Tl_list = []
        pl_list = []
        ahl_list = []
        awl_list = []
        indices = [ix1, ix2, ix3, ix4]

        for idx in indices:
            undul.append(u_vec[idx])
            hgt_l = hell - u_vec[idx]
            T0_l = (Tgrid[idx, 0] + Tgrid[idx, 1] * cosfy + Tgrid[idx, 2] * sinfy +
                    Tgrid[idx, 3] * coshy + Tgrid[idx, 4] * sinhy)
            p0_l = (pgrid[idx, 0] + pgrid[idx, 1] * cosfy + pgrid[idx, 2] * sinfy +
                    pgrid[idx, 3] * coshy + pgrid[idx, 4] * sinhy)
            Ql_val = (Qgrid[idx, 0] + Qgrid[idx, 1] * cosfy + Qgrid[idx, 2] * sinfy +
                      Qgrid[idx, 3] * coshy + Qgrid[idx, 4] * sinhy)
            Ql.append(Ql_val)
            Hs_val = Hs[idx]
            redh_l = hgt_l - Hs_val
            dTl_val = (dTgrid[idx, 0] + dTgrid[idx, 1] * cosfy + dTgrid[idx, 2] * sinfy +
                       dTgrid[idx, 3] * coshy + dTgrid[idx, 4] * sinhy)
            dTl.append(dTl_val)
            Tl_val = T0_l + dTl_val * redh_l - 273.15
            Tl_list.append(Tl_val)
            Tv_l = T0_l * (1 + 0.6077 * Ql_val)
            c_l = gm * dMtr / (Rg * Tv_l)
            pl_val = (p0_l * math.exp(-c_l * redh_l)) / 100.0
            pl_list.append(pl_val)
            ahl_val = (ahgrid[idx, 0] + ahgrid[idx, 1] * cosfy + ahgrid[idx, 2] * sinfy +
                       ahgrid[idx, 3] * coshy + ahgrid[idx, 4] * sinhy)
            ahl_list.append(ahl_val)
            awl_val = (awgrid[idx, 0] + awgrid[idx, 1] * cosfy + awgrid[idx, 2] * sinfy +
                       awgrid[idx, 3] * coshy + awgrid[idx, 4] * sinhy)
            awl_list.append(awl_val)

        dnpod1 = abs(diffpod)
        dnpod2 = 1 - dnpod1
        dnlon1 = abs(difflon)
        dnlon2 = 1 - dnlon1

        # 统一计算 undu
        R1_u = dnpod2 * undul[0] + dnpod1 * undul[1]
        R2_u = dnpod2 * undul[2] + dnpod1 * undul[3]
        undu = dnlon2 * R1_u + dnlon1 * R2_u

        if opt == 1:
            R1 = dnpod2 * pl_list[0] + dnpod1 * pl_list[1]
            R2 = dnpod2 * pl_list[2] + dnpod1 * pl_list[3]
            p = dnlon2 * R1 + dnlon1 * R2
            # 对于 opt==1，其它参数保持 np.nan
        elif opt == 0:
            R1 = dnpod2 * pl_list[0] + dnpod1 * pl_list[1]
            R2 = dnpod2 * pl_list[2] + dnpod1 * pl_list[3]
            p = dnlon2 * R1 + dnlon1 * R2

            R1 = dnpod2 * Tl_list[0] + dnpod1 * Tl_list[1]
            R2 = dnpod2 * Tl_list[2] + dnpod1 * Tl_list[3]
            T = dnlon2 * R1 + dnlon1 * R2

            R1 = dnpod2 * dTl[0] + dnpod1 * dTl[1]
            R2 = dnpod2 * dTl[2] + dnpod1 * dTl[3]
            dT = (dnlon2 * R1 + dnlon1 * R2) * 1000

            R1 = dnpod2 * Ql[0] + dnpod1 * Ql[1]
            R2 = dnpod2 * Ql[2] + dnpod1 * Ql[3]
            Q = dnlon2 * R1 + dnlon1 * R2
            e = (Q * p) / (0.622 + 0.378 * Q)

            R1 = dnpod2 * ahl_list[0] + dnpod1 * ahl_list[1]
            R2 = dnpod2 * ahl_list[2] + dnpod1 * ahl_list[3]
            ah = dnlon2 * R1 + dnlon1 * R2

            R1 = dnpod2 * awl_list[0] + dnpod1 * awl_list[1]
            R2 = dnpod2 * awl_list[2] + dnpod1 * awl_list[3]
            aw = dnlon2 * R1 + dnlon1 * R2
        else:
            raise ValueError("opt 必须为 0 或 1")

    return p, T, dT, e, ah, aw, undu

def gmf_f_hu(dmjd, dlat, dlon, dhgt, zd):
    """
    计算 gmfh 和 gmfw 的 Python 函数。

    参数:
      dmjd: Modified Julian Date（标量）
      dlat: 地理纬度（单位：弧度）
      dlon: 地理经度（单位：弧度）
      dhgt: 高度（单位：米）
      zd: 天顶角（单位：弧度）

    返回:
      gmfh, gmfw：计算得到的干项与湿项修正值
    """
    # 定义常量数组（共55个元素，每个数组均为一维 numpy 数组）
    ah_mean = np.array([
        +1.2517e+02, +8.503e-01, +6.936e-02, -6.760e+00, +1.771e-01,
        +1.130e-02, +5.963e-01, +1.808e-02, +2.801e-03, -1.414e-03,
        -1.212e+00, +9.300e-02, +3.683e-03, +1.095e-03, +4.671e-05,
        +3.959e-01, -3.867e-02, +5.413e-03, -5.289e-04, +3.229e-04,
        +2.067e-05, +3.000e-01, +2.031e-02, +5.900e-03, +4.573e-04,
        -7.619e-05, +2.327e-06, +3.845e-06, +1.182e-01, +1.158e-02,
        +5.445e-03, +6.219e-05, +4.204e-06, -2.093e-06, +1.540e-07,
        -4.280e-08, -4.751e-01, -3.490e-02, +1.758e-03, +4.019e-04,
        -2.799e-06, -1.287e-06, +5.468e-07, +7.580e-08, -6.300e-09,
        -1.160e-01, +8.301e-03, +8.771e-04, +9.955e-05, -1.718e-06,
        -2.012e-06, +1.170e-08, +1.790e-08, -1.300e-09, +1.000e-10
    ])

    bh_mean = np.array([
        +0.000e+00, +0.000e+00, +3.249e-02, +0.000e+00, +3.324e-02,
        +1.850e-02, +0.000e+00, -1.115e-01, +2.519e-02, +4.923e-03,
        +0.000e+00, +2.737e-02, +1.595e-02, -7.332e-04, +1.933e-04,
        +0.000e+00, -4.796e-02, +6.381e-03, -1.599e-04, -3.685e-04,
        +1.815e-05, +0.000e+00, +7.033e-02, +2.426e-03, -1.111e-03,
        -1.357e-04, -7.828e-06, +2.547e-06, +0.000e+00, +5.779e-03,
        +3.133e-03, -5.312e-04, -2.028e-05, +2.323e-07, -9.100e-08,
        -1.650e-08, +0.000e+00, +3.688e-02, -8.638e-04, -8.514e-05,
        -2.828e-05, +5.403e-07, +4.390e-07, +1.350e-08, +1.800e-09,
        +0.000e+00, -2.736e-02, -2.977e-04, +8.113e-05, +2.329e-07,
        +8.451e-07, +4.490e-08, -8.100e-09, -1.500e-09, +2.000e-10
    ])

    ah_amp = np.array([
        -2.738e-01, -2.837e+00, +1.298e-02, -3.588e-01, +2.413e-02,
        +3.427e-02, -7.624e-01, +7.272e-02, +2.160e-02, -3.385e-03,
        +4.424e-01, +3.722e-02, +2.195e-02, -1.503e-03, +2.426e-04,
        +3.013e-01, +5.762e-02, +1.019e-02, -4.476e-04, +6.790e-05,
        +3.227e-05, +3.123e-01, -3.535e-02, +4.840e-03, +3.025e-06,
        -4.363e-05, +2.854e-07, -1.286e-06, -6.725e-01, -3.730e-02,
        +8.964e-04, +1.399e-04, -3.990e-06, +7.431e-06, -2.796e-07,
        -1.601e-07, +4.068e-02, -1.352e-02, +7.282e-04, +9.594e-05,
        +2.070e-06, -9.620e-08, -2.742e-07, -6.370e-08, -6.300e-09,
        +8.625e-02, -5.971e-03, +4.705e-04, +2.335e-05, +4.226e-06,
        +2.475e-07, -8.850e-08, -3.600e-08, -2.900e-09, +0.000e+00
    ])

    bh_amp = np.array([
        +0.000e+00, +0.000e+00, -1.136e-01, +0.000e+00, -1.868e-01,
        -1.399e-02, +0.000e+00, -1.043e-01, +1.175e-02, -2.240e-03,
        +0.000e+00, -3.222e-02, +1.333e-02, -2.647e-03, -2.316e-05,
        +0.000e+00, +5.339e-02, +1.107e-02, -3.116e-03, -1.079e-04,
        -1.299e-05, +0.000e+00, +4.861e-03, +8.891e-03, -6.448e-04,
        -1.279e-05, +6.358e-06, -1.417e-07, +0.000e+00, +3.041e-02,
        +1.150e-03, -8.743e-04, -2.781e-05, +6.367e-07, -1.140e-08,
        -4.200e-08, +0.000e+00, -2.982e-02, -3.000e-03, +1.394e-05,
        -3.290e-05, -1.705e-07, +7.440e-08, +2.720e-08, -6.600e-09,
        +0.000e+00, +1.236e-02, -9.981e-04, -3.792e-05, -1.355e-05,
        +1.162e-06, -1.789e-07, +1.470e-08, -2.400e-09, -4.000e-10
    ])

    aw_mean = np.array([
        +5.640e+01, +1.555e+00, -1.011e+00, -3.975e+00, +3.171e-02,
        +1.065e-01, +6.175e-01, +1.376e-01, +4.229e-02, +3.028e-03,
        +1.688e+00, -1.692e-01, +5.478e-02, +2.473e-02, +6.059e-04,
        +2.278e+00, +6.614e-03, -3.505e-04, -6.697e-03, +8.402e-04,
        +7.033e-04, -3.236e+00, +2.184e-01, -4.611e-02, -1.613e-02,
        -1.604e-03, +5.420e-05, +7.922e-05, -2.711e-01, -4.406e-01,
        -3.376e-02, -2.801e-03, -4.090e-04, -2.056e-05, +6.894e-06,
        +2.317e-06, +1.941e+00, -2.562e-01, +1.598e-02, +5.449e-03,
        +3.544e-04, +1.148e-05, +7.503e-06, -5.667e-07, -3.660e-08,
        +8.683e-01, -5.931e-02, -1.864e-03, -1.277e-04, +2.029e-04,
        +1.269e-05, +1.629e-06, +9.660e-08, -1.015e-07, -5.000e-10
    ])

    bw_mean = np.array([
        +0.000e+00, +0.000e+00, +2.592e-01, +0.000e+00, +2.974e-02,
        -5.471e-01, +0.000e+00, -5.926e-01, -1.030e-01, -1.567e-02,
        +0.000e+00, +1.710e-01, +9.025e-02, +2.689e-02, +2.243e-03,
        +0.000e+00, +3.439e-01, +2.402e-02, +5.410e-03, +1.601e-03,
        +9.669e-05, +0.000e+00, +9.502e-02, -3.063e-02, -1.055e-03,
        -1.067e-04, -1.130e-04, +2.124e-05, +0.000e+00, -3.129e-01,
        +8.463e-03, +2.253e-04, +7.413e-05, -9.376e-05, -1.606e-06,
        +2.060e-06, +0.000e+00, +2.739e-01, +1.167e-03, -2.246e-05,
        -1.287e-04, -2.438e-05, -7.561e-07, +1.158e-06, +4.950e-08,
        +0.000e+00, -1.344e-01, +5.342e-03, +3.775e-04, -6.756e-05,
        -1.686e-06, -1.184e-06, +2.768e-07, +2.730e-08, +5.700e-09
    ])

    aw_amp = np.array([
        +1.023e-01, -2.695e+00, +3.417e-01, -1.405e-01, +3.175e-01,
        +2.116e-01, +3.536e+00, -1.505e-01, -1.660e-02, +2.967e-02,
        +3.819e-01, -1.695e-01, -7.444e-02, +7.409e-03, -6.262e-03,
        -1.836e+00, -1.759e-02, -6.256e-02, -2.371e-03, +7.947e-04,
        +1.501e-04, -8.603e-01, -1.360e-01, -3.629e-02, -3.706e-03,
        -2.976e-04, +1.857e-05, +3.021e-05, +2.248e+00, -1.178e-01,
        +1.255e-02, +1.134e-03, -2.161e-04, -5.817e-06, +8.836e-07,
        -1.769e-07, +7.313e-01, -1.188e-01, +1.145e-02, +1.011e-03,
        +1.083e-04, +2.570e-06, -2.140e-06, -5.710e-08, +2.000e-08,
        -1.632e+00, -6.948e-03, -3.893e-03, +8.592e-04, +7.577e-05,
        +4.539e-06, -3.852e-07, -2.213e-07, -1.370e-08, +5.800e-09
    ])

    bw_amp = np.array([
        +0.000e+00, +0.000e+00, -8.865e-02, +0.000e+00, -4.309e-01,
        +6.340e-02, +0.000e+00, +1.162e-01, +6.176e-02, -4.234e-03,
        +0.000e+00, +2.530e-01, +4.017e-02, -6.204e-03, +4.977e-03,
        +0.000e+00, -1.737e-01, -5.638e-03, +1.488e-04, +4.857e-04,
        -1.809e-04, +0.000e+00, -1.514e-01, -1.685e-02, +5.333e-03,
        -7.611e-05, +2.394e-05, +8.195e-06, +0.000e+00, +9.326e-02,
        -1.275e-02, -3.071e-04, +5.374e-05, -3.391e-05, -7.436e-06,
        +6.747e-07, +0.000e+00, -8.637e-02, -3.807e-03, -6.833e-04,
        -3.861e-05, -2.268e-05, +1.454e-06, +3.860e-07, -1.068e-07,
        +0.000e+00, -2.658e-02, -1.947e-03, +7.131e-04, -3.506e-05,
        +1.885e-07, +5.792e-07, +3.990e-08, +2.000e-08, -5.700e-09
    ])

    pi = np.pi

    # 计算日数（doy）
    # doy = dmjd - 44239 + 1 - 28 = dmjd - 44266
    doy = dmjd - 44266.0

    nmax = 9

    # 球坐标计算，假定 dlat, dlon 为弧度
    x = np.cos(dlat) * np.cos(dlon)
    y = np.cos(dlat) * np.sin(dlon)
    z = np.sin(dlat)

    # 初始化 V 和 W 矩阵
    V = np.zeros((nmax + 1, nmax + 1))
    W = np.zeros((nmax + 1, nmax + 1))

    # 第一列赋值（相当于 m=0 的部分）
    V[0, 0] = 1.0
    W[0, 0] = 0.0
    V[1, 0] = z * V[0, 0]
    W[1, 0] = 0.0

    for n in range(2, nmax + 1):
        V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / n
        W[n, 0] = 0.0

    # 递归计算 V 和 W 的其他部分

    for m in range(1, nmax + 1):

        V[m, m] = (2 * m - 1) * (x * V[m - 1, m - 1] - y * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (x * W[m - 1, m - 1] + y * V[m - 1, m - 1])

        if m < nmax:

            V[m + 1, m] = (2 * m + 1) * z * V[m, m]

            W[m + 1, m] = (2 * m + 1) * z * W[m, m]

        # 内层 n 循环，从 n = m+2 到 nmax (含)
        for n in range(m + 2, nmax + 1):

            V[n, m] = ((2 * n - 1) * z * V[n - 1, m] - (n + m - 1) * V[n - 2, m]) / (n - m)

            W[n, m] = ((2 * n - 1) * z * W[n - 1, m] - (n + m - 1) * W[n - 2, m]) / (n - m)

    # 地磁干项计算
    bh = 0.0029
    c0h = 0.062
    # dlat 为弧度，当 dlat < 0 时
    if dlat < 0:
        phh = pi
        c11h = 0.007
        c10h = 0.002
    else:
        phh = 0.0
        c11h = 0.005
        c10h = 0.001

    ch = c0h + (((np.cos(doy / 365.25 * 2 * pi + phh) + 1) * c11h / 2 + c10h) * (1 - np.cos(dlat)))

    ahm = 0.0
    aha = 0.0
    i = 0
    for n in range(0, nmax + 1):
        for m in range(0, n + 1):
            ahm += ah_mean[i] * V[n, m] + bh_mean[i] * W[n, m]
            aha += ah_amp[i] * V[n, m] + bh_amp[i] * W[n, m]
            i += 1

    # 注意：这里的 cos() 部分使用 doy 计算年周期变化
    ah = (ahm + aha * np.cos(doy / 365.25 * 2 * pi)) * 1e-5

    sine = np.sin(pi / 2 - zd)

    beta = bh / (sine + ch)
    gamma = ah / (sine + beta)
    topcon = 1.0 + ah / (1.0 + bh / (1.0 + ch))
    gmfh = topcon / (sine + gamma)

    # 高度修正项
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    hs_km = dhgt / 1000.0

    beta = b_ht / (sine + c_ht)
    gamma = a_ht / (sine + beta)
    topcon = 1.0 + a_ht / (1.0 + b_ht / (1.0 + c_ht))
    ht_corr_coef = 1.0 / sine - topcon / (sine + gamma)
    ht_corr = ht_corr_coef * hs_km
    gmfh = gmfh + ht_corr

    # 湿项部分计算
    bw = 0.00146
    cw = 0.04391

    awm = 0.0
    awa = 0.0
    i = 0
    for n in range(0, nmax + 1):
        for m in range(0, n + 1):
            awm += aw_mean[i] * V[n, m] + bw_mean[i] * W[n, m]
            awa += aw_amp[i] * V[n, m] + bw_amp[i] * W[n, m]
            i += 1
    aw = (awm + awa * np.cos(doy / 365.25 * 2 * pi)) * 1e-5

    beta = bw / (sine + cw)
    gamma = aw / (sine + beta)
    topcon = 1.0 + aw / (1.0 + bw / (1.0 + cw))
    gmfw = topcon / (sine + gamma)

    return gmfh, gmfw

def trop_gmf(rec, sat, dmjd, p):
    """
    Trop_GMF 模型（GMF 大气折射延迟模型）

    输入:
      rec  : 接收机坐标（例如 [dlat, dlon, hell]，dlat和dlon以弧度表示，hell 单位：米）
      sat  : 卫星坐标（用于计算观测方向）
      dmjd : Modified Julian Date（标量）
      p    : 地面气压（单位：hPa）

    返回:
      Trop : 对流折射延迟（延迟距离，与 ZHD 相乘）
      Mwet : 湿项折射延迟函数值（gmfw）
      Mn   : 水平（北向）分量
      Me   : 水平（东向）分量
      ZHD  : 干延迟（Zenith Hydrostatic Delay）
    """
    # 1. 将接收机坐标转换为大地坐标，得到 dlat、dlon（均以弧度表示）和 hell（米）
    ellp = xyz2plh(rec, 0)
    dlat = ellp[0]
    dlon = ellp[1]
    hell = ellp[2]

    # 2. 计算接收机与卫星的局部观测角（Az: 方位角，Elv: 仰角），单位均为弧度
    Az, Elv = local(rec, sat, 0)

    # 3. 干延迟基本常数
    f = 0.0022768
    # 计算 k 值，注意 0.28*10^-6 写作 0.28e-6
    k = 1 - (0.00266 * math.cos(2 * dlat)) - (0.28e-6 * hell)
    ZHD = f * (p / k)

    # 4. 计算 GMF 模型中的干项和湿项函数值：
    #    此处调用 gmf_f_hu(dmjd, dlat, dlon, hell, (pi/2 - Elv))
    gmfh, gmfw = gmf_f_hu(dmjd, dlat, dlon, hell, (math.pi / 2 - Elv))

    # 5. 对流延迟
    Trop = gmfh * ZHD
    Mwet = gmfw

    # 6. 计算水平分量
    # Mg = 1/((tan(Elv)*sin(Elv)) + 0.0032)
    Mg = 1 / ((math.tan(Elv) * math.sin(Elv)) + 0.0032)
    Mn = Mg * math.cos(Az)
    Me = Mg * math.sin(Az)

    return Trop, Mwet, Mn, Me, ZHD

