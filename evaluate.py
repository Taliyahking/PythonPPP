import numpy as np
from helper_function import xyz2plh


def evaluate(xs, ref):

    # 1. 处理ref - 确保是列向量
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)

    # 2. 计算差值
    set_values = xs[0:3, :] - ref


    # 3. 获取geodetic坐标
    elip = xyz2plh(ref.flatten(), 1)
    lat = elip[0]
    lon = elip[1]

    # 4. 构建转换矩阵
    A = np.array([
        [-np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)), -np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
         np.cos(np.deg2rad(lat))],
        [-np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon)), 0],
        [np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon)), np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon)),
         np.sin(np.deg2rad(lat))]
    ])

    # 5. 坐标变换 - 关键修改：添加转置操作
    loc = np.zeros((set_values.shape[1], 3))
    for i in range(set_values.shape[1]):
        result = A @ set_values[:, i]
        loc[i, :] = result.T

    # 6. 提取分量
    n = loc[:, 0]
    e = loc[:, 1]
    u = loc[:, 2]

    # 7. 计算3D距离
    thrD = np.sqrt(n ** 2 + e ** 2 + u ** 2)

    CT = 1
    dur = 19
    for i in range(1, set_values.shape[1] + 1):
        if i + dur < thrD.shape[0]:
            val = thrD[i - 1:i + dur - 1] > 0.1
            if np.sum(val) == 0:
                CT = i
                break
        else:
            val = thrD[i - 1:] > 0.1
            if np.sum(val) == 0:
                CT = i
                break

    # 9. 计算RMS值
    rms = np.zeros((3, 1))
    rms[0, 0] = np.sqrt(np.mean(n[CT - 1:] ** 2))
    rms[1, 0] = np.sqrt(np.mean(e[CT - 1:] ** 2))
    rms[2, 0] = np.sqrt(np.mean(u[CT - 1:] ** 2))

    return n, e, u, CT, thrD, rms
