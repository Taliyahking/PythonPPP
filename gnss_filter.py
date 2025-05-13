from helper_function import dtr_satno,frequencies,i_free,dtr_sys,dtr_satlist
import numpy as np
from numpy.linalg import norm, pinv


def kalman_filtering(sno, sls, pno, meas, x1k, p1k, sit, bp, freq, options):
    """
    用于GNSS处理的卡尔曼滤波函数

    参数：
    --------
    sno : int
        卫星数量
    sls : numpy.ndarray
        卫星编号列表
    pno : int
        参数数量
    meas : numpy.ndarray
        测量矩阵
    x1k : numpy.ndarray
        初始状态向量
    p1k : numpy.ndarray
        初始状态协方差矩阵
    sit : int
        系统识别类型
    bp : int
        基本参数
    freq : numpy.ndarray
        频率矩阵
    options : object
        选项对象

    返回：
    --------
    xk : numpy.ndarray
        更新后的状态向量
    pk : numpy.ndarray
        更新后的状态协方差矩阵
    kof : numpy.ndarray
        卡尔曼滤波信息矩阵
    """
    # 测量数量
    mno = 2 * sno

    # 初始化矩阵
    Hk = np.zeros((mno, pno))
    Zk = np.zeros((mno, 1))
    Ck = np.zeros((mno, 1))

    # 初始化测量噪声协方差矩阵
    Rk = np.eye(mno)

    # 处理每颗卫星
    for k in range(1, sno + 1):
        # 卫星坐标
        sat = meas[(4 * k) - 4, 7:10]

        # 计算几何距离
        rho = norm(sat - x1k[0:3, 0].T)

        # Partial derivatives
        Jx = (x1k[0, 0] - sat[0]) / rho
        Jy = (x1k[1, 0] - sat[1]) / rho
        Jz = (x1k[2, 0] - sat[2]) / rho
        Jt = 1
        Jw = meas[(4 * k - 4), 27]
        Jtn = meas[(4 * k - 4), 28]
        Jte = meas[(4 * k - 4), 29]
        Jn = 1
        Jr = 1
        Je = 1
        Jc = 1

        # 矩阵索引
        s = (2 * k - 2)
        f = (2 * k - 1)

        # 填充设计矩阵
        Hk[s:f + 1, 0] = Jx
        Hk[s:f + 1, 1] = Jy
        Hk[s:f + 1, 2] = Jz
        Hk[s:f + 1, 3] = Jt
        Hk[s:f + 1, 4] = Jw
        Hk[f, (bp + k - 1)] = Jn

        # 基于对流层梯度选项填充设计矩阵
        if options['TroGrad'] == 0:
            if sit == 3 or sit == 4 or sit == 5:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 5] = 0
                else:
                    Hk[s:f + 1, 5] = 1
            elif sit == 6 or sit == 7:
                if sls[k - 1] < 59:
                    Hk[s:f + 1, 5] = 0
                else:
                    Hk[s:f + 1, 5] = 1
            elif sit == 8 or sit == 9:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 0
                elif sls[k - 1] < 59:
                    Hk[s:f + 1, 5] = 1
                    Hk[s:f + 1, 6] = 0
                else:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 1
            elif sit == 10:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 5] = 1
                    Hk[s:f + 1, 6] = 0
                else:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 1
            elif sit == 11:
                if sls[k - 1] < 59:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 5] = 1
                    Hk[s:f + 1, 6] = 0
                else:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 1
            elif sit == 12:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 0
                    Hk[s:f + 1, 7] = 0
                elif sls[k - 1] < 59:
                    Hk[s:f + 1, 5] = 1
                    Hk[s:f + 1, 6] = 0
                    Hk[s:f + 1, 7] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 1
                    Hk[s:f + 1, 7] = 0
                elif sls[k - 1] < 106:
                    Hk[s:f + 1, 5] = 0
                    Hk[s:f + 1, 6] = 0
                    Hk[s:f + 1, 7] = 1
        else:  # options['TroGrad'] == 1
            Hk[s:f + 1, 5] = Jtn
            Hk[s:f + 1, 6] = Jte
            if sit == 3 or sit == 4 or sit == 5:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 7] = 0
                else:
                    Hk[s:f + 1, 7] = 1
            elif sit == 6 or sit == 7:
                if sls[k - 1] < 59:
                    Hk[s:f + 1, 7] = 0
                else:
                    Hk[s:f + 1, 7] = 1
            elif sit == 8 or sit == 9:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 0
                elif sls[k - 1] < 59:
                    Hk[s:f + 1, 7] = 1
                    Hk[s:f + 1, 8] = 0
                else:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 1
            elif sit == 10:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 7] = 1
                    Hk[s:f + 1, 8] = 0
                else:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 1
            elif sit == 11:
                if sls[k - 1] < 59:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 7] = 1
                    Hk[s:f + 1, 8] = 0
                else:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 1
            elif sit == 12:
                if sls[k - 1] < 33:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 0
                    Hk[s:f + 1, 9] = 0
                elif sls[k - 1] < 59:
                    Hk[s:f + 1, 7] = 1
                    Hk[s:f + 1, 8] = 0
                    Hk[s:f + 1, 9] = 0
                elif sls[k - 1] < 89:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 1
                    Hk[s:f + 1, 9] = 0
                elif sls[k - 1] < 106:
                    Hk[s:f + 1, 7] = 0
                    Hk[s:f + 1, 8] = 0
                    Hk[s:f + 1, 9] = 1

        # 填充测量向量（电离层无关测量）
        if sls[k - 1] < 33:
            Zk[s, 0] = i_free(meas[(4 * k - 4), 5], meas[(4 * k - 3), 5], 0)
            Zk[f, 0] = i_free(meas[(4 * k - 2), 5], meas[(4 * k - 1), 5], 0)
        elif sls[k - 1] < 59:
            Zk[s, 0] = i_free(meas[(4 * k - 4), 5], meas[(4 * k - 3), 5], 1)
            Zk[f, 0] = i_free(meas[(4 * k - 2), 5], meas[(4 * k - 1), 5], 1)
        elif sls[k - 1] < 89:
            Zk[s, 0] = i_free(meas[(4 * k - 4), 5], meas[(4 * k - 3), 5], 2)
            Zk[f, 0] = i_free(meas[(4 * k - 2), 5], meas[(4 * k - 1), 5], 2)
        elif sls[k - 1] < 106:
            Zk[s, 0] = i_free(meas[(4 * k - 4), 5], meas[(4 * k - 3), 5], 3)
            Zk[f, 0] = i_free(meas[(4 * k - 2), 5], meas[(4 * k - 1), 5], 3)

        #  填充计算向量（电离层无关修正）
        p1c = meas[(4 * k - 4), 6]
        p2c = meas[(4 * k - 3), 6]
        l1c = meas[(4 * k - 2), 6]
        l2c = meas[(4 * k - 1), 6]

        if sls[k - 1] < 33:
            pc = i_free(p1c, p2c, 0)
            lc = i_free(l1c, l2c, 0)
        elif sls[k - 1] < 59:
            pc = i_free(p1c, p2c, 1)
            lc = i_free(l1c, l2c, 1)
        elif sls[k - 1] < 89:
            pc = i_free(p1c, p2c, 2)
            lc = i_free(l1c, l2c, 2)
        elif sls[k - 1] < 106:
            pc = i_free(p1c, p2c, 3)
            lc = i_free(l1c, l2c, 3)

        # 填充基于对流层梯度选项的计算值
        if options['TroGrad'] == 0:
            if sit == 1:
                Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 2:
                Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 3 or sit == 4 or sit == 5:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
            elif sit == 6 or sit == 7:
                if sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
            elif sit == 8 or sit == 9:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
            elif sit == 10:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
            elif sit == 11:
                if sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[6, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
            elif sit == 12:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jr * x1k[5, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Je * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Je * x1k[6, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 106:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jc * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jc * x1k[7, 0]) + (
                                Jn * x1k[bp + k - 1, 0])
        else:  # options['TroGrad'] == 1
            if sit == 1:
                Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0]) + (
                            Jn * x1k[bp + k - 1, 0])
            elif sit == 2:
                Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0]) + (
                            Jn * x1k[bp + k - 1, 0])
            elif sit == 3 or sit == 4 or sit == 5:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 6 or sit == 7:
                if sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 8 or sit == 9:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 10:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 11:
                if sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
                else:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[8, 0]) + (Jn * x1k[bp + k - 1, 0])
            elif sit == 12:
                if sls[k - 1] < 33:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (Jte * x1k[6, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 59:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jr * x1k[7, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 89:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Je * x1k[8, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Je * x1k[8, 0]) + (Jn * x1k[bp + k - 1, 0])
                elif sls[k - 1] < 106:
                    Ck[s, 0] = rho + pc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jc * x1k[9, 0])
                    Ck[f, 0] = rho + lc + (Jt * x1k[3, 0]) + (Jw * x1k[4, 0]) + (Jtn * x1k[5, 0]) + (
                                Jte * x1k[6, 0]) + (Jc * x1k[9, 0]) + (Jn * x1k[bp + k - 1, 0])

        # 权重方法 - 高度相关
        if options['WeMethod'] == 'Elevation Dependent':
            if sls[k - 1] < 33:
                f1 = freq[sls[k - 1] - 1, 0]
                f2 = freq[sls[k - 1] - 1, 1]

                ab1 = ((f1 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                ab2 = ((f2 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                elv = meas[(4 * k - 2), 25]

                kp = 1
                kl = 1
                code_var = ((options['CodeStd'] * kp) ** 2) * (ab1 + ab2)
                phas_var = ((options['PhaseStd'] * kl) ** 2) * (ab1 + ab2)

                Rk[s, s] = code_var / (np.sin(np.radians(elv)))
                Rk[f, f] = phas_var / (np.sin(np.radians(elv)))

            elif sls[k - 1] < 59:
                f1 = freq[sls[k - 1] - 1, 0]
                f2 = freq[sls[k - 1] - 1, 1]

                ab1 = ((f1 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                ab2 = ((f2 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                elv = meas[(4 * k - 2), 25]

                if sit == 2 or sit == 6 or sit == 7 or sit == 11:
                    kp = 1
                    kl = 1
                else:
                    kp = 2
                    kl = 1
                code_var = ((options['CodeStd'] * kp) ** 2) * (ab1 + ab2)
                phas_var = ((options['PhaseStd'] * kl) ** 2) * (ab1 + ab2)

                Rk[s, s] = code_var / (np.sin(np.radians(elv)))
                Rk[f, f] = phas_var / (np.sin(np.radians(elv)))

            elif sls[k - 1] < 89:
                f1 = freq[sls[k - 1] - 1, 0]
                f2 = freq[sls[k - 1] - 1, 1]

                ab1 = ((f1 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                ab2 = ((f2 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                elv = meas[(4 * k - 2), 25]

                kp = 2
                kl = 2
                code_var = ((options['CodeStd'] * kp) ** 2) * (ab1 + ab2)
                phas_var = ((options['PhaseStd'] * kl) ** 2) * (ab1 + ab2)

                Rk[s, s] = code_var / (np.sin(np.radians(elv)))
                Rk[f, f] = phas_var / (np.sin(np.radians(elv)))

            elif sls[k - 1] < 106:
                f1 = freq[sls[k - 1] - 1, 0]
                f2 = freq[sls[k - 1] - 1, 1]

                ab1 = ((f1 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                ab2 = ((f2 ** 2) / (f1 ** 2 - f2 ** 2)) ** 2
                elv = meas[(4 * k - 2), 25]

                kp = 2
                kl = 2
                code_var = ((options['CodeStd'] * kp) ** 2) * (ab1 + ab2)
                phas_var = ((options['PhaseStd'] * kl) ** 2) * (ab1 + ab2)

                Rk[s, s] = code_var / (np.sin(np.radians(elv)))
                Rk[f, f] = phas_var / (np.sin(np.radians(elv)))

    # 处理NaN值
    if np.any(np.isnan(Zk)) or np.any(np.isnan(Ck)):
        # 找出Zk与Ck各自NaN所在行，然后并集
        aZ = np.where(np.isnan(Zk))[0]
        aC = np.where(np.isnan(Ck))[0]
        a = np.unique(np.concatenate([aZ, aC]))
        # 一次性删除
        Zk = np.delete(Zk, a, axis=0)
        Ck = np.delete(Ck, a, axis=0)
        Hk = np.delete(Hk, a, axis=0)
        Rk = np.delete(Rk, a, axis=0)
        Rk = np.delete(Rk, a, axis=1)

    # 初始化残差
    sres0 = np.zeros((Zk.shape[0], 1))

    # 迭代式稳健卡尔曼滤波
    while True:
        Vk = Zk - Ck
        Sk = (Hk @ p1k @ Hk.T) + Rk

        abf = np.sum(Vk ** 2) / np.trace(Sk)
        c0 = 2.5
        c1 = 6.5
        if abf > c1:
            af = 10 ** 10
        elif abf > c0:
            af = (c0 / abs(abf)) * ((c1 - abs(abf)) / (c1 - c0)) ** 2
        else:
            af = 1

        Sk = ((1 / af) * (Hk @ p1k @ Hk.T)) + Rk

        if np.linalg.cond(Sk) > 1e-15:
            Kk = ((1 / af) * (p1k @ Hk.T)) @ np.linalg.inv(Sk)
        else:
            Kk = ((1 / af) * (p1k @ Hk.T)) @ pinv(Sk)

        dx = Kk @ Vk
        xk = x1k + dx
        tnk = np.eye(pno) - Kk @ Hk
        pk = tnk @ p1k @ tnk.T + Kk @ Rk @ Kk.T
        # 强制协方差矩阵对称
        pk = (pk + pk.T) / 2

        if np.linalg.cond(Rk) > 1e-15:
            kof = pinv(Hk.T @ np.linalg.inv(Rk) @ Hk)
        else:
            kof = pinv(Hk.T @ pinv(Rk) @ Hk)

        kof = kof[0:5, 0:5]

        res = (Ck + (Hk @ dx)) - Zk
        vres = np.abs(Rk - (Hk @ pk @ Hk.T))

        sres = np.zeros((res.shape[0], 1))
        for si in range(res.shape[0]):
            sres[si, 0] = abs(res[si, 0]) / np.sqrt(vres[si, si])

        dres = np.abs(sres - sres0)

        if np.any(dres > 0.1):
            mm = np.argmax(np.abs(sres))
            k0 = 2.5
            k1 = 6.5
            if sres[mm, 0] > k1:
                sm = 1e-10
                Rk[mm, mm] = Rk[mm, mm] / sm
            elif sres[mm, 0] > k0:
                sm = (k0 / abs(sres[mm, 0])) * ((k1 - abs(sres[mm, 0])) / (k1 - k0)) ** 2
                Rk[mm, mm] = Rk[mm, mm] / sm

            sres0 = sres
        else:
            break

    return xk, pk, kof

def gnssfilter(model, data, options):
    """
    基于多GNSS系统的卡尔曼滤波定位处理

    参数:
    model (numpy.ndarray): 测量模型
    data (dict): 包含观测数据的字典
    options (dict): 计算选项

    返回:
    tuple: (xs, kofs, pks) 状态估计、系数矩阵和协方差矩阵
    """

    satlist = dtr_satlist(data['obsd'])

    # 获取每个历元的可用卫星数
    satno = dtr_satno(data['obsd'])

    # 历元总数
    rn = data['obsd']['st'].shape[0]

    # 获取频率信息
    freq, _ = frequencies()

    # 获取系统参数
    bp, _, sit = dtr_sys(options)

    # 初始化结果数组
    xs = np.zeros((bp + 105, rn))
    pks = np.zeros((bp + 105, bp + 105, rn))
    kofs = np.zeros((5, 5, rn))

    n = 0  # 初始索引（Python从0开始）

    # 遍历所有历元
    for i in range(rn):
        # 当前历元的可见卫星列表 (1-based索引)
        sls = satlist[i]

        # 当前历元的可见卫星数
        sno = int(satno[i, 0])

        # 参数总数
        pno = bp + sno

        # 计算当前批次测量数据的索引范围
        nk = n + (4 * sno)
        meas = model[n:nk, :]

        # 更新下一批次的起始索引
        n = nk

        # 初始化过程噪声协方差矩阵
        Q = np.zeros((pno, pno))

        # 根据处理模式设置噪声（0:运动学 1:静态）
        if options['ProMod'] == 0:
            Q[0, 0] = (options['NosPos'] * 10 ** (options['NosPos2'])) * (data['obsh']['time']['obsinterval'])
            Q[1, 1] = (options['NosPos'] * 10 ** (options['NosPos2'])) * (data['obsh']['time']['obsinterval'])
            Q[2, 2] = (options['NosPos'] * 10 ** (options['NosPos2'])) * (data['obsh']['time']['obsinterval'])

        Q[3, 3] = (options['NosClk'] * 10 ** (options['NosClk2'])) * (data['obsh']['time']['obsinterval'])
        Q[4, 4] = (options['NosTrop'] * 10 ** (options['NosTrop2'])) * (data['obsh']['time']['obsinterval'])

        # 根据对流层梯度选项设置噪声
        if options['TroGrad'] == 0:
            if bp == 6:
                Q[5, 5] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
            elif bp == 7:
                Q[5, 5] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[6, 6] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
            elif bp == 8:
                Q[5, 5] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[6, 6] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[7, 7] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
        elif options['TroGrad'] == 1:
            Q[5, 5] = (1 * 10 ** (-12)) * (data['obsh']['time']['obsinterval'])
            Q[6, 6] = (1 * 10 ** (-12)) * (data['obsh']['time']['obsinterval'])
            if bp == 8:
                Q[7, 7] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
            elif bp == 9:
                Q[7, 7] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[8, 8] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
            elif bp == 10:
                Q[7, 7] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[8, 8] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])
                Q[9, 9] = (options['NosSTD'] * 10 ** (options['NosSTD2'])) * (data['obsh']['time']['obsinterval'])

        # 状态转移矩阵（恒等矩阵）
        F = np.eye(pno)

        # 初始化和状态预测
        if i == 0:  # 第一个历元，使用先验信息
            # 初始化状态向量
            x1k = np.zeros((pno, 1))

            # 初始化先验协方差矩阵
            p1k = np.zeros((pno, pno))
            p1k[0, 0] = (options['IntPos'] * 10 ** (options['IntPos2'])) ** 2
            p1k[1, 1] = (options['IntPos'] * 10 ** (options['IntPos2'])) ** 2
            p1k[2, 2] = (options['IntPos'] * 10 ** (options['IntPos2'])) ** 2
            p1k[3, 3] = (options['IntClk'] * 10 ** (options['IntClk2'])) ** 2
            p1k[4, 4] = (options['IntTrop'] * 10 ** (options['IntTrop2'])) ** 2

            # 根据对流层梯度选项设置初始协方差
            if options['TroGrad'] == 0:
                if bp == 6:
                    p1k[5, 5] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                elif bp == 7:
                    p1k[5, 5] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[6, 6] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                elif bp == 8:
                    p1k[5, 5] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[6, 6] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[7, 7] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
            elif options['TroGrad'] == 1:
                p1k[5, 5] = (0 * 10 ** (1)) ** 2
                p1k[6, 6] = (0 * 10 ** (1)) ** 2
                if bp == 8:
                    p1k[7, 7] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                elif bp == 9:
                    p1k[7, 7] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[8, 8] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                elif bp == 10:
                    p1k[7, 7] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[8, 8] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2
                    p1k[9, 9] = (options['IntSTD'] * 10 ** (options['IntSTD2'])) ** 2

            # 设置模糊度先验协方差
            for u in range(bp, pno):
                p1k[u, u] = (options['IntAmb'] * 10 ** (options['IntAmb2'])) ** 2

        elif i > 0:  # 非第一个历元，使用上一历元的结果
            # 初始化状态向量
            x1k = np.zeros((pno, 1))

            # 复制上一历元的基本参数
            x1k[0:bp, 0] = xs[0:bp, i - 1]

            # 复制上一历元的模糊度参数 - 注意sls是1-based索引
            for k in range(sno):
                snm = int(sls[k])
                x1k[bp + k, 0] = xs[bp + snm - 1, i - 1]

            # 状态预测
            x1k = np.dot(F, x1k)

            # 协方差预测
            p1k = np.zeros((pno, pno))
            for r in range(pno):
                for c in range(pno):
                    if r < bp and c < bp:
                        p1k[r, c] = pks[r, c, i - 1]  # 使用上一历元的pks
                    elif r < bp and c >= bp:
                        sn = int(sls[c - bp])
                        p1k[r, c] = pks[r, bp + sn - 1, i - 1]
                    elif r >= bp and c < bp:
                        sn = int(sls[r - bp])
                        p1k[r, c] = pks[bp + sn - 1, c, i - 1]
                    else:
                        f1 = int(sls[r - bp])
                        f2 = int(sls[c - bp])
                        p1k[r, c] = pks[bp + f1 - 1, bp + f2 - 1, i - 1]

            # 检查并修复零协方差模糊度
            for k in range(sno):
                snm = int(sls[k]) + bp - 1  # 调整为Python索引
                if pks[snm, snm, i - 1] == 0:
                    p1k[k + bp, k + bp] = (options['IntAmb'] * 10 ** (options['IntAmb2'])) ** 2

            # 应用状态转移和过程噪声
            p1k = np.dot(np.dot(F, p1k), F.T) + Q
            # 强制预测协方差矩阵对称
            p1k = (p1k + p1k.T) / 2

        # 设置初始位置并使用卡尔曼滤波
        if i == 0:
            # 使用先验信息设置初始位置
            if options['ApMethod'] == 'RINEX':  # 使用RINEX文件中的位置
                x1k[0:3, 0] = data['obsh']['sta']['pos']
            elif options['ApMethod'] == 'Specify':  # 使用用户指定的位置
                x1k[0:3, 0] = np.array([options['AprioriX'], options['AprioriY'], options['AprioriZ']])

        # 使用卡尔曼滤波
        xk, pk, kof = kalman_filtering(sno, sls, pno, meas, x1k, p1k, sit, bp, freq, options)

        # 存储滤波结果
        kofs[:, :, i] = kof

        # 存储基本参数
        xs[0:bp, i] = xk[0:bp, 0]

        # 存储模糊度参数 - 注意sls是1-based索引
        for k in range(sno):
            snm = int(sls[k]) + bp
            xs[snm - 1, i] = xk[bp + k, 0]  # 减1以获取正确的Python索引

        # 存储协方差矩阵
        ps = np.zeros((bp + 105, bp + 105))
        for r in range(pk.shape[0]):
            for c in range(pk.shape[1]):
                if r < bp and c < bp:
                    ps[r, c] = pk[r, c]
                elif r < bp and c >= bp:
                    sn = int(sls[c - bp])
                    ps[r, bp + sn - 1] = pk[r, c]  # 减1以获取正确的Python索引
                elif r >= bp and c < bp:
                    sn = int(sls[r - bp])
                    ps[bp + sn - 1, c] = pk[r, c]  # 减1以获取正确的Python索引
                else:
                    sn1 = int(sls[r - bp])
                    sn2 = int(sls[c - bp])
                    ps[bp + sn1 - 1, bp + sn2 - 1] = pk[r, c]  # 减1以获取正确的Python索引

        # 强制协方差矩阵对称
        ps = (ps + ps.T) / 2
        pks[:, :, i] = ps

    # 返回结果
    return xs, kofs, pks
