from gnss_filter import gnssfilter
from datahand import data_read
from pre_process import preprocess
import numpy as np
from helper_function import xyz2plh, save_visualization_data
from model_correct import nmodel
from evaluate import evaluate

if __name__ == "__main__":
    # 1. 定义数据文件路径
    files = {
        'obs': './Data/algo0670.22o',
        'sp3': './Data/gfz22002.sp3',
        'sp3a': './Data/gfz22003.sp3',
        'sp3b': './Data/gfz22001.sp3',
        'clk': './Data/gfz22002.clk',
        'atx': './Data/igs14.atx'
    }

    # 2. 配置处理选项
    options = {
        # 基本选项
        "dcb": 0,
        "clk_file": 1,
        "clk_int": 30,
        "lono": 1,
        "system": {
            "gps": 1,
            "glo": 1,
            "gal": 1,
            "bds": 1
        },

        # Kalman滤波器相关选项
        "ProMod": 1,
        "NosPos": 0,
        "NosPos2": 0,
        "NosClk": 1.0,
        "NosClk2": 5,
        "NosTrop": 1.0,
        "NosTrop2": -9,
        "NosSTD": 1.0,
        "NosSTD2": -7,
        "IntPos": 1.0,
        "IntPos2": 2,
        "IntClk": 1.0,
        "IntClk2": 5,
        "IntTrop": 0.5,
        "IntTrop2": 0,
        "IntSTD": 1.0,
        "IntSTD2": 0,
        "IntAmb": 2,
        "IntAmb2": 1,
        "TroGrad": 1,
        "ApMethod": "RINEX",
        "WeMethod": "Elevation Dependent",
        "CodeStd": 3,
        "PhaseStd": 0.003
    }

    # 3. 导入数据
    print("-------------1 正在导入数据-------------")
    data = data_read(files, options)
    data["obs"] = data["obsd"]
    print("-------------1 数据导入完成-------------")

    # 4. 预处理配置与执行
    print("-------------2 正在预处理数据-------------")
    options.update({
        "from": data["obsd"]["epoch"][0, 0],
        "to": data["obsd"]["epoch"][-1, 0],
        "elvangle": 15,
        "CSMw": 1,
        "CSGf": 1,
        "clkjump": 0,
        "codsmth": 1
    })

    data = preprocess(data, options)
    print("-------------2 预处理数据完成-------------")

    # 5. 生成导航模型
    print("-------------3 正在生成导航模型-------------")
    # 添加导航模型所需参数
    options.update({
        "SatClk": 1,
        "SatAPC": 1,
        "RecAPC": 1,
        "RecARP": 1,
        "RelClk": 1,
        "SatWind": 1,
        "AtmTrop": 1,
        "Iono": 1,
        "RelPath": 1,
        "Solid": 1
    })

    model = nmodel(data, options)
    print("-------------3 生成导航模型完成-------------")
    print(f"导航模型形状: {model.shape}")

    # 6. 执行GNSS滤波
    print("-------------4 进行滤波-------------")
    xs_mgnss, kofs_mgnss, pks_mgnss = gnssfilter(model, data, options)

    # 7. 保存定位结果
    np.savetxt('./result/xs.txt', xs_mgnss.T, fmt='%.6f', delimiter='\t')
    print(f"------------4 滤波完成-------------")

    # 8. 评估结果
    ref = np.mean(xs_mgnss[0:3, :], axis=1)
    n, e, u, CT, thrD, rms = evaluate(xs_mgnss, ref.T)

    # 计算综合中误差
    neu_rms = np.sqrt(rms[0, 0] ** 2 + rms[1, 0] ** 2 + rms[2, 0] ** 2)

    # 输出评估结果
    print(f"CT值: {CT}")
    print(f"中误差：{neu_rms:.6f} m")
    print(f"n方向中误差：{rms[0, 0]:.6f} m")
    print(f"e方向中误差：{rms[1, 0]:.6f} m")
    print(f"u方向中误差：{rms[2, 0]:.6f} m")

    # 保存误差数据
    np.savetxt('./result/n.txt', n, fmt='%.8e', delimiter='\t')
    np.savetxt('./result/e.txt', e, fmt='%.8e', delimiter='\t')
    np.savetxt('./result/u.txt', u, fmt='%.8e', delimiter='\t')
    np.savetxt('./result/thrD.txt', thrD, fmt='%.8e', delimiter='\t')

    # 9. 输出最终定位结果
    final_pos_mgnss = xs_mgnss[0:3, -1]
    geo_mgnss = xyz2plh(final_pos_mgnss, 1)

    print("\n============ 定位结果 ============")
    print(
        f"MGNSS_filter函数最终位置 (ECEF): X = {final_pos_mgnss[0]:.3f} m, Y = {final_pos_mgnss[1]:.3f} m, Z = {final_pos_mgnss[2]:.3f} m")
    print(
        f"MGNSS_filter函数最终位置 (大地坐标): 纬度 = {geo_mgnss[0]:.6f}°, 经度 = {geo_mgnss[1]:.6f}°, 高度 = {geo_mgnss[2]:.3f} m")

    # 10. 保存可视化数据
    save_visualization_data(xs_mgnss, n, e, u, thrD, rms, CT)
    print(f"可视化数据已保存到 gnss_visualization_data.json")