import numpy as np
from helper_function import clc_doy, frequencies

def data_read(file, options):

    try:
        obsh, obs = read_obs(file['obs'], options)
        print("Successfully read observation file")
    except Exception as e:
        print(f"Error reading observation file: {e}")
        raise

    try:
        sp3, obsh = read_sp3(file['sp3'], obsh, options)
        print("Successfully read SP3 file")
    except Exception as e:
        print(f"Error reading SP3 file: {e}")
        raise

    # 初始化sat字典，避免未定义问题
    sat = {'sp3': sp3['sp3']}  # 默认使用从sp3文件读取的数据

    # 将三个精密星历进行拼接(106*4*105)
    options['intrp'] = 0  # intrp=0 表示没有前一天或后一天的精密轨道文件
    if file.get('sp3b') and file.get('sp3a'):
        try:
            sp3b, _ = read_sp3(file['sp3b'], obsh, options)
            sp3a, _ = read_sp3(file['sp3a'], obsh, options)

            if (sp3b['sp3'].shape[0] == sp3a['sp3'].shape[0] and
                    sp3b['sp3'].shape[0] == sp3['sp3'].shape[0] and
                    sp3['sp3'].shape[0] == sp3a['sp3'].shape[0]):
                sat['sp3'] = np.vstack((sp3b['sp3'][-5:, :, :], sp3['sp3'], sp3a['sp3'][:5, :, :]))

            options['intrp'] = 1
        except Exception as e:
            print(f"Warning: Error reading additional SP3 files: {e}")
            # 如果读取额外的SP3文件失败，继续使用基本SP3文件

    # 根据options确定钟差改正来自sp3还是clock file
    if options.get('clk_file') == 1:
        clk = read_clock(file['clk'], options)
        obsh['time']['clkinterval'] = options['clk_int']
    else:
        clk = sat['sp3'][:, 3, :]
        obsh['time']['clkinterval'] = obsh['time']['sp3interval']

    atx = read_atx(file['atx'], obsh, options)

    data = {
        'obsh': obsh,
        'obsd': obs,
        'sat': sat,  # 拼接后的精密星历
        'clk': clk,
        'atx': atx,
        'opt': options,
        'files': file
    }

    return data

def read_atx(f_atx, obsh, options):
    """
    Read antenna exchange format file

    Parameters:
    f_atx (str): Path to ATX file
    obsh (dict): Observation header
    options (dict): Processing options

    Returns:
    dict: Antenna information
    """
    try:
        fid = open(f_atx, 'r')
    except Exception as e:
        raise Exception(f'Antex file error: {str(e)}')

    type_val = obsh['sta']['anttype']
    satno = 105
    s_pco = np.full((satno, 3, 2), np.nan)  # unit in file: millimeters
    r_pco = np.full((1, 3, 4), np.nan)

    linenum = 0
    while True:
        tline = fid.readline()
        if not tline:
            break

        linenum += 1
        tline = tline.rstrip()

        if len(tline) >= 61:
            label = tline[60:].strip()
        else:
            continue

        if label == 'START OF ANTENNA':
            tline = fid.readline()
            if not tline:
                break

            linenum += 1
            tline = tline.rstrip()

            if len(tline) >= 61:
                label = tline[60:].strip()
            else:
                continue

            # GPS
            if label == 'TYPE / SERIAL NO' and tline[20] == 'G' and options['system']['gps'] == 1:
                satno = int(tline[21:23])
                if satno > 32:
                    continue

                while label != 'END OF ANTENNA':
                    tline = fid.readline()
                    if not tline:
                        break

                    linenum += 1
                    tline = tline.rstrip()

                    if len(tline) >= 61:
                        label = tline[60:].strip()
                    else:
                        continue

                    if label == 'START OF FREQUENCY' and tline[3:6] == 'G01':  # L1
                        frqno = 0
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'G02':  # L2
                        frqno = 1
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

            # GLONASS
            elif label == 'TYPE / SERIAL NO' and tline[20] == 'R' and options['system']['glo'] == 1:
                satno = 32 + int(tline[21:23])
                if satno > 59:
                    continue

                while label != 'END OF ANTENNA':
                    tline = fid.readline()
                    if not tline:
                        break

                    linenum += 1
                    tline = tline.rstrip()

                    if len(tline) >= 61:
                        label = tline[60:].strip()
                    else:
                        continue

                    if label == 'START OF FREQUENCY' and tline[3:6] == 'R01':  # L1
                        frqno = 0
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'R02':  # L2
                        frqno = 1
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

            # GALILEO
            elif label == 'TYPE / SERIAL NO' and tline[20] == 'E' and options['system']['gal'] == 1:
                satno = 32 + 27 + int(tline[21:23])
                if satno > 95:
                    continue

                while label != 'END OF ANTENNA':
                    tline = fid.readline()
                    if not tline:
                        break

                    linenum += 1
                    tline = tline.rstrip()

                    if len(tline) >= 61:
                        label = tline[60:].strip()
                    else:
                        continue

                    if label == 'START OF FREQUENCY' and tline[3:6] == 'E01':  # L1
                        frqno = 0
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'E05':  # L2
                        frqno = 1
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            s_pco[satno - 1, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

            # Receiver antenna PCV
            elif label == 'TYPE / SERIAL NO' and tline[:20].strip() == type_val:
                while label != 'END OF ANTENNA':
                    tline = fid.readline()
                    if not tline:
                        break

                    linenum += 1
                    tline = tline.rstrip()

                    if len(tline) >= 61:
                        label = tline[60:].strip()
                    else:
                        continue

                    if label == 'START OF FREQUENCY' and tline[3:6] == 'G01':  # L1-GPS
                        frqno = 0
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            r_pco[0, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'G02':  # L2-GPS
                        frqno = 1
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            r_pco[0, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'R01':  # L1-GLO
                        frqno = 2
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            r_pco[0, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

                    elif label == 'START OF FREQUENCY' and tline[3:6] == 'R02':  # L2-GLO
                        frqno = 3
                        tline = fid.readline()
                        if not tline:
                            break

                        linenum += 1
                        tline = tline.rstrip()

                        if len(tline) >= 61:
                            label = tline[60:].strip()
                        else:
                            continue

                        if label == 'NORTH / EAST / UP':
                            r_pco[0, :, frqno] = np.array([float(x) for x in tline.split()[:3]])

    fid.close()

    atx = {
        'sat': {'pco': s_pco / 1000},
        'rcv': {'pco': r_pco / 1000, 'type': type_val}
    }

    return atx


def read_clock(f_clock, options):
    try:
        fid = open(f_clock, 'r')
    except Exception as e:
        raise Exception(f'Clock file error: {str(e)}')

    sno = 105
    # 对于整数结果，不需要向上取整
    clk_tn = int(86400 / options['clk_int'])  # 如果clk_int是30，结果就是2880
    clk = np.full((clk_tn, sno), np.nan)

    for tline in fid:
        tline = tline.strip()
        flag = 0

        if tline.startswith('AS G') and options['system']['gps'] == 1:
            flag = 1
            line = np.array([float(x) for x in tline[4:].split()])
            satno = int(line[0])
            if satno > 32:
                continue

        elif tline.startswith('AS R') and options['system']['glo'] == 1:
            flag = 1
            line = np.array([float(x) for x in tline[4:].split()])
            satno = 32 + int(line[0])
            if satno > 59:
                continue

        elif tline.startswith('AS E') and options['system']['gal'] == 1:
            flag = 1
            line = np.array([float(x) for x in tline[4:].split()])
            satno = 32 + 27 + int(line[0])
            if satno > 95:
                continue

        elif tline.startswith('AS C') and options['system'].get('bds', 0) == 1:
            flag = 1
            line = np.array([float(x) for x in tline[4:].split()])
            satno = 32 + 27 + 36 + int(line[0])
            if satno > 105:
                continue

        if flag == 1:
            # 自动向下取整用于索引
            epoch_float = (line[4] * 3600 + line[5] * 60 + line[6]) / options['clk_int'] + 1
            epoch = int(epoch_float)  # 将索引转为整数
            if 1 <= epoch <= clk_tn:  # 确保索引在有效范围内
                clk[epoch - 1, satno - 1] = line[8]  # 调整为Python的0基索引

    fid.close()
    return clk


def read_obs(f_obs, options):

    print(f"Reading observation file: {f_obs}")
    try:
        fid = open(f_obs, 'r')
    except Exception as e:
        raise Exception(f'Observation file error: {str(e)}')

    # Initialize observation header
    obsh = {
        'seq': {
            'gps': np.zeros(8),  # C1 C2 C5 P1 P2 L1 L2 L5
            'glo': np.zeros(6),  # C1 C2 P1 P2 L1 L2
            'gal': np.zeros(4)  # C1 C5 L1 L5
        },
        'rinex': {},
        'sta': {},
        'time': {},
        'sat': {}
    }

    # Read header
    while True:
        tline = fid.readline()
        if not tline:
            break

        tline = tline.rstrip()

        if len(tline) >= 61:
            label = tline[60:].strip()
        else:
            continue

        if label == 'RINEX VERSION / TYPE':
            if tline[20] == 'O':
                obsh['rinex']['ver'] = tline[:10].strip()
                obsh['rinex']['type'] = tline[20]
                obsh['sat']['system'] = tline[40]
            else:
                raise Exception('It is not an observation file!')

        elif label == 'APPROX POSITION XYZ':
            obsh['sta']['pos'] = np.array([float(x) for x in tline[:60].split()])

        elif label == 'REC # / TYPE / VERS':
            obsh['sta']['recsno'] = tline[:20].strip()
            obsh['sta']['rectype'] = tline[20:40].strip()
            obsh['sta']['version'] = tline[40:60].strip()

        elif label == 'MARKER NAME':
            obsh['sta']['name'] = tline[:60].strip()

        elif label == 'MARKER NUMBER':
            obsh['sta']['marker'] = tline[:20].strip()

        elif label == 'ANT # / TYPE':
            obsh['sta']['antsno'] = tline[:20].strip()
            obsh['sta']['anttype'] = tline[20:40].strip()

        elif label == 'ANTENNA: DELTA H/E/N':
            obsh['sta']['antdel'] = np.array([float(x) for x in tline[:60].split()])

        elif label == '# / TYPES OF OBSERV':
            obsh['obsno'] = int(tline[:6])
            j = 10
            for i in range(obsh['obsno']):
                if j > 58:
                    tline = fid.readline()
                    j = 10

                obstype = tline[j:j + 2].strip()

                if obstype == 'C1':
                    obsh['seq']['gps'][0] = i
                    obsh['seq']['glo'][0] = i
                    obsh['seq']['gal'][0] = i
                elif obstype == 'C2':
                    obsh['seq']['gps'][1] = i
                    obsh['seq']['glo'][1] = i
                elif obstype == 'C5':
                    obsh['seq']['gps'][2] = i
                    obsh['seq']['gal'][1] = i
                elif obstype == 'P1':
                    obsh['seq']['gps'][3] = i
                    obsh['seq']['glo'][2] = i
                elif obstype == 'P2':
                    obsh['seq']['gps'][4] = i
                    obsh['seq']['glo'][3] = i
                elif obstype == 'L1':
                    obsh['seq']['gps'][5] = i
                    obsh['seq']['glo'][4] = i
                    obsh['seq']['gal'][2] = i
                elif obstype == 'L2':
                    obsh['seq']['gps'][6] = i
                    obsh['seq']['glo'][5] = i
                elif obstype == 'L5':
                    obsh['seq']['gps'][7] = i
                    obsh['seq']['gal'][3] = i

                j += 6

        elif label == 'INTERVAL':
            try:
                interval_str = tline[:10].strip()
                obsh['time']['obsinterval'] = float(interval_str)
            except ValueError:
                print(f"Warning: Could not parse interval '{interval_str}', using default value 30.0")
                obsh['time']['obsinterval'] = 30.0

        elif label == 'TIME OF FIRST OBS':
            obsh['time']['first'] = np.array([int(float(x)) for x in tline[:43].split()])
            obsh['time']['system'] = tline[43:60].strip()

        elif label == 'TIME OF LAST OBS':
            obsh['time']['last'] = np.array([int(float(x)) for x in tline[:43].split()])

        elif label == 'LEAP SECONDS':
            obsh['time']['leap'] = int(tline.split()[0])

        elif label == 'END OF HEADER':
            break

    print(f"Finished reading header. Found {obsh['obsno']} observation types.")
    # Calculate year day (DOY)
    doy = clc_doy(obsh['time']['first'][0], obsh['time']['first'][1], obsh['time']['first'][2])
    obsh['time']['doy'] = doy

    # Read observation data
    t_start = obsh['time']['first'][3] * 3600 + obsh['time']['first'][4] * 60 + obsh['time']['first'][5]
    t_end = obsh['time']['last'][3] * 3600 + obsh['time']['last'][4] * 60 + obsh['time']['last'][5]
    t_obsnum = int((t_end - t_start) / obsh['time']['obsinterval'] + 1)

    d = {
        'year': obsh['time']['first'][0],
        'mon': obsh['time']['first'][1],
        'day': obsh['time']['first'][2]
    }

    if d['year'] < 2000:
        d['year'] = d['year'] - 1900
    else:
        d['year'] = d['year'] - 2000

    sno = 105
    p1s = np.full((t_obsnum, sno), np.nan)  # p1
    p2s = np.full((t_obsnum, sno), np.nan)  # p2
    l1s = np.full((t_obsnum, sno), np.nan)  # L1
    l2s = np.full((t_obsnum, sno), np.nan)  # L2
    epoch = np.full((t_obsnum, 1), np.nan)  # Store epoch time
    sats = np.full((t_obsnum, sno), np.nan)  # Store observed satellites
    state = np.zeros((t_obsnum, sno))  # Set to 1 if satellite has L1/L2/P1/P2 observations

    _, wavl = frequencies()

    linenum = 0
    epochn = 0

    while True:
        tline = fid.readline()
        if not tline:
            break

        linenum += 1
        tline = tline.rstrip()

        # Read yymmdd.ss/epoch flag/number of satellite
        if len(tline) < 32:
            try:
                ep = np.array([float(x) for x in tline.split()])
            except:
                continue
        else:
            try:
                ep = np.array([float(x) for x in tline[:32].split()])
            except:
                continue

        if len(ep) == 8 and ep[0] == d['year'] and ep[1] == d['mon'] and ep[2] == d['day'] and len(tline) > 33:
            epochn += 1  # Count the epoch being read
            epocht = ep[3] * 3600 + ep[4] * 60 + ep[5]
            epoch[epochn - 1, 0] = epocht
            sats_no = int(ep[7])
            j = 32

            for i in range(sats_no):
                if j > 67:
                    tline = fid.readline()
                    j = 32

                satline = tline[j:j + 3].strip()

                if 'G' in satline or 'R' in satline or 'E' in satline or ' ' in satline:
                    satline = satline.replace('G', '1')  # 32 satellites
                    satline = satline.replace(' ', '1')  # blank: GPS
                    satline = satline.replace('R', '2')  # 27 satellites
                    satline = satline.replace('E', '3')  # 36 satellites

                try:
                    satlines = int(satline)
                except:
                    j += 3
                    continue

                # Convert PRN to satellite number stored in sats matrix
                if 100 < satlines < 133 and options['system']['gps'] == 1:
                    sats[epochn - 1, i] = satlines - 100
                elif 200 < satlines < 227 and options['system']['glo'] == 1:
                    sats[epochn - 1, i] = satlines - 200 + 32
                elif 300 < satlines < 336 and options['system']['gal'] == 1:
                    sats[epochn - 1, i] = satlines - 300 + 32 + 27

                j += 3

            for i in range(sats_no):
                if np.isnan(sats[epochn - 1, i]):
                    continue

                sat_idx = int(sats[epochn - 1, i])

                ls = np.full(obsh['obsno'], np.nan)  # 用来存储文件中一个卫星所有的观测值
                count = 1

                if sat_idx > 95:
                    for k in range(int(np.ceil(obsh['obsno'] / 5))):
                        fid.readline()
                        linenum += 1
                else:
                    for k in range(int(np.ceil(obsh['obsno'] / 5))):  # 循环对应卫星每行观测值
                        tline = fid.readline()
                        linenum += 1
                        tline = tline.rstrip()

                        for n in range(5):  # 循环每行中的每个观测值
                            st = 16 * n
                            fn = 16 * (n + 1) - 2

                            if fn <= len(tline):  # 检查是否在行边界内
                                try:
                                    tls = float(tline[st:fn])
                                    ls[(5 * (k)) + n] = tls
                                except:
                                    ls[(5 * (k)) + n] = 0
                            else:
                                ls[(5 * (k)) + n] = 0  # 将空观测值设为0

                            count += 1
                            if count > obsh['obsno']:
                                break

                if sat_idx < 33:  # GPS
                    # p1
                    p1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gps'][3])]
                    # p2
                    p2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gps'][4])]
                    # L1
                    l1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gps'][5])] * wavl[sat_idx - 1, 0]
                    # L2
                    l2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gps'][6])] * wavl[sat_idx - 1, 1]

                elif sat_idx < 60:  # GLONASS
                    # p1
                    p1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['glo'][2])]
                    # p2
                    p2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['glo'][3])]
                    # L1
                    l1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['glo'][4])] * wavl[sat_idx - 1, 0]
                    # L2
                    l2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['glo'][5])] * wavl[sat_idx - 1, 1]

                elif sat_idx < 96:  # GALILEO
                    # c1
                    p1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gal'][0])]
                    # c5
                    p2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gal'][1])]
                    # L1
                    l1s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gal'][2])] * wavl[sat_idx - 1, 0]
                    # L5
                    l2s[epochn - 1, sat_idx - 1] = ls[int(obsh['seq']['gal'][3])] * wavl[sat_idx - 1, 1]

    all_obs = p1s + p2s + l1s + l2s
    state[~np.isnan(all_obs)] = 1

    if t_obsnum > epochn:
        p1s = p1s[:epochn, :]
        p2s = p2s[:epochn, :]
        l1s = l1s[:epochn, :]
        l2s = l2s[:epochn, :]
        epoch = epoch[:epochn, :]
        sats = sats[:epochn, :]
        state = state[:epochn, :]

    obs = {
        'p1': p1s,
        'p2': p2s,
        'l1': l1s,
        'l2': l2s,
        'epoch': epoch,
        'stats': sats,
        'st': state
    }
    fid.close()

    return obsh, obs


def read_sp3(f_orbit, obsh, options):
    """
    Read SP3 precise orbit file

    Parameters:
    f_orbit (str): Path to SP3 file
    obsh (dict): Observation header
    options (dict): Processing options

    Returns:
    tuple: (sp3_struct, obsh) SP3 data and updated observation header
    """
    try:
        fid = open(f_orbit, 'r')
    except Exception as e:
        raise Exception(f'SP3 file error: {str(e)}')

    sno = 105
    sp3 = np.full((96, 4, sno), np.nan)

    # Read SP3 precise ephemeris
    satn = 0

    while True:
        line = fid.readline()
        if not line:
            break

        line = line.rstrip()

        # Read file header
        if line.startswith('#') and not line.startswith('##'):
            date = np.array([float(x) for x in line[3:13].split()])
            epochn = int(line[32:39])  # Number of epochs
            sp3 = np.full((epochn, 4, sno), np.nan)

            sp3_struct = {
                'date': date,
                'epochn': epochn
            }

        if line.startswith('##'):
            sp3int = float(line[24:38])
            obsh['time']['sp3interval'] = sp3int
            sp3_struct['sp3interval'] = sp3int

        if line.startswith('+'):
            try:
                temp = int(line[4:6])
                if not np.isnan(temp):  # Prevent matching other lines starting with '+'
                    satn = temp
            except:
                pass

        # Read file body
        if line.startswith('*'):
            try:
                epoch = np.array([float(x) for x in line[2:].split()])

                if epoch[0] != date[0] or epoch[1] != date[1] or epoch[2] != date[2]:
                    continue

                epno = int((epoch[3] * 3600 + epoch[4] * 60 + epoch[5]) / sp3int)

                for i in range(satn):
                    line = fid.readline()
                    if not line:
                        break

                    line = line.rstrip()

                    if line[1] == 'G' and options['system']['gps'] == 1:  # GPS
                        satno = int(line[2:4])
                        if satno > 32:
                            continue

                    elif line[1] == 'R' and options['system']['glo'] == 1:  # GLONASS
                        satno = 32 + int(line[2:4])
                        if satno > 59:
                            continue

                    elif line[1] == 'E' and options['system']['gal'] == 1:  # Galileo
                        satno = 58 + int(line[2:4])
                        if satno > 95:
                            continue
                    elif line[1] == 'C' and options['system']['bds'] == 1:  # BeiDou
                        satno = 88 + int(line[2:4])
                        if satno > 105:
                            continue
                    else:
                        continue

                    tempdata = np.array([float(x) for x in line[4:].split()])

                    # Writing part
                    sp3[epno, 0:3, satno - 1] = tempdata[0:3] * 1000  # X/Y/Z km->m
                    sp3[epno, 3, satno - 1] = tempdata[3] * 10 ** -6  # second microsec-sec
            except:
                continue

    obsh['sat']['sp3'] = sp3
    sp3_struct['sp3'] = sp3
    fid.close()

    return sp3_struct, obsh


