import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyproj


# NOTE Test1/Position_PDR_5.csv has bad timestamps
TESTS = [["Test1",
         {
             "est": "Position_PDR_3.csv",
             "inp": "5th_floor_03.csv",
             "argt": "pose_3.csv",
             "bpgt": "pose_backpack_3.csv",
         }],
         ["Test2",
         {
             "est": "Position_EKF_PDR_5.csv",
             "inp": "5th_floor_05.csv",
             "argt": "pose_5.csv",
             "bpgt": "pose_backpack_5.csv",
         }],
         ["Test",
         {
              "est": "position_all_5_2022_07_20.csv",
              "inp": "5th_floor_05.csv",
              "argt": "pose_5.csv",
              "bpgt": "pose_backpack_5.csv",
         }]
]


def latlon_to_ne(lat, lon):
    tr = pyproj.Transformer.from_crs("epsg:4326", "epsg:5186")
    north, east = tr.transform(lat, lon)
    return north, east


def load_test_data_v1(test):
    """Loads PDR test data.
       Parameters:
         test: a list whose first element is a path to the data
               and second element is a dictionary with
               'est' - estimations file name  (? type, 5 columns)
               'inp' - input data file name 
               'argt' - ARCore data file name
               'bpgt' - BackPack data file name
       Returns:
               Data as Pandas DataFrame objects
               'estd' - estimations data
               'inpd' - input data
               'argtd' - ARCore data
               'bpgtd' - BackPack data
        
    """
    dpath = test[0]
    # data file names
    estfn = os.path.join(dpath, test[1]['est'])
    inpfn = os.path.join(dpath, test[1]['inp'])
    argtfn = os.path.join(dpath, test[1]['argt'])
    bpgtfn = os.path.join(dpath, test[1]['bpgt'])

    # data column names
    estcn = ["ts",
             "f_posx","f_posy",
             "g_posx","g_posy"]
    inpcn = ["ts",
             "tx","ty","tz",
             "qx","qy","qz","qw",
             "acc_x","acc_y","acc_z",
             "gy_x","gy_y","gy_z",
             "mg_x","mg_y","mg_z",
             "pres",
             "bp_lon","bp_lat"]
    argtcn = ["ts",
              "x", "y", "z",
              "qw", "qx", "qy", "qz"]
    bpgtcn = ["tsbp", "ts",
              "lon", "lat", "z",
              "roll", "pitch", "yaw", "dummy"]

    # data
    estd = pd.read_csv(estfn, delimiter=',', header=None, names=estcn)
    inpd = pd.read_csv(inpfn, delimiter=',', header=0, names=inpcn)
    argtd = pd.read_csv(argtfn, delimiter=',', header=None, names=argtcn)
    bpgtd = pd.read_csv(bpgtfn, delimiter=',', header=None, names=bpgtcn)

    # change ts to datetime type
    estd.ts = pd.to_datetime(estd.ts, unit='s')
    inpd.ts = pd.to_datetime(inpd.ts, unit='s')
    argtd.ts = pd.to_datetime(argtd.ts)
    bpgtd.ts = pd.to_datetime(bpgtd.ts)
    # we do not care about bpgtd.tsbp

    # add north/east to input data
    north, east = latlon_to_ne(inpd.bp_lat, inpd.bp_lon)
    inpd['north'] = north
    inpd['east'] = east
    inpd['north_rel'] = north - north[0]
    inpd['east_rel'] = east - east[0]
    
    return estd, inpd, argtd, bpgtd



def load_test_data_v2(test):
    """Loads PDR test data.
       Parameters:
         test: a list whose first element is a path to the data
               and second element is a dictionary with
               'est' - estimations file name (EKF type, 3 columns)
               'inp' - input data file name
               'argt' - ARCore data file name
               'bpgt' - BackPack data file name
       Returns:
               Data as Pandas DataFrame objects
               'estd' - estimations data
               'inpd' - input data
               'argtd' - ARCore data
               'bpgtd' - BackPack data
        
    """
    dpath = test[0]
    # data file names
    estfn = os.path.join(dpath, test[1]['est'])
    inpfn = os.path.join(dpath, test[1]['inp'])
    argtfn = os.path.join(dpath, test[1]['argt'])
    bpgtfn = os.path.join(dpath, test[1]['bpgt'])

    # data column names
    estcn = ["ts",
             "posx","posy"]
    inpcn = ["ts",
             "tx","ty","tz",
             "qx","qy","qz","qw",
             "acc_x","acc_y","acc_z",
             "gy_x","gy_y","gy_z",
             "mg_x","mg_y","mg_z",
             "pres",
             "bp_lon","bp_lat"]
    argtcn = ["ts",
              "x", "y", "z",
              "qw", "qx", "qy", "qz"]
    bpgtcn = ["tsbp", "ts",
              "lon", "lat", "z",
              "roll", "pitch", "yaw", "dummy"]

    # data
    estd = pd.read_csv(estfn, delimiter=',', header=None, names=estcn)
    inpd = pd.read_csv(inpfn, delimiter=',', header=0, names=inpcn)  # ignore csv header
    argtd = pd.read_csv(argtfn, delimiter=',', header=None, names=argtcn)
    bpgtd = pd.read_csv(bpgtfn, delimiter=',', header=None, names=bpgtcn)

    # change ts to datetime type
    estd.ts = pd.to_datetime(estd.ts, unit='s')
    inpd.ts = pd.to_datetime(inpd.ts, unit='s')
    argtd.ts = pd.to_datetime(argtd.ts)
    bpgtd.ts = pd.to_datetime(bpgtd.ts)
    # we do not care about bpgtd.tsbp

    # add north/east to input data
    north, east = latlon_to_ne(inpd.bp_lat, inpd.bp_lon)
    inpd['north'] = north
    inpd['east'] = east
    inpd['north_rel'] = north - north[0]
    inpd['east_rel'] = east - east[0]
    
    return estd, inpd, argtd, bpgtd


def load_test_data_v3(test):
    """Loads PDR test data.
       Parameters:
         test: a list whose first element is a path to the data
               and second element is a dictionary with
               'est' - estimations file name (EKF type, 7 columns)
               'inp' - input data file name
               'argt' - ARCore data file name
               'bpgt' - BackPack data file name
       Returns:
               Data as Pandas DataFrame objects
               'estd' - estimations data
               'inpd' - input data
               'argtd' - ARCore data
               'bpgtd' - BackPack data
        
    """
    dpath = test[0]
    # data file names
    estfn = os.path.join(dpath, test[1]['est'])
    inpfn = os.path.join(dpath, test[1]['inp'])
    argtfn = os.path.join(dpath, test[1]['argt'])
    bpgtfn = os.path.join(dpath, test[1]['bpgt'])

    # data column names
    estcn = ["ts",
             "ahrs_posx","ahrs_posy",
             "pca_posx","pca_posy",
             "ekf_posx","ekf_posy"]
    inpcn = ["ts",
             "tx","ty","tz",
             "qx","qy","qz","qw",
             "acc_x","acc_y","acc_z",
             "gy_x","gy_y","gy_z",
             "mg_x","mg_y","mg_z",
             "pres",
             "bp_lon","bp_lat"]
    argtcn = ["ts",
              "x", "y", "z",
              "qw", "qx", "qy", "qz"]
    bpgtcn = ["tsbp", "ts",
              "lon", "lat", "z",
              "roll", "pitch", "yaw", "dummy"]

    # data
    estd = pd.read_csv(estfn, delimiter=',', header=None, names=estcn)
    inpd = pd.read_csv(inpfn, delimiter=',', header=0, names=inpcn)  # ignore csv header
    argtd = pd.read_csv(argtfn, delimiter=',', header=None, names=argtcn)
    bpgtd = pd.read_csv(bpgtfn, delimiter=',', header=None, names=bpgtcn)

    # change ts to datetime type
    estd.ts = pd.to_datetime(estd.ts, unit='s')
    inpd.ts = pd.to_datetime(inpd.ts, unit='s')
    argtd.ts = pd.to_datetime(argtd.ts)
    bpgtd.ts = pd.to_datetime(bpgtd.ts)
    # we do not care about bpgtd.tsbp

    # add north/east to input data
    north, east = latlon_to_ne(inpd.bp_lat, inpd.bp_lon)
    inpd['north'] = north
    inpd['east'] = east
    inpd['north_rel'] = north - north[0]
    inpd['east_rel'] = east - east[0]
    
    return estd, inpd, argtd, bpgtd

## plan
# - load data
# - estimate position errors in respect to arcore GT
#   err.f_ar:  (estd.f_posx, estd.f_posy) =?= (inpd.tz, inpd.ty); (for closest timestamps)
#   err.g_ar:  (estd.g_posx, estd.g_posy) =?= (inpd.tz, inpd.ty); (for closest timestamps)  
# - estimate position errors in respect to backpack GT
#   err.f_bp:  (estd.f_posx, estd.f_posy) =?= (-inpd.north_rel, inpd.east_rel); (for closest timestamps)
#   err.g_bp:  (estd.g_posx, estd.g_posy) =?= (-inpd.north_rel, inpd.east_rel); (for closest timestamps)  
#             where, inpd.north_rel = inpd.north - inpd.north[0]
#                    inpd.east_rel = inpd.east - inpd.east[0]


def compute_err(est_pos, gt_pos):
    """Compute position errors (Euclidean distance)
       est_pos is a (N,2) numpy array with columns: x, y
       gt_pos is a (N,2) numpy array with columns: x, y
    """
    err = np.linalg.norm((est_pos - gt_pos), axis=1)
    return err

def get_corresponding_gt_pos(ts, gt):
    """Find gt poositions cosest in time to timestamp
       ts is a (N,) pandas Series with column: ts
       gt is a (M,3) pandas DataFrame with columns: ts, [positions] 
    """
    gti = gt.set_index(gt.ts, drop=True)
    idx = [gti.index.get_loc(t, method='nearest') for t in ts]
    gt_cor = gt.iloc[idx, :]
    return gt_cor


# Using Test2 since Test1/Position_PDR_5.csv has bad timestamps
# estd, inpd, argtd, bpgtd = load_test_data_v2(TESTS[1])
estd, inpd, argtd, bpgtd = load_test_data_v3(TESTS[2])



gt = inpd[["ts", "tz","tx","north_rel","east_rel"]]
gt_cor = get_corresponding_gt_pos(estd.ts, gt)


# ARCore gt error
est_pos_ahrs = estd[["ahrs_posx","ahrs_posy"]].to_numpy()
est_pos_pca = estd[["pca_posx","pca_posy"]].to_numpy()
est_pos_ekf = estd[["ekf_posx","ekf_posy"]].to_numpy()
argt_pos = -1* gt_cor[["tz","tx"]].to_numpy()
ar_err_ahrs = compute_err(est_pos_ahrs, argt_pos)

ar_err_pca = compute_err(est_pos_pca, argt_pos)
ar_err_ekf = compute_err(est_pos_ekf, argt_pos)
ar_err_CDF_ahrs = np.array([np.percentile(ar_err_ahrs, p) for p in range(101)])
ar_err_CDF_pca = np.array([np.percentile(ar_err_pca, p) for p in range(101)])
ar_err_CDF_ekf = np.array([np.percentile(ar_err_ekf, p) for p in range(101)])



fplot=0

if fplot == 1:
    plt.plot(est_pos_ahrs[:,0], est_pos_ahrs[:,1], label='PDR_ahrs')
    plt.plot(est_pos_pca[:,0], est_pos_pca[:,1], label='PDR_pca')
    plt.plot(est_pos_ekf[:,0], est_pos_ekf[:,1], label='PDR_ekf')
    plt.plot(argt_pos[:,0], argt_pos[:,1], label='ARCore t_pos')
    plt.axis('equal')
    plt.title('ARCore trajectory comparison')
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.legend()
    plt.grid()
    plt.show()

if fplot == 2:
    plt.plot(ar_err_CDF_ahrs, range(101), label='CDF_ahrs')
    plt.plot(ar_err_CDF_pca, range(101), label='CDF_pca')
    plt.plot(ar_err_CDF_ekf, range(101), label='CDF_ekf')
    plt.grid()
    plt.xlabel('pos. error, m')
    plt.ylabel('percentile')
    plt.legend()
    plt.title('Error CDF (ARCore gt)')
    plt.show()

np.mean(ar_err_ahrs)
np.mean(ar_err_pca)
np.mean(ar_err_ekf)

Traj_RMSE_ahrs =np.sqrt(np.mean(np.square(np.linalg.norm(argt_pos-est_pos_ahrs,axis=-1))))
Traj_RMSE_pca =np.sqrt(np.mean(np.square(np.linalg.norm(argt_pos-est_pos_pca,axis=-1))))
Traj_RMSE_ekf =np.sqrt(np.mean(np.square(np.linalg.norm(argt_pos-est_pos_ekf,axis=-1))))

# backpack gt error
est_pos_ahrs = estd[["ahrs_posx","ahrs_posy"]].to_numpy()
est_pos_pca = estd[["pca_posx","pca_posy"]].to_numpy()
est_pos_ekf = estd[["ekf_posx","ekf_posy"]].to_numpy()

# est_pos = estd[["posx","posy"]].to_numpy()

bpgt_pos = gt_cor[["north_rel","east_rel"]].to_numpy()
bpgt_pos[:,0] = -1* bpgt_pos[:,0]
bp_err_ahrs = compute_err(est_pos_ahrs, bpgt_pos)
bp_err_pca = compute_err(est_pos_pca, bpgt_pos)
bp_err_ekf = compute_err(est_pos_ekf, bpgt_pos)

# bp_err = compute_err(est_pos, bpgt_pos)
# bp_err_CDF = np.array([np.percentile(bp_err, p) for p in range(101)])
bp_err_CDF_ahrs = np.array([np.percentile(bp_err_ahrs, p) for p in range(101)])
bp_err_CDF_pca = np.array([np.percentile(bp_err_pca, p) for p in range(101)])
bp_err_CDF_ekf = np.array([np.percentile(bp_err_ekf, p) for p in range(101)])



fplot=4

if fplot == 3:
    plt.plot(est_pos_ahrs[:,0], est_pos_ahrs[:,1], label='PDR_ahrs')
    plt.plot(est_pos_pca[:,0], est_pos_pca[:,1], label='PDR_pca')
    plt.plot(est_pos_ekf[:,0], est_pos_ekf[:,1], label='PDR_ekf')
    plt.plot(bpgt_pos[:,0], bpgt_pos[:,1], label='Backpack t_pos')
    plt.axis('equal')
    plt.title('backpack trajectory comparison')
    plt.xlabel('x, m')
    plt.ylabel('y, m')
    plt.legend()
    plt.grid()
    plt.show()

if fplot == 4:
    plt.plot(bp_err_CDF_ahrs, range(101), label='CDF_ahrs')
    plt.plot(bp_err_CDF_pca, range(101), label='CDF_pca')
    plt.plot(bp_err_CDF_ekf, range(101), label='CDF_ekf')
    plt.grid()
    plt.xlabel('pos. error, m')
    plt.ylabel('percentile')
    plt.legend()
    plt.title('Error CDF (Backpack gt)')
    plt.show()
np.mean(bp_err_CDF_ahrs)
np.mean(bp_err_CDF_pca)
np.mean(bp_err_CDF_ekf)

Traj_RMSE_ahrs =np.sqrt(np.mean(np.square(np.linalg.norm(bpgt_pos-est_pos_ahrs,axis=-1))))
Traj_RMSE_pca =np.sqrt(np.mean(np.square(np.linalg.norm(bpgt_pos-est_pos_pca,axis=-1))))
Traj_RMSE_ekf =np.sqrt(np.mean(np.square(np.linalg.norm(bpgt_pos-est_pos_ekf,axis=-1))))
# plt.plot(est_pos[:,0], est_pos[:,1], label='PDR')
# # plt.scatter(est_pos[:,0], est_pos[:,1])
# plt.plot(bpgt_pos[:,0], bpgt_pos[:,1], label='BackPack gt_pos')
# # plt.scatter(bpgt_pos[:,0], bpgt_pos[:,1])
# plt.axis('equal')
# plt.title('BackPack trajectory comparison')
# plt.xlabel('x, m')
# plt.ylabel('y, m')
# plt.legend()
# plt.grid()
# plt.show()

# plt.plot(bp_err_CDF, range(101))
# plt.xlabel('pos error, m')
# plt.ylabel('percentile')
# plt.title('Error CDF (BackPack gt)')
# plt.grid()
# plt.show()
