# -*- coding: utf-8 -*-
'''
Hyytiälä pine forest, Apr-now 2016
Extract, combine, and correct chamber sensor data

Revision
--------
05/26/2016, W.S.
The two PAR sensors are now called 'PAR_1' and 'PAR_2', because their 
association with the chambers changes during the campaign.

08/29/2016, W.S.
Temporary modification (line 82: "continue") to skip the day 28 Aug 2016 for missing data.
'''
import numpy as np
# from scipy import stats, signal
import pandas as pd
# import matplotlib.pyplot as plt
import os, glob, datetime, linecache, copy

pd.options.display.float_format = '{:.2f}'.format  # pandas dataframe displays float with 2 decimal places

flag_run_recent_days = False
# if False, run it over all the data; if True, run only the recent 3 days

'''
Functions
---------
'''
def IQR_bounds_func(x):
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25,75])
        IQR = q3 - q1
        return(q1 - 2 * IQR, q3 + 5 * IQR)
    else:
        return(np.nan, np.nan)

print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))

# local time is UTC+2
doy_today = (datetime.datetime.utcnow() - datetime.datetime(2016,1,1)).total_seconds() / 86400. + 2./24.

if flag_run_recent_days:
    doy_start = np.int(doy_today - 3.)
else:
    doy_start = 97 # the campaign starts on 04/07/2016

sensor_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/sensor_data/'

# get file list of sensor data
cop_flist = glob.glob(sensor_dir + '/sm_cop/*.cop') # leaf chamber sensors
mpr_flist = glob.glob(sensor_dir + '/sm_mpr/*.mpr') # soil chamber sensors

# cop file columns (starting from 0):
# 0 - time; 1 - PAR (large leaf chamber); 2 - PAR (small leaf chamber B)
# 8 - ambient T; 10 - T_ch from small leaf chamber A (T_ch1); 
# 11 - T_ch from small leaf chamber B (T_ch2); 12 - T_ch from the large leaf chamber (T_ch3)

# mpr file columns (starting from 0):
# 0 - time; 5 - soil chamber 1 (T_ch4); 6 - soil chamber 2 (T_ch5)
# 7 - soil chamber 3 (T_ch6)
doy_start = np.int(doy_today) - 7  # modified 08/29/2016
for doy in range(doy_start,np.int(doy_today)):
    run_date_str = (datetime.datetime(2016,1,1) + datetime.timedelta(doy+0.5)).strftime("%y%m%d")
    current_cop_files = [s for s in cop_flist if run_date_str in s]
    current_mpr_files = [s for s in mpr_flist if run_date_str in s]
    # reading leaf chamber sensor data
    cop_data = np.array([])
    if len(current_cop_files) > 0:
        for loop_num in range(len(current_cop_files)):
            cop_data_loaded = np.genfromtxt(current_cop_files[loop_num], usecols=(0,1,2,8,10,11,12), 
              dtype=[('time', '|S16'), ('PAR_1', 'f8'), ('PAR_2', 'f8'), ('T_amb', 'f8'), 
                ('T_ch1', 'f8'), ('T_ch2', 'f8'), ('T_ch3', 'f8')],
              invalid_raise=False)
            if cop_data.size:
                cop_data = np.concatenate((cop_data, cop_data_loaded))
            else:
                cop_data = np.copy(cop_data_loaded)
            del(cop_data_loaded)
    else:
        print('File for leaf chamber sensor data not found on day 20' + run_date_str)
        continue
    # reading soil chamber sensor data
    mpr_data = np.array([])
    if len(current_mpr_files) > 0:
        for loop_num in range(len(current_mpr_files)):
            mpr_data_loaded = np.genfromtxt(current_mpr_files[loop_num], usecols=(0,5,6,7), 
              dtype=[('time', '|S16'), ('T_ch4', 'f8'), ('T_ch5', 'f8'), ('T_ch6', 'f8')],
              invalid_raise=False)
            if mpr_data.size:
                mpr_data = np.concatenate((mpr_data, mpr_data_loaded))
            else:
                mpr_data = np.copy(mpr_data_loaded)
            del(mpr_data_loaded)
    else:
        print('File for soil chamber sensor data not found on day 20' + run_date_str)
    # parse datetime strings
    cop_doy = np.zeros(cop_data.shape[0]) * np.nan
    for i in range(cop_doy.size):
        if len(cop_data[i]['time']) > 0:
            cop_doy[i] = (datetime.datetime.strptime(cop_data[i,]['time'], '%Y%m%d%H%M%S') - datetime.datetime(2016, 1, 1, 0, 0)).total_seconds() / 86400.
        else:
            cop_doy[i] = np.nan
    mpr_doy = np.zeros(mpr_data.shape[0]) * np.nan
    for i in range(mpr_doy.size):
        if len(mpr_data[i]['time']) > 0:
            mpr_doy[i] = (datetime.datetime.strptime(mpr_data[i,]['time'], '%Y%m%d%H%M%S') - datetime.datetime(2016, 1, 1, 0, 0)).total_seconds() / 86400.
        else:
            mpr_doy[i] = np.nan
    # --- corrections ---
    cop_data['PAR_2'] *= 200.
    cop_data['PAR_1'] *= 210.  # use 210 as a rough estimate for now. needs to be re-calibrated
    cop_data['T_ch1'] = cop_data['T_ch1'] * 0.94 + 0.75
    cop_data['T_ch2'] = cop_data['T_ch2'] * 0.96 - 0.20
    if doy < 103:
        # before 04/13/2016, not including that day
        cop_data['T_ch3'] = cop_data['T_ch3'] * 0.98 - 0.89
    else:
        # TC in the large leaf chamber reinstalled 04/13/2016 11:20am
        # before that time on 04/13/2016, temp data were corrupt for this chamber, which will be masked
        cop_data['T_ch3'] = cop_data['T_ch3'] * 0.97 - 0.39
    # --- masking corrupt data ---
    # T_ch3: between 04/08/2016 and 04/13/2016, T_ch3 is corrupt
    if doy == 98:
        break_pt = (datetime.datetime(2016, 4, 8, 9, 33, 41) - datetime.datetime(2016,1,1)).total_seconds() / 86400.
        cop_data['T_ch3'][cop_doy > break_pt] = np.nan
        del(break_pt)
    if (doy > 98 and doy < 103):
        cop_data['T_ch3'] = np.nan
    if doy == 103:
        break_pt = (datetime.datetime(2016, 4, 13, 11, 20, 24) - datetime.datetime(2016,1,1)).total_seconds() / 86400.
        cop_data['T_ch3'][cop_doy < break_pt] = np.nan
        del(break_pt)
    # soil chamber temps: before 04/12/2016 (not including that day), there were no soil chamber sensors
    if doy < 102:
        mpr_data['T_ch4'] = np.nan; mpr_data['T_ch5'] = np.nan; mpr_data['T_ch6'] = np.nan;
    if doy == 102:
        break_pt = (datetime.datetime(2016, 4, 12, 10, 37, 9) - datetime.datetime(2016,1,1)).total_seconds() / 86400.
        mpr_data['T_ch4'][mpr_doy < break_pt] = np.nan
        mpr_data['T_ch5'][mpr_doy < break_pt] = np.nan
        mpr_data['T_ch6'][mpr_doy < break_pt] = np.nan
    # PAR_2: data before before 04/08/2016 9:40 has to be removed
    if doy == 97:
    	cop_data['PAR_2'] = np.nan
    elif doy == 98:
    	break_pt = (datetime.datetime(2016, 4, 8, 9, 40, 25) - datetime.datetime(2016,1,1)).total_seconds() / 86400.
    	cop_data['PAR_2'][cop_doy < break_pt] = np.nan
    	del(break_pt)
    cop_data['PAR_2'][ cop_data['PAR_2'] < 0. ] = np.nan
    cop_data['PAR_1'][ cop_data['PAR_1'] < 0. ] = np.nan
    # identify corrupt thermocouple measurements using IQR criteria
    if np.sum(np.isfinite(cop_data['T_amb'])) > 0:
    	T_amb_lolim, T_amb_uplim = IQR_bounds_func(cop_data['T_amb'])
    	cop_data['T_amb'][ (cop_data['T_amb'] < T_amb_lolim) | (cop_data['T_amb'] > T_amb_uplim) ] = np.nan
    	del(T_amb_lolim, T_amb_uplim)
    if np.sum(np.isfinite(cop_data['T_ch1'])) > 0:
    	T_ch1_lolim, T_ch1_uplim = IQR_bounds_func(cop_data['T_ch1'])
    	cop_data['T_ch1'][ (cop_data['T_ch1'] < T_ch1_lolim) | (cop_data['T_ch1'] > T_ch1_uplim) ] = np.nan
    	del(T_ch1_lolim, T_ch1_uplim)
    if np.sum(np.isfinite(cop_data['T_ch2'])) > 0:
    	T_ch2_lolim, T_ch2_uplim = IQR_bounds_func(cop_data['T_ch2'])
    	cop_data['T_ch2'][ (cop_data['T_ch2'] < T_ch2_lolim) | (cop_data['T_ch2'] > T_ch2_uplim) ] = np.nan
    	del(T_ch2_lolim, T_ch2_uplim)
    if np.sum(np.isfinite(cop_data['T_ch3'])) > 0:
    	T_ch3_lolim, T_ch3_uplim = IQR_bounds_func(cop_data['T_ch3'])
    	cop_data['T_ch3'][ (cop_data['T_ch3'] < T_ch3_lolim) | (cop_data['T_ch3'] > T_ch3_uplim) ] = np.nan
    	del(T_ch3_lolim, T_ch3_uplim)
    if np.sum(np.isfinite(mpr_data['T_ch4'])) > 0:
    	T_ch4_lolim, T_ch4_uplim = IQR_bounds_func(mpr_data['T_ch4'])
    	mpr_data['T_ch4'][ (mpr_data['T_ch4'] < T_ch4_lolim) | (mpr_data['T_ch4'] > T_ch4_uplim) ] = np.nan
    	del(T_ch4_lolim, T_ch4_uplim)
    if np.sum(np.isfinite(mpr_data['T_ch5'])) > 0:
    	T_ch5_lolim, T_ch5_uplim = IQR_bounds_func(mpr_data['T_ch5'])
    	mpr_data['T_ch5'][ (mpr_data['T_ch5'] < T_ch5_lolim) | (mpr_data['T_ch5'] > T_ch5_uplim) ] = np.nan
    	del(T_ch5_lolim, T_ch5_uplim)
    if np.sum(np.isfinite(mpr_data['T_ch6'])) > 0:
    	T_ch6_lolim, T_ch6_uplim = IQR_bounds_func(mpr_data['T_ch6'])
    	mpr_data['T_ch6'][ (mpr_data['T_ch6'] < T_ch6_lolim) | (mpr_data['T_ch6'] > T_ch6_uplim) ] = np.nan
    	del(T_ch6_lolim, T_ch6_uplim)
    '''
    cop_data['T_amb'][ cop_data['T_amb'] < -20. ] = np.nan
    cop_data['T_ch1'][ cop_data['T_ch1'] < -20. ] = np.nan
    cop_data['T_ch2'][ cop_data['T_ch2'] < -20. ] = np.nan
    cop_data['T_ch3'][ cop_data['T_ch3'] < -10. ] = np.nan
    '''
    #
    sensor_data = pd.DataFrame(columns=['day_of_year', 'PAR_1', 'PAR_2', 'T_amb', 'T_ch1', 'T_ch2', 'T_ch3', 'T_ch4', 'T_ch5', 'T_ch6'], dtype='float64')
    sensor_data['day_of_year'] = doy + np.arange(0,86400,5) / 86400.
    for i in range(sensor_data.shape[0]):
        loc_cop = np.where( np.abs(cop_doy - sensor_data['day_of_year'][i]) < 1e-5)[0]
        loc_mpr = np.where( np.abs(mpr_doy - sensor_data['day_of_year'][i]) < 1e-5)[0]
        if loc_cop.size > 0:
            sensor_data.set_value(i, 'PAR_1', cop_data['PAR_1'][loc_cop[0]])
            sensor_data.set_value(i, 'PAR_2', cop_data['PAR_2'][loc_cop[0]])
            sensor_data.set_value(i, 'T_amb', cop_data['T_amb'][loc_cop[0]])
            sensor_data.set_value(i, 'T_ch1', cop_data['T_ch1'][loc_cop[0]])
            sensor_data.set_value(i, 'T_ch2', cop_data['T_ch2'][loc_cop[0]])
            sensor_data.set_value(i, 'T_ch3', cop_data['T_ch3'][loc_cop[0]])
        if loc_mpr.size > 0:
            sensor_data.set_value(i, 'T_ch4', mpr_data['T_ch4'][loc_mpr[0]])
            sensor_data.set_value(i, 'T_ch5', mpr_data['T_ch5'][loc_mpr[0]])
            sensor_data.set_value(i, 'T_ch6', mpr_data['T_ch6'][loc_mpr[0]])
    print('\n' + str(sensor_data.shape[0]) + ' lines read from sensor data file(s) on day 20' + run_date_str)
    print(sensor_data.describe().transpose())
    # dump data into csv files
    output_fname = sensor_dir + '/corrected/hyy_sensor_data_20' + run_date_str + '.csv'
    sensor_data.to_csv(output_fname, sep=',', na_rep='NaN', index=False)  # do not output 'row name'

print('Done.')

print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
