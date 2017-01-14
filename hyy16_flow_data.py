# -*- coding: utf-8 -*-
'''
Hyytiälä pine forest, Apr-now 2016
Extract, combine and average flow data

Revision
--------
07/06/2016, W.S.
- Process only the recent 7 days of data. 
'''
import numpy as np
# from scipy import stats, signal
import pandas as pd
# import matplotlib.pyplot as plt
import os, glob, datetime, linecache, copy

pd.options.display.float_format = '{:.2f}'.format  # pandas dataframe displays float with 2 decimal places

print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
print('Subsetting, gapfilling and downsampling the flow data...')

flo_dir = '/Users/wusun/Dropbox/QCLdata/Flowmeter/'
if not os.path.exists(flo_dir): flo_dir = '../../Volumes/Perovskite/projects/ulli_lab/2016_Finland/QCL_data/Flowmeter/'

output_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/flow_data/'

# convert mac timestamp to day of year
def timestamp_to_doy(ts_array, year=2016):
    mac_sec_start = datetime.datetime(1904,1,1,0,0)
    mac_sec_this_year = (datetime.datetime(year,1,1,0,0) - mac_sec_start).total_seconds()
    return( (ts_array - mac_sec_this_year) / 86400.)

# first, load all flow data files
# columns: 0 - time, 1 - outflow?, 2 to 6 - inflows
# chamber assignments (starting from col2)
# before Apr 21: flow1 - 'LC-S-B'; flow2 - 'LC-S-A'; ch3 - 'LC-L-A' or 'LC-XL'; ch4 - 'SC-1'; ch5 - 'SC-2'; flow for ch6 is missing, use ch5 - 'SC-3'
# after Apr 21: flow2 - 'LC-S-B'; flow1 - 'LC-S-A'; ch3 - 'LC-XL'; ch4 - 'SC-1'; ch5 - 'SC-2'; flow for ch6 is missing, use ch5 - 'SC-3'flow_data = np.array([])
flow_data = np.array([])
for i in range(40,400):
    flo_fname = flo_dir + '/data_' + '%d' % i + '.dat'
    if os.path.isfile(flo_fname) == False:
        continue
    flow_data_loaded = np.genfromtxt(flo_fname, skip_header=0, invalid_raise=False)
    if flow_data.size > 0:
        flow_data = np.concatenate((flow_data, flow_data_loaded))
    else: flow_data = flow_data_loaded
    del(flow_data_loaded)

flow_doy = timestamp_to_doy(flow_data[:,0])

# doy_start = 97.
doy_start = np.ceil(flow_doy[-1]) - 7.  # modified 07/06/2016
doy_end = np.ceil(flow_doy[-1])
# binning daily data, gapfill, and downsample to 1 min step
for doy in np.arange(doy_start, doy_end+1):
    run_date_str = (datetime.datetime(2016,1,1) + datetime.timedelta(doy+0.5)).strftime("%Y%m%d")
    # extract_loc = np.where( (flow_doy >= doy) & (flow_doy < doy+1.) )
    # flow_data_extract = flow_data[extract_loc, :]  # daily extract
    # gapfilling
    flow_data_gapfilled = np.zeros((24*60*60*2, 7)) * np.nan # 0.5 sec step
    flow_data_gapfilled[:,0] = np.arange(0,86400,0.5) / 86400. + doy
    for col_num in range(1,7):
        finite_loc = np.where(np.isfinite(flow_data[:,col_num]))[0]
        flow_data_gapfilled[:,col_num] = np.interp(flow_data_gapfilled[:,0], flow_doy[finite_loc], flow_data[ finite_loc, col_num ])
    # downsampling to 1 min step
    flow_data_downsampled = pd.DataFrame(columns=['day_of_year', 'flow_out', 'flow_1', 'flow_2', 'flow_3', 'flow_4', 'flow_5'], dtype='float64')
    flow_data_downsampled['day_of_year'] = (np.arange(1440) + 0.5) / 1440. + doy
    for row_num in range(flow_data_downsampled.shape[0]):
        for col_num in range(1,7):
            flow_data_downsampled.set_value(row_num, flow_data_downsampled.columns[col_num], 
                np.nanmean(flow_data_gapfilled[row_num*120:(row_num+1)*120, col_num]))
    # dump data into csv files
    output_fname = output_dir + '/hyy_flow_data_' + run_date_str + '.csv'
    flow_data_downsampled.to_csv(output_fname, sep=',', na_rep='NaN', index=False)  # do not output 'row name'
    print('\n' + str(flow_data_downsampled.shape[0]) + ' lines read from flow data file(s) on day ' + run_date_str + '\nDownsampled to 1 min step.\n')
    print(flow_data_downsampled.describe().transpose())
    del(flow_data_gapfilled, flow_data_downsampled)

print('\nDone.')
print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))


'''
# downsample the flow data to 1 min step
flow_data_downsampled = pd.DataFrame(columns=['timestamp', 'flow_out', 'flow_1', 'flow_2', 'flow_3', 'flow_4', 'flow_5'], dtype='float64')
timestamp_start = np.ceil(flow_data[0,0] / 60) * 60. + 60.
timestamp_end = np.floor(flow_data[-1,0] / 60) * 60.

flow_data_downsampled['timestamp'] = np.arange(timestamp_start, timestamp_end, 60.)

for row_num in range(flow_data_downsampled.shape[0]):
    extract_loc = np.where((flow_data[:,0] >= flow_data_downsampled.iloc[row_num,0] - 30.) & 
      (flow_data[:,0] < flow_data_downsampled.iloc[row_num,0] + 30.))
    if extract_loc[0].size > 0:
        for col_num in range(1,7): flow_data_downsampled.set_value(row_num, flow_data_downsampled.columns[col_num], np.nanmean(flow_data[extract_loc, col_num]) )
    else:
        for col_num in range(1,7): flow_data_downsampled.set_value(row_num, flow_data_downsampled.columns[col_num], np.nan)
            
# interpolate to remove NaNs
for col_num in range(1,7):
    copy_of_column = np.copy(flow_data_downsampled.values[:, col_num])
    finite_loc = np.where(copy_of_column)[0]
    flow_data_downsampled[flow_data_downsampled.columns[col_num]] = np.interp(flow_data_downsampled['timestamp'], 
        flow_data_downsampled['timestamp'][finite_loc], flow_data_downsampled.values[finite_loc, col_num])

# insert 'day of year' column
flow_data_downsampled.insert(0, 'day_of_year', np.nan)
mac_sec_start = datetime.datetime(1904,1,1,0,0)
mac_sec_2016 = (datetime.datetime(2016,1,1,0,0) - mac_sec_start).total_seconds()
flow_data_downsampled['day_of_year'] = (flow_data_downsampled['timestamp'] - mac_sec_2016) / 86400.  # convert seconds to day of year


# dump data into csv files
output_fname = output_dir + '/hyy_flow_data.csv'
flow_data_downsampled.to_csv(output_fname, sep=',', na_rep='NaN', index=False)  # do not output 'row name'

print('\n' + str(flow_data.shape[0]) + ' lines read from flow data file(s). Downsampled to 1 min step.\n')
print(flow_data_downsampled.describe().transpose())
'''