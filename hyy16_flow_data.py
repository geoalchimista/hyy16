"""
Extract, combine, and average flow data.
For pre-processing only, not intended for general-purpose use.

Hyytiälä COS campaign, April-November 2016

Copyright 2016 Wu Sun <wu.sun@ucla.edu>

Revision history
----------------
26 May 2016, W.S.
- Created.

6 Jul 2016, W.S.
- Changed to process only the recent 7 days of data.

14 Jan 2017, W.S.
- Code review and small edits.

"""
import os
import datetime
import numpy as np
import pandas as pd
import preproc_config  # preprocessing config file, in the same directory


def timesec_to_doy(ts_array, year=2016):
    """Convert time in seconds to day of year."""
    time_sec_start = datetime.datetime(1904, 1, 1, 0, 0)
    time_sec_current_year = (datetime.datetime(year, 1, 1) -
                             time_sec_start).total_seconds()
    return (ts_array - time_sec_current_year) / 86400.


# echo program starting
print('Subsetting, gapfilling and downsampling the flow data...')
dt_start = datetime.datetime.now()
print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
print('numpy version = ' + np.__version__)
print('pandas version = ' + pd.__version__)


# settings
pd.options.display.float_format = '{:.2f}'.format
# let pandas dataframe displays float with 2 decimal places

flow_dir = preproc_config.data_dir['flow_data_raw']
output_dir = preproc_config.data_dir['flow_data_reformatted']


# load all flow data files
df_flow = None
for i in range(40, 340):
    flow_fname = flow_dir + '/data_%d.dat' % i
    if os.path.isfile(flow_fname) is False:
        continue

    df_flow_loaded = pd.read_csv(
        flow_fname, delimiter='\t',
        names=['time_sec', 'flow_out', 'flow_1',
               'flow_2', 'flow_3', 'flow_4', 'flow_5'], engine='c')

    if df_flow is None:
        df_flow = df_flow_loaded
    else:
        df_flow = pd.concat([df_flow, df_flow_loaded], ignore_index=True)

    del(df_flow_loaded)


# echo flow data status
print('%d lines read from flow data.' % df_flow.shape[0])

# convert time variable to day of the year
doy_flow = timesec_to_doy(df_flow['time_sec'].values)


doy_start = np.floor(doy_flow[0])
# doy_start = np.ceil(doy_flow[-1]) - 7.  # enable to process only recent data
doy_end = np.ceil(doy_flow[-1])


# to bin the data by day, gapfill, and downsample to 1 min step
for doy in np.arange(doy_start, doy_end):
    run_date_str = (
        datetime.datetime(2016, 1, 1) +
        datetime.timedelta(doy + 0.5)).strftime('%Y%m%d')
    # gapfilling: to oversample to 0.5 s step and fill by interpolation
    # note the gapfilled data are in a numpy array for convenience
    flow_data_gapfilled = np.zeros((24 * 60 * 60 * 2, 7)) * np.nan
    flow_data_gapfilled[:, 0] = np.arange(0, 86400, 0.5) / 86400. + doy
    for col_num in range(1, 7):
        finite_loc = np.where(np.isfinite(df_flow.iloc[:, col_num].values))[0]
        flow_data_gapfilled[:, col_num] = np.interp(
            flow_data_gapfilled[:, 0], doy_flow[finite_loc],
            df_flow.iloc[finite_loc, col_num].values)
    # downsampling to 1 min step
    df_flow_downsampled = pd.DataFrame(
        columns=['doy', 'flow_out', 'flow_1', 'flow_2',
                 'flow_3', 'flow_4', 'flow_5'], dtype=np.float64)
    df_flow_downsampled['doy'] = (np.arange(1440) + 0.5) / 1440. + doy
    for row_num in range(df_flow_downsampled.shape[0]):
        for col_num in range(1, 7):
            df_flow_downsampled.set_value(
                row_num, df_flow_downsampled.columns[col_num],
                np.nanmean(
                    flow_data_gapfilled[row_num * 120:(row_num + 1) * 120,
                                        col_num]))
    # '%.6f' is the accuracy of the raw data; round the flow rates
    df_flow_downsampled = df_flow_downsampled.round({
        'doy': 14, 'flow_out': 6, 'flow_1': 6, 'flow_2': 6, 'flow_3': 6,
        'flow_4': 6, 'flow_5': 6})

    # dump data into csv files; do not output row index
    output_fname = output_dir + '/hyy16_flow_data_' + run_date_str + '.csv'
    df_flow_downsampled.to_csv(output_fname, na_rep='NaN', index=False)
    print('\n%d lines converted from flow data file(s) on the day %s' %
          (df_flow_downsampled.shape[0], run_date_str) +
          '\nDownsampled to 1 min step.\n')
    print(df_flow_downsampled.describe().transpose())
    del(flow_data_gapfilled, df_flow_downsampled)


# echo program ending
dt_end = datetime.datetime.now()
print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
print('Done. Finished in %.2f seconds.' % (dt_end - dt_start).total_seconds())
