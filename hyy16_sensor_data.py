"""
Extract, combine, and correct chamber sensor data.
For pre-processing only, not intended for general-purpose use.

Hyytiälä COS campaign, April-November 2016

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

Revision history
----------------
26 May 2016, W.S.
- The two PAR sensors are now called 'PAR_ch_1' and 'PAR_ch_2', because their
  association with the chambers changed throughout the campaign.

29 Aug 2016, W.S.
- Continue to the next day's file in the loop when the current day's file
  is not found. This is to skip the day 28 Aug 2016 for missing data.

16 Jan 2017, W.S.
- Running options are now controlled by an external config file.
- Code review and small edits
- Ad hoc filtering criteria added
- Daily plot option added, which is controlled by the preprocessing config

"""
import argparse
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import preproc_config  # preprocessing config file, in the same directory


def IQR_bounds_func(x):
    """Filter thermocouple data by IQR bounds. Used only in this script."""
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25, 75])
        IQR = q3 - q1
        return(q1 - 2 * IQR, q3 + 5 * IQR)
    else:
        return(np.nan, np.nan)


# define terminal argument parser
parser = argparse.ArgumentParser(
    description='Extract, combine, and correct chamber sensor data.')
parser.add_argument('-s', '--silent', dest='flag_silent_mode',
                    action='store_true',
                    help='silent mode: run without printing daily summary')
args = parser.parse_args()


# echo program starting
print('Subsetting, gapfilling and downsampling the biomet sensor data...')
dt_start = datetime.datetime.now()
print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
print('numpy version = ' + np.__version__)
print('pandas version = ' + pd.__version__)
if preproc_config.run_options['plot_sensor_data']:
    print('Plotting option is enabled. Will generate daily plots.')


# settings
pd.options.display.float_format = '{:.2f}'.format
# let pandas dataframe displays float with 2 decimal places

plt.rcParams.update({'mathtext.default': 'regular'})  # sans-serif math
plt.style.use('ggplot')

sensor_dir = preproc_config.data_dir['sensor_data_raw']
output_dir = preproc_config.data_dir['sensor_data_reformatted']


# get file list of sensor data
lc_sensor_flist = glob.glob(
    sensor_dir + '/sm_cop/*.cop')  # leaf chamber sensors
sc_sensor_flist = glob.glob(
    sensor_dir + '/sm_mpr/*.mpr')  # soil chamber sensors


# local time is UTC+2
doy_today = (datetime.datetime.utcnow() -
             datetime.datetime(2016, 1, 1)).total_seconds() / 86400. + 2. / 24.

if preproc_config.run_options['process_recent_period']:
    doy_start = np.int(doy_today -
                       preproc_config.run_options['traceback_in_days'])
    doy_end = np.int(np.ceil(doy_today))
else:
    doy_start = 97  # campaign starts on 7 Apr 2016
    doy_end = 315  # campaign ends on 10 Nov 2016 (plus one for `range()`)

year_start = 2016  # starting year for converting day of year values

# data fields in the leaf chamber sensor data file (*.cop)
# correspondence between chamber number and sensor number was changing
# throughout the campaign. refer to the metadata table for the information.
# 0 - time; 1 - PAR_ch_1; 2 - PAR_ch_2;
# 8 - ambient T; 10 - T_ch_1;
# 11 - T_ch_2; 12 - T_ch_3;

# data fields in the soil chamber sensor data file (*.mpr)
# 0 - time; 5 - soil chamber 1 (T_ch_4); 6 - soil chamber 2 (T_ch_5)
# 7 - soil chamber 3 (T_ch_6)
for doy in range(doy_start, doy_end):
    run_date_str = (datetime.datetime(2016, 1, 1) +
                    datetime.timedelta(doy + 0.5)).strftime('%y%m%d')
    current_lc_sensor_files = [s for s in lc_sensor_flist if run_date_str in s]
    current_sc_sensor_files = [s for s in sc_sensor_flist if run_date_str in s]

    # reading leaf chamber sensor data
    df_lc_sensor = None
    if len(current_lc_sensor_files) > 0:
        for entry in current_lc_sensor_files:
            df_lc_sensor_loaded = pd.read_csv(
                entry, sep='\\s+', usecols=[0, 1, 2, 8, 10, 11, 12],
                names=['datetime', 'PAR_ch_1', 'PAR_ch_2', 'T_amb',
                       'T_ch_1', 'T_ch_2', 'T_ch_3'],
                dtype={'datetime': str, 'PAR_ch_1': np.float64,
                       'PAR_ch_2': np.float64, 'T_amb': np.float64,
                       'T_ch_1': np.float64, 'T_ch_2': np.float64,
                       'T_ch_3': np.float64},
                parse_dates={'timestamp': [0]},
                date_parser=lambda s: np.datetime64(
                    '%s-%s-%s %s:%s:%s' % (s[0:4], s[4:6], s[6:8],
                                           s[8:10], s[10:12], s[12:14])),
                engine='c', na_values='-')
            if df_lc_sensor is None:
                df_lc_sensor = df_lc_sensor_loaded
            else:
                df_lc_sensor = pd.concat([df_lc_sensor, df_lc_sensor_loaded],
                                         ignore_index=True)
            del df_lc_sensor_loaded
    else:
        print('Leaf chamber sensor data file not found on day 20%s' %
              run_date_str)
        continue

    # reading soil chamber sensor data
    df_sc_sensor = None
    if len(current_sc_sensor_files) > 0:
        for entry in current_sc_sensor_files:
            df_sc_sensor_loaded = pd.read_csv(
                entry, sep='\\s+', usecols=[0, 5, 6, 7],
                names=['datetime', 'T_ch_4', 'T_ch_5', 'T_ch_6'],
                dtype={'datetime': str, 'T_ch_4': np.float64,
                       'T_ch_5': np.float64, 'T_ch_6': np.float64},
                parse_dates={'timestamp': [0]},
                date_parser=lambda s: np.datetime64(
                    '%s-%s-%s %s:%s:%s' % (s[0:4], s[4:6], s[6:8],
                                           s[8:10], s[10:12], s[12:14])),
                engine='c')
            if df_sc_sensor is None:
                df_sc_sensor = df_sc_sensor_loaded
            else:
                df_sc_sensor = pd.concat([df_sc_sensor, df_sc_sensor_loaded],
                                         ignore_index=True)
            del df_sc_sensor_loaded
    else:
        print('Soil chamber sensor data file not found on day 20%s' %
              run_date_str)
        continue

    # convert day of year number
    doy_lc_sensor = \
        (df_lc_sensor['timestamp'] - pd.Timestamp('%s-01-01' % year_start)) / \
        pd.Timedelta(days=1)

    # parse datetime strings
    # doy_lc_sensor = np.zeros(df_lc_sensor.shape[0]) * np.nan
    # for i in range(df_lc_sensor.shape[0]):
    #     dt_str = df_lc_sensor.loc[i, 'datetime']
    #     if len(dt_str) == 14:
    #         # accelerate datetime parsing with manual operations
    #         dt_converted = datetime.datetime(
    #             int(dt_str[0:4]), int(dt_str[4:6]), int(dt_str[6:8]),
    #             int(dt_str[8:10]), int(dt_str[10:12]), int(dt_str[12:14]))
    #         doy_lc_sensor[i] = \
    #             (dt_converted -
    #                 datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
    #         # doy_lc_sensor[i] = (
    #         #     datetime.datetime.strptime(dt_str, '%Y%m%d%H%M%S') -
    #         #     datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
    #     else:
    #         doy_lc_sensor[i] = np.nan

    # indices for insertion, range 0 to 17279
    ind_lc_sensor = (doy_lc_sensor - doy) * 86400. / 5.
    ind_lc_sensor = np.round(ind_lc_sensor).astype(np.int64)

    # convert day of year number
    doy_sc_sensor = \
        (df_sc_sensor['timestamp'] - pd.Timestamp('%s-01-01' % year_start)) / \
        pd.Timedelta(days=1)

    # doy_sc_sensor = np.zeros(df_sc_sensor.shape[0]) * np.nan
    # for i in range(df_sc_sensor.shape[0]):
    #     dt_str = df_sc_sensor.loc[i, 'datetime']
    #     if len(dt_str) == 14:
    #         # accelerate datetime parsing with manual operations
    #         dt_converted = datetime.datetime(
    #             int(dt_str[0:4]), int(dt_str[4:6]), int(dt_str[6:8]),
    #             int(dt_str[8:10]), int(dt_str[10:12]), int(dt_str[12:14]))
    #         doy_sc_sensor[i] = \
    #             (dt_converted -
    #                 datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
    #         # doy_sc_sensor[i] = (
    #         #     datetime.datetime.strptime(dt_str, '%Y%m%d%H%M%S') -
    #         #     datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
    #     else:
    #         doy_sc_sensor[i] = np.nan

    # indices for insertion, range 0 to 17279
    ind_sc_sensor = (doy_sc_sensor - doy) * 86400. / 5.
    ind_sc_sensor = np.round(ind_sc_sensor).astype(np.int64)

    # corrections for PAR and TC values
    # parameters from Juho Aalto <juho.aalto@helsinki.fi>, 13 April 2016
    # correction factor for 'PAR_ch_2' was updated 27 October 2016,
    # according to Juho Aalto <juho.aalto@helsinki.fi>
    df_lc_sensor['PAR_ch_1'] *= 200.  # was 210-220
    df_lc_sensor['PAR_ch_2'] *= 205.  # was 200
    df_lc_sensor['T_ch_1'] = df_lc_sensor['T_ch_1'] * 0.94 + 0.75
    df_lc_sensor['T_ch_2'] = df_lc_sensor['T_ch_2'] * 0.96 - 0.20
    if doy < 103:
        # before 13 April 2016, but not including that day
        df_lc_sensor['T_ch_3'] = df_lc_sensor['T_ch_3'] * 0.98 - 0.89
    else:
        # TC in the large leaf chamber reinstalled 13 April 2016 11:20 am
        # before that, temperature data were corrupt at this channel
        df_lc_sensor['T_ch_3'] = df_lc_sensor['T_ch_3'] * 0.97 - 0.39

    # mask corrupt data
    # 1. 'T_ch_3' data between April 8 and 13 of 2016 were corrupt
    if doy == 98:
        break_pt = (datetime.datetime(2016, 4, 8, 9, 33, 41) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor > break_pt, 'T_ch_3'] = np.nan
        del break_pt
    elif 98 < doy < 103:
        df_lc_sensor['T_ch_3'] = np.nan
    elif doy == 103:
        break_pt = (datetime.datetime(2016, 4, 13, 11, 20, 24) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor < break_pt, 'T_ch_3'] = np.nan
        del break_pt

    # 2. no soil chamber sensors before 12 April 2016 10:37:09 am
    if doy < 102:
        df_sc_sensor[['T_ch_4', 'T_ch_5', 'T_ch_6']] = np.nan
    elif doy == 102:
        break_pt = (datetime.datetime(2016, 4, 12, 10, 37, 9) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_sc_sensor.loc[doy_sc_sensor < break_pt,
                         ['T_ch_4', 'T_ch_5', 'T_ch_6']] = np.nan
        del break_pt

    # 3. remove 'PAR_ch_2' data before before 8 April 2016 09:40:25 am
    if doy == 97:
        df_lc_sensor['PAR_ch_2'] = np.nan
    elif doy == 98:
        break_pt = (datetime.datetime(2016, 4, 8, 9, 40, 25) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor < break_pt, 'PAR_ch_2'] = np.nan
        del break_pt

    # 4. 'PAR_ch_2' data from 08:40 to 09:41 on 7 June 2016 were corrupt
    if doy == 158:
        break_pt1 = (datetime.datetime(2016, 6, 7, 8, 40) -
                     datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        break_pt2 = (datetime.datetime(2016, 6, 7, 9, 41) -
                     datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[
            (doy_lc_sensor > break_pt1) & (doy_lc_sensor < break_pt2) &
            (df_lc_sensor['PAR_ch_1'].values < 400.), 'PAR_ch_1'] = np.nan
        df_lc_sensor.loc[
            (doy_lc_sensor > break_pt1) & (doy_lc_sensor < break_pt2) &
            (df_lc_sensor['PAR_ch_2'].values < 400.), 'PAR_ch_2'] = np.nan
        del break_pt1, break_pt2

    # 5. power failure for leaf chamber sensor logger
    # no data from 30 Aug 2016 13:44:36 to 5 Sep 2016 11:22:44
    if doy == 242:
        break_pt = (datetime.datetime(2016, 8, 30, 13, 44, 36) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor > break_pt, 1:] = np.nan
    if 242 < doy < 248:
        df_lc_sensor.loc[:, 1:] = np.nan
    if doy == 248:
        break_pt = (datetime.datetime(2016, 9, 5, 11, 22, 44) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor < break_pt, 1:] = np.nan

    # 6. thermocouple at channel 11 (T_ch_2) was fallen during
    # 29 Aug 2016 09:00 to 12 Sep 2016 11:00
    if 241 <= doy < 255:
        df_lc_sensor['T_ch_2'] = np.nan
    elif doy == 255:
        break_pt = (datetime.datetime(2016, 9, 12, 11, 0, 0) -
                    datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
        df_lc_sensor.loc[doy_lc_sensor < break_pt, 'T_ch_2'] = np.nan

    # 7. Bad PAR measurements from 10:30 to 11:00 on 5 Oct 2016 (?)
    # no action, since no abnormal measurements were detected in this period

    # 8. allow -5 as the lower limit of PAR (tolerance for random errors)
    df_lc_sensor.loc[df_lc_sensor['PAR_ch_1'] < -5., 'PAR_ch_1'] = np.nan
    df_lc_sensor.loc[df_lc_sensor['PAR_ch_2'] < -5., 'PAR_ch_2'] = np.nan

    # 9. identify corrupt thermocouple measurements using IQR criteria
    for col in ['T_amb', 'T_ch_1', 'T_ch_2', 'T_ch_3']:
        if np.sum(np.isfinite(df_lc_sensor[col].values)) > 0:
            TC_lolim, TC_uplim = IQR_bounds_func(df_lc_sensor[col].values)
            df_lc_sensor.loc[(df_lc_sensor[col] < TC_lolim) |
                             (df_lc_sensor[col] > TC_uplim), col] = np.nan

    for col in ['T_ch_4', 'T_ch_5', 'T_ch_6']:
        if np.sum(np.isfinite(df_sc_sensor[col].values)) > 0:
            TC_lolim, TC_uplim = IQR_bounds_func(df_sc_sensor[col].values)
            df_sc_sensor.loc[(df_sc_sensor[col] < TC_lolim) |
                             (df_sc_sensor[col] > TC_uplim), col] = np.nan

    df_all_sensor = pd.DataFrame(
        columns=['doy', 'PAR_ch_1', 'PAR_ch_2', 'T_amb', 'T_ch_1', 'T_ch_2',
                 'T_ch_3', 'T_ch_4', 'T_ch_5', 'T_ch_6'], dtype=np.float64)
    df_all_sensor['doy'] = doy + np.arange(0, 86400, 5) / 86400.

    for col in df_lc_sensor.columns.values[1:]:
        df_all_sensor.loc[ind_lc_sensor, col] = df_lc_sensor[col]
    for col in df_sc_sensor.columns.values[1:]:
        df_all_sensor.loc[ind_sc_sensor, col] = df_sc_sensor[col]

    # for i in range(df_all_sensor.shape[0]):
    #     loc_lc_sensor = np.where(
    #         np.abs(doy_lc_sensor - df_all_sensor.loc[i, 'doy']) < 1e-5)[0]
    #     loc_sc_sensor = np.where(
    #         np.abs(doy_sc_sensor - df_all_sensor.loc[i, 'doy']) < 1e-5)[0]
    #     if loc_lc_sensor.size > 0:
    #         for col in df_lc_sensor.columns.values[1:]:
    #             df_all_sensor.set_value(
    #                 i, col, df_lc_sensor.loc[loc_lc_sensor[0], col])
    #     if loc_sc_sensor.size > 0:
    #         for col in df_sc_sensor.columns.values[1:]:
    #             df_all_sensor.set_value(
    #                 i, col, df_sc_sensor.loc[loc_sc_sensor[0], col])

    # '%.2f' is the accuracy of the raw data; round the sensor data
    df_all_sensor = df_all_sensor.round({
        'doy': 14, 'PAR_ch_1': 2, 'PAR_ch_2': 2, 'T_amb': 2,
        'T_ch_1': 2, 'T_ch_2': 2, 'T_ch_3': 2, 'T_ch_4': 2,
        'T_ch_5': 2, 'T_ch_6': 2})

    # dump data into csv files; do not output row index
    output_fname = output_dir + '/hyy16_sensor_data_20%s.csv' % run_date_str
    df_all_sensor.to_csv(output_fname, sep=',', na_rep='NaN', index=False)

    # daily plots for diagnosing wrong measurements
    if preproc_config.run_options['plot_sensor_data']:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
        time_in_hour = (df_all_sensor['doy'].values - doy) * 24.
        for col in ['PAR_ch_1', 'PAR_ch_2']:
            axes[0].plot(time_in_hour, df_all_sensor[col].values,
                         label=col, lw=1.)
        axes[0].legend(loc='upper left', frameon=False, fontsize=10, ncol=2)

        for col in ['T_amb', 'T_ch_1', 'T_ch_2', 'T_ch_3']:
            axes[1].plot(time_in_hour, df_all_sensor[col].values,
                         label=col, lw=1.)
        axes[1].legend(loc='upper left', frameon=False, fontsize=10, ncol=4)

        for col in ['T_ch_4', 'T_ch_5', 'T_ch_6']:
            axes[2].plot(time_in_hour, df_all_sensor[col].values,
                         label=col, lw=1.)
        axes[2].legend(loc='upper left', frameon=False, fontsize=10, ncol=3)

        axes[0].set_ylabel('PAR ($\mu$mol m$^{-2}$ s$^{-1}$)')
        axes[1].set_ylabel('Temperature ($\degree$C)')
        axes[2].set_ylabel('Temperature ($\degree$C)')
        axes[2].set_xlim([0, 24])
        axes[2].xaxis.set_ticks(range(0, 25, 3))
        axes[2].set_xlabel('Hour (UTC+2)')

        fig.tight_layout()
        fig.savefig(output_dir +
                    '/plots/hyy16_sensor_data_20%s.png' % run_date_str)
        fig.clf()
        del fig, axes

    if not args.flag_silent_mode:
        print(
            '\n%d lines converted from sensor data file(s) on the day 20%s.' %
            (df_all_sensor.shape[0], run_date_str))
        print(df_all_sensor.describe().transpose())

    del df_lc_sensor, df_sc_sensor, df_all_sensor


# echo program ending
dt_end = datetime.datetime.now()
print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
print('Done. Finished in %.2f seconds.' % (dt_end - dt_start).total_seconds())
