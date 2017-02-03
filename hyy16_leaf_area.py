"""
Reformat leaf area data as a table.

Hyytiälä COS campaign, April-November 2016

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import preproc_config  # preprocessing config file, in the same directory


# plot settings
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams['hatch.color'] = 'w'  # white hatching lines
plt.style.use('ggplot')


# echo program starting
print('Reformatting the leaf area data....')
dt_start = datetime.datetime.now()
print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
print('numpy version = ' + np.__version__)
print('pandas version = ' + pd.__version__)


filepath_aspen_XL = preproc_config.data_dir['leaf_area_data_raw'] + \
    '/aspen_leaf_area_LC-XL.csv'
filepath_aspen_slide = preproc_config.data_dir['leaf_area_data_raw'] + \
    '/aspen_leaf_area_LC-Slide.csv'
filepath_pine = preproc_config.data_dir['leaf_area_data_raw'] + \
    '/chamber_metadata.csv'


df_aspen_XL = pd.read_csv(
    filepath_aspen_XL, engine='c', comment='#', parse_dates=[0],
    usecols=[0, 1], infer_datetime_format=True)
df_aspen_slide = pd.read_csv(
    filepath_aspen_slide, engine='c', comment='#', parse_dates=[0],
    usecols=[0, 1], infer_datetime_format=True)
df_pine = pd.read_csv(
    filepath_pine, engine='c', comment='#', parse_dates=[14, 15],
    usecols=None, infer_datetime_format=True)
df_pine = df_pine[['species', 'leaf_area', 'ch_label',
                   'install_datetime', 'uninstall_datetime', 'ch_no']]


# the aggregated leaf area table, for input in flux calculation
df_la = pd.DataFrame(columns=['datetime', 'LC-S-A', 'LC-S-B', 'LC-L-A',
                              'LC-XL', 'LC-Slide'])
for ch_label in ['LC-S-A', 'LC-S-B', 'LC-L-A']:
    df_extracted = df_pine.loc[df_pine['ch_label'] == ch_label, :]
    df_extracted.loc[:, 'uninstall_datetime'] = \
        df_extracted['uninstall_datetime'] - np.timedelta64(1, 's')
    df_la = df_la.append(
        df_extracted[['leaf_area', 'install_datetime']].rename(
            columns={'leaf_area': ch_label, 'install_datetime': 'datetime'}),
        ignore_index=True)
    df_la = df_la.append(
        df_extracted[['leaf_area', 'uninstall_datetime']].rename(
            columns={'leaf_area': ch_label, 'uninstall_datetime': 'datetime'}),
        ignore_index=True)

df_la = df_la.append(df_aspen_XL.rename(columns={'leaf_area': 'LC-XL'}))
df_la = df_la.append(df_aspen_slide.rename(columns={'leaf_area': 'LC-Slide'}))

df_la = df_la.sort_values(by=['datetime'])
df_la = df_la.reset_index(drop=True)

for ch_label in ['LC-S-A', 'LC-S-B', 'LC-L-A']:
    x = (df_la.loc[np.isnan(df_la[ch_label]), 'datetime'].values -
         np.datetime64('2016-01-01')) / np.timedelta64(1, 'D')
    xp = (df_la.loc[np.isfinite(df_la[ch_label]), 'datetime'].values -
          np.datetime64('2016-01-01')) / np.timedelta64(1, 'D')
    fp = df_la.loc[np.isfinite(df_la[ch_label]), ch_label].values
    df_la.loc[np.isnan(df_la[ch_label]), ch_label] = np.interp(x, xp, fp)
    # constant leaf area is assumed in each interval

for ch_label in ['LC-XL', 'LC-Slide']:
    x = (df_la.loc[np.isnan(df_la[ch_label]), 'datetime'].values -
         np.datetime64('2016-01-01')) / np.timedelta64(1, 'D')
    xp = (df_la.loc[np.isfinite(df_la[ch_label]), 'datetime'].values -
          np.datetime64('2016-01-01')) / np.timedelta64(1, 'D')
    fp = df_la.loc[np.isfinite(df_la[ch_label]), ch_label].values
    df_la.loc[np.isnan(df_la[ch_label]), ch_label] = \
        np.interp(x, xp, fp, left=np.nan, right=None)

# remove duplicate entries
df_la = df_la.drop_duplicates()
df_la = df_la[['datetime', 'LC-S-A', 'LC-S-B', 'LC-L-A', 'LC-XL', 'LC-Slide']]
df_la = df_la.reset_index(drop=True)

df_la.insert(1, 'doy', np.nan)
df_la['doy'] = (df_la['datetime'] - pd.Timestamp('2016-01-01')) / \
    np.timedelta64(1, 'D')

df_la = df_la.round({'LC-S-A': 3, 'LC-S-B': 3, 'LC-L-A': 3, 'LC-XL': 6,
                     'LC-Slide': 6})

df_la.to_csv(preproc_config.data_dir['leaf_area_data_reformatted'] +
             '/leaf_area.csv', index=False, na_rep='NaN')


# plot chamber arrangement schemes throughout the campaign
# this is complicated, but a figure could make it clear
color_list = plt.rcParams['axes.prop_cycle'].by_key()['color'] + \
    ['#b15928', '#ffed6f']
df_pine['install_doy'] = \
    (df_pine['install_datetime'] - pd.Timestamp('2016-01-01')) / \
    np.timedelta64(1, 'D')
df_pine['uninstall_doy'] = \
    (df_pine['uninstall_datetime'] - pd.Timestamp('2016-01-01')) / \
    np.timedelta64(1, 'D')


fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.set_xlim([90, 320])
ax.xaxis.set_ticks(range(90, 330, 10))
ax.set_xlabel('Date, or days since 1 Jan 2016')
ax.set_ylim([0.5, 7.2])
ax.yaxis.set_ticklabels(range(7))
ax.set_ylabel('Chamber number')
for i, ch_label in enumerate(['LC-S-A', 'LC-S-B', 'LC-L-A', 'LC-XL',
                              'LC-Slide', 'SC1', 'SC2', 'SC2-T', 'SC3']):
    ax.text(95 + i * 23, 6.8, ch_label, fontsize=12,
            bbox={'facecolor': color_list[i], 'alpha': 1., 'pad': 5})
    df_subset = df_pine.loc[df_pine['ch_label'] == ch_label, :]
    x_pos = df_subset['install_doy'].values
    y_pos = df_subset['ch_no'].values - 0.25
    width = df_subset['uninstall_doy'].values - df_subset['install_doy'].values
    height = 0.5
    for k in range(x_pos.size):
        rect = ax.add_patch(Rectangle((x_pos[k], y_pos[k]), width[k], height,
                                      facecolor=color_list[i]))
        if df_subset['species'].values[k] == 'blank':
            rect.set_hatch('///')

ax.text(295, 6.8, 'Blank test', fontsize=12, color='k',
        bbox={'facecolor': 'darkgray', 'alpha': 1., 'pad': 5, 'hatch': '///'})

# add description for date
xax_doy_array = ax.xaxis.get_ticklocs()
xax_doy_array = xax_doy_array.astype(np.int64)
xax_date_array = [''] * len(xax_doy_array)
for i, doy in enumerate(xax_doy_array):
    doy = int(doy)
    xax_date_array[i] = (pd.Timestamp('2016-01-01') +
                         np.timedelta64(doy, 'D')).strftime('%m/%d')

xax_doy_array = list(map(str, xax_doy_array))
xax_combined_array = list(map(''.join, zip(xax_date_array,
                                           np.repeat('\n', len(xax_doy_array)),
                                           xax_doy_array)))
ax.xaxis.set_ticklabels(xax_combined_array)

fig.tight_layout()
fig.savefig(preproc_config.data_dir['leaf_area_data_raw'] +
            '/chamber_arrangement.pdf', dpi=150)


# echo program ending
dt_end = datetime.datetime.now()
print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
print('Done. Finished in %.2f seconds.' % (dt_end - dt_start).total_seconds())
