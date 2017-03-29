"""Preprocessing configuration file."""

# filepaths for raw data (input), and reformatted data (output)
data_dir = {
    'flow_data_raw':
    '/Volumes/Perovskite/projects/ulli_lab/2016_Finland/QCL_data/Flowmeter/',

    'flow_data_reformatted':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/flow/',

    'sensor_data_raw':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/sensor_data/',

    'sensor_data_reformatted':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/sensor/',

    'met_data':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/met/',

    'leaf_area_data_raw':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/auxiliary/',

    'leaf_area_data_reformatted':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/leaf_area/',
}

run_options = {
    'process_recent_period': False,

    'traceback_in_days': 3,

    'plot_flow_data': False,

    'plot_sensor_data': False,
}
