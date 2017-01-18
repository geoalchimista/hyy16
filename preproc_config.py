"""Preprocessing configuration file."""

# filepaths for raw data (input), and reformatted data (output)
data_dir = {
    'flow_data_raw':
    '/Volumes/Perovskite/projects/ulli_lab/2016_Finland/QCL_data/Flowmeter/',
    'flow_data_reformatted':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/flow/',
    'sensor_data_raw':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/sensor_data/',
    'sensor_data_reformatted':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/sensor/',
    'met_data':
    '/Users/wusun/Dropbox/Projects/hyytiala_2016/data/preprocessed/met/'
}

run_options = {
    'process_recent_period': False,
    'traceback_in_days': 3,
    'plot_flow_data': True,
    'plot_sensor_data': True
}
