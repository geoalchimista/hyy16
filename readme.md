Ecosystem-Atmosphere Exchange of Carbonyl Sulfide (COS), Hyytiälä, Finland

Wu Sun (wu.sun@ucla.edu)

This is a code repository for the project.


# Requirements

- Python 3.5 or 3.6
- Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `requests`

The Anaconda Distribution of Python (https://continuum.io/downloads) is recommended as it bundles all the required packages with Python.


# What do they do?

`preproc_config.py`: Configuration of preprocessing settings. **Modify the directories in this script before you run any other script.**
- To configure it for daily online processing, set the key `process_recent_period` in `run_options` to `True`. By default, the processing traces back 3 days in time. This can be configured through the key `traceback_in_days`. 

`hyy16_fetch_smear_data.py`: Fetch SMEAR II meteorological data through its official API portal. Optional arguments are
- `-n`: get the data from the starting date till now. Enable this for daily online processing.
- `-v`: get one variable at a time, slow mode. Use this if it is too slow to get all the variables in one request.

`hyy16_flow_data.py`: Gapfill flow data and subset by day. Optional argument is `-s`, to run in silent mode without printing daily summary.

`hyy16_leaf_area.py`: Interpolate leaf area.

`hyy16_sensor_data.py`: Reformat and filter sensor data. Optional argument is `-s`, to run in silent mode without printing daily summary.

**Note**: the old flux calculation programs (`hyy16_chdata_proc.py` and `hyy16_chdata_proc_all.py`) are deprecated and removed from this repository. Use the tool [PyChamberFlux](https://github.com/geoalchimista/chflux/) for flux calculation.


# How long it takes to run

These are running times over all 2016 data, on a quad-core MacBook Pro 15'' (May 2015) model with single thread/process.

- `hyy16_fetch_smear_data.py`: ~ 10 seconds (It may also depend on the bandwidth.)
- `hyy16_flow_data.py`: ~ 5 minutes without plotting
- `hyy16_leaf_area.py`: ~ 0.5 second
- `hyy16_sensor_data.py`: ~ 90 seconds without plotting; ~ 220 seconds with plotting
