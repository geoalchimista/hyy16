Ecosystem-Atmosphere Exchange of Carbonyl Sulfide (COS), Hyytiälä, Finland

Wu Sun (wu.sun@ucla.edu)

This is a code repository for the project.

==**Warning: Under construction; some scripts may not be able to run.**==

# Requirements

- Python 3.5 or 3.6
- Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `requests`

The Anaconda Distribution of Python (https://continuum.io/downloads) is recommended as it bundles all the required packages with Python.

# What do they do?

`preproc_config.py`: Configuration of preprocessing settings.

`hyy16_fetch_smear_data.py`: Fetch SMEAR II meteorological data through its offical API portal.

`hyy16_flow_data.py`: Gapfill flow data and subset by day.

`hyy16_leaf_area.py`: Interpolate leaf area.

`hyy16_sensor_data.py`: Reformat and filter sensor data.

==**Work in progess**==
`hyy16_chdata_proc.py`: Flux calculation. Unable to run yet.
`hyy16_chdata_proc_all.py`: Flux calculation over all the data. Unable to run yet.
