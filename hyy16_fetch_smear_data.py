"""
Fetch meteorological data from the SMEAR website and bind them as a CSV table.

Hyytiälä COS campaign, April-November 2016

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import io
import copy
import datetime
import requests
import numpy as np
import pandas as pd
import preproc_config


# echo program starting
print('Retrieving meteorological data from ' +
      'SMEAR <http://avaa.tdata.fi/web/smart/smear> ... ')
dt_start = datetime.datetime.now()
print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
print('numpy version = ' + np.__version__)
print('pandas version = ' + pd.__version__)


output_dir = preproc_config.data_dir['met_data']

# local time is UTC+2
start_dt = '2016-04-01 00:00:00'
# end_dt = (datetime.datetime.utcnow() +
#           datetime.timedelta(0, seconds=7200)).strftime('%Y-%m-%d %H:%M:%S')
end_dt = '2016-11-11 00:00:00'


# variable names for retrieval from the SMEAR data website API
varnames = ['Pamb0', 'T1250', 'T672', 'T504', 'T336', 'T168', 'T84', 'T42',
            'RHIRGA1250', 'RHIRGA672', 'RHIRGA504', 'RHIRGA336',
            'RHIRGA168', 'RHIRGA84', 'RHIRGA42',
            'RPAR', 'PAR', 'diffPAR', 'maaPAR',
            'tsoil_humus', 'tsoil_A', 'tsoil_B1', 'tsoil_B2', 'tsoil_C1',
            'wsoil_humus', 'wsoil_A', 'wsoil_B1', 'wsoil_B2', 'wsoil_C1',
            'Precipacc']

# make a copy and insert custom timestamps
colnames = copy.copy(varnames)
colnames.insert(0, 'timestamp')
colnames.insert(1, 'doy')

# renaming will be done after filling all the variables in the met dataframe
renaming_dict = {
    'Pamb0': 'pres',
    'T1250': 'T_atm_125m',
    'T672': 'T_atm_67m',
    'T504': 'T_atm_50m',
    'T336': 'T_atm_34m',
    'T168': 'T_atm_17m',
    'T84': 'T_atm_8m',
    'T42': 'T_atm_4m',
    'RHIRGA1250': 'RH_125m',
    'RHIRGA672': 'RH_67m',
    'RHIRGA504': 'RH_50m',
    'RHIRGA336': 'RH_34m',
    'RHIRGA168': 'RH_17m',
    'RHIRGA84': 'RH_8m',
    'RHIRGA42': 'RH_4m',
    'RPAR': 'PAR_reflected',
    'PAR': 'PAR',
    'diffPAR': 'PAR_diffuse',
    'maaPAR': 'PAR_below',
    'tsoil_humus': 'T_soil_surf',
    'tsoil_A': 'T_soil_A',
    'tsoil_B1': 'T_soil_B1',
    'tsoil_B2': 'T_soil_B2',
    'tsoil_C1': 'T_soil_C1',
    'wsoil_humus': 'w_soil_surf',
    'wsoil_A': 'w_soil_A',
    'wsoil_B1': 'w_soil_B1',
    'wsoil_B2': 'w_soil_B2',
    'wsoil_C1': 'w_soil_C1',
    'Precipacc': 'precip', }


df_met = pd.DataFrame(columns=colnames)

# an url example
# url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?' +
#       'variable=Pamb0&table=HYY_META&' +
#       'from=2016-04-01 00:00:00&to=2016-04-02 00:00:00&'
#       'quality=ANY&averaging=30MIN&type=ARITHMETIC'


flag_timestamp_parsed = False
# fetch and dump data: dump each variable into TXT and combine 'em as CSV
for var in varnames:
    print('Fetching variable \'%s\' ...' % var, end=' ')

    # precipitation must be summed not averaged over the 30 min interval
    if var != 'Precipacc':
        avg_type = 'ARITHMETIC'
    else:
        avg_type = 'SUM'

    url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variable=' + var + \
          '&table=HYY_META&from=' + start_dt + '&to=' + end_dt + \
          '&quality=ANY&averaging=30MIN&type=' + avg_type

    response = requests.get(url, verify=True)
    # set `verify=True` to check SSL certificate
    if response.status_code != 200:
        print('Status %d: No response from the request for variable \'%s\'.' %
              (response.status_code, var))
        continue
    else:
        print('Successful!')

    # reading the response text as data table
    fetched_data = np.genfromtxt(
        io.BytesIO(response.text.encode('utf-8')), delimiter='"',
        dtype=[(var, 'float64'), ('timestamp', 'U30')],
        usecols=[0, 1], invalid_raise=False)
    if var == 'Pamb0':
        fetched_data[var][fetched_data[var] == 0.] = np.nan

    df_met[var] = fetched_data[var]

    # fill timestamps and convert to day of year
    if not flag_timestamp_parsed:
        df_met['timestamp'] = np.empty(fetched_data.shape[0], dtype=str)
        for i in range(fetched_data.shape[0]):
            dt_parsed = datetime.datetime.strptime(
                fetched_data['timestamp'][i],
                '%b %d, %Y %I:%M:%S %p')
            dt_str = str(dt_parsed)
            doy_converted = (
                dt_parsed -
                datetime.datetime(2016, 1, 1)).total_seconds() / 86400.
            df_met.set_value(i, 'timestamp', "'" + dt_str + "'")
            # use single quotation mark to enforce timestamp as string
            df_met.set_value(i, 'doy', doy_converted)
            del dt_parsed, dt_str, doy_converted
        else:
            # if the for-loop executes normally, go on to the else-clause
            flag_timestamp_parsed = True
            print('Timestamps parsed.')

    del url, response, fetched_data


# renaming column names in the output dataframe
for col in df_met.columns.values:
    if col in renaming_dict:
        df_met.rename(columns={col: renaming_dict[col]}, inplace=True)

print('Variable fields have been renamed in the output data.')

df_met.to_csv(output_dir + '/hyy16_met_data.csv', na_rep='NaN', index=False)
print('Tabulated data written to %s/hyy_met_data.csv' % output_dir)


# echo program ending
dt_end = datetime.datetime.now()
print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
print('Done. Finished in %.2f seconds.' % (dt_end - dt_start).total_seconds())
