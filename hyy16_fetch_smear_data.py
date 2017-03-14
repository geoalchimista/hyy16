"""
Fetch meteorological data from the SMEAR website and bind them as a CSV table.

Hyytiälä COS campaign, April-November 2016

(c) 2016-2017 Wu Sun <wu.sun@ucla.edu>

"""
import io
import argparse
import copy
import datetime
import requests
import numpy as np
import pandas as pd
import preproc_config


def timestamp_parser(*args):
    """
    A timestamp parser for `pandas.read_csv()`.
    Argument list: year, month, day, hour, minute, second
    """
    return np.datetime64('%s-%s-%s %s:%s:%s' %
                         args)


# define terminal argument parser
parser = argparse.ArgumentParser(description='Get SMEAR meteorological data.')
parser.add_argument('-v', '--variable', dest='flag_get_variable',
                    action='store_true',
                    help='get one variable at a time, slow mode')
parser.add_argument('-n', '--now', dest='flag_now', action='store_true',
                    help='get the data from the starting date till now')
args = parser.parse_args()


# echo program starting
print('Retrieving meteorological data from ' +
      'SMEAR <http://avaa.tdata.fi/web/smart/smear> ... ')
dt_start = datetime.datetime.now()
print(datetime.datetime.strftime(dt_start, '%Y-%m-%d %X'))
print('numpy version = ' + np.__version__)
print('pandas version = ' + pd.__version__)


output_dir = preproc_config.data_dir['met_data']

# local winter time is UTC+2
start_dt = '2016-04-01 00:00:00'
if not args.flag_now:
    end_dt = '2016-11-11 00:00:00'
else:
    end_dt = (datetime.datetime.utcnow() +
              datetime.timedelta(2. / 24.)).strftime('%Y-%m-%d %H:%M:%S')


# variable names for retrieval from the SMEAR data website API
varnames = ['Pamb0', 'T1250', 'T672', 'T504', 'T336', 'T168', 'T84', 'T42',
            'RHIRGA1250', 'RHIRGA672', 'RHIRGA504', 'RHIRGA336',
            'RHIRGA168', 'RHIRGA84', 'RHIRGA42',
            'RPAR', 'PAR', 'diffPAR', 'maaPAR',
            'tsoil_humus', 'tsoil_A', 'tsoil_B1', 'tsoil_B2', 'tsoil_C1',
            'wsoil_humus', 'wsoil_A', 'wsoil_B1', 'wsoil_B2', 'wsoil_C1',
            'Precipacc']


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


# an url example
# url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?' +
#       'variables=Pamb0,&table=HYY_META&' +
#       'from=2016-04-01 00:00:00&to=2016-04-02 00:00:00&'
#       'quality=ANY&averaging=30MIN&type=ARITHMETIC'


if not args.flag_get_variable:
    # first, request all variables except precipitation
    print("Fetching variables '%s' ..." % ', '.join(varnames[0:-1]), end=' ')

    avg_type = 'ARITHMETIC'
    url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variables=' + \
        ','.join(varnames[0:-1]) + ',&table=HYY_META&from=' + \
        start_dt + '&to=' + end_dt + \
        '&quality=ANY&averaging=30MIN&type=' + avg_type

    response = requests.get(url, verify=True)
    # set `verify=True` to check SSL certificate
    if response.status_code != 200:
        print('Status %d: No response from the request.' %
              response.status_code)
    else:
        print('Successful!')

    df_met = pd.read_csv(
        io.BytesIO(response.text.encode('utf-8')), sep=',', header=0,
        names=['year', 'month', 'day', 'hour', 'minute', 'second',
               *varnames[0:-1]],
        parse_dates={'timestamp': [0, 1, 2, 3, 4, 5]},
        date_parser=timestamp_parser,
        engine='c', encoding='utf-8')

    start_year = df_met['timestamp'][0].year

    df_met.insert(
        1, 'doy',
        (df_met['timestamp'] - pd.Timestamp('%d-01-01' % start_year)) /
        pd.Timedelta(days=1))
    print('Timestamps parsed.')

    # mask zero pressure as NaN
    df_met.loc[df_met['Pamb0'] == 0., 'Pamb0'] = np.nan

    # append precipitation; it's treated separately due to different averaging
    del url, response
    print("Fetching variable '%s' ..." % varnames[-1], end=' ')
    avg_type = 'SUM'
    url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variables=' + \
        varnames[-1] + ',&table=HYY_META&from=' + \
        start_dt + '&to=' + end_dt + \
        '&quality=ANY&averaging=30MIN&type=' + avg_type

    response = requests.get(url, verify=True)
    # set `verify=True` to check SSL certificate
    if response.status_code != 200:
        print('Status %d: No response from the request.' %
              response.status_code)
    else:
        print('Successful!')

    df_precip = pd.read_csv(
        io.BytesIO(response.text.encode('utf-8')), sep=',', header=0,
        names=[varnames[-1]], usecols=[6],
        parse_dates=False,
        engine='c', encoding='utf-8')

    df_met = pd.concat([df_met, df_precip], axis=1)
else:
    # one variable a time
    # make a copy and insert custom timestamps
    colnames = copy.copy(varnames)
    colnames.insert(0, 'timestamp')
    colnames.insert(1, 'doy')

    df_met = pd.DataFrame(columns=colnames)

    flag_timestamp_parsed = False

    # fetch and dump data: dump each variable into TXT and combine 'em as CSV
    for var in varnames:
        print("Fetching variable '%s' ..." % var, end=' ')

        # precipitation must be summed not averaged over the 30 min interval
        if var != 'Precipacc':
            avg_type = 'ARITHMETIC'
        else:
            avg_type = 'SUM'

        url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variables=' + \
              var + ',&table=HYY_META&from=' + start_dt + '&to=' + end_dt + \
              '&quality=ANY&averaging=30MIN&type=' + avg_type

        response = requests.get(url, verify=True)
        # set `verify=True` to check SSL certificate
        if response.status_code != 200:
            print(
                "Status %d: No response from the request for variable '%s'." %
                (response.status_code, var))
            continue
        else:
            print('Successful!')

        if not flag_timestamp_parsed:
            fetched_data = pd.read_csv(
                io.BytesIO(response.text.encode('utf-8')), sep=',', header=0,
                names=['year', 'month', 'day',
                       'hour', 'minute', 'second', var],
                parse_dates={'timestamp': [0, 1, 2, 3, 4, 5]},
                date_parser=timestamp_parser,
                engine='c', encoding='utf-8')
        else:
            fetched_data = pd.read_csv(
                io.BytesIO(response.text.encode('utf-8')), sep=',', header=0,
                names=[var], usecols=[6],
                parse_dates=False,
                engine='c', encoding='utf-8')

        # if var == 'Pamb0':
        #     fetched_data[var][fetched_data[var] == 0.] = np.nan
        if var == 'Pamb0':
            fetched_data.loc[fetched_data[var] == 0., var] = np.nan

        if not flag_timestamp_parsed:
            # fill timestamps and convert to day of year
            df_met['timestamp'] = fetched_data['timestamp']
            flag_timestamp_parsed = True
            df_met['doy'] = (
                df_met['timestamp'] -
                pd.Timestamp('%d-01-01' %
                             fetched_data['timestamp'][0].year)) / \
                pd.Timedelta(days=1)
            print('Timestamps parsed.')

        df_met[var] = fetched_data[var]

        del url, response, fetched_data


# round met variables to '%.6f' except precipitation
# keep 'precip' as '%.2f'. nothing to be done for it
# do not round day of year variable 'doy'
df_met = df_met.round({var: 6 for var in varnames[0:-1]})

# renaming column names in the output dataframe
for col in df_met.columns.values:
    if col in renaming_dict:
        df_met.rename(columns={col: renaming_dict[col]}, inplace=True)

print('Variable fields have been renamed in the output data.')

df_met.to_csv(output_dir + '/hyy16_met_data.csv', na_rep='NaN', index=False)
print('Tabulated data written to %s/hyy16_met_data.csv' % output_dir)


# echo program ending
dt_end = datetime.datetime.now()
print(datetime.datetime.strftime(dt_end, '%Y-%m-%d %X'))
print('Done. Finished in %.2f seconds.' % (dt_end - dt_start).total_seconds())
