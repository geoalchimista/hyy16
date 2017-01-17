import datetime, copy, os, glob
import requests
import numpy as np
import pandas as pd

# fetch biomet variables from the SMEAR website and dump them as temporary text files
# bind the text files as a CSV table
# delete those temporary text files

output_dir = '/Users/wusun/Dropbox/Projects/Hyytiala_2016/met_data/'
temp_dir = output_dir + '/tmp/'

# local time is UTC+2
start_dt = '2016-04-01 00:00:00'
end_dt = (datetime.datetime.utcnow()+ datetime.timedelta(0, seconds=7200)).strftime('%Y-%m-%d %H:%M:%S')
varnames= ['Pamb0', 'T1250', 'T672', 'T504', 'T336', 'T168', 'T84', 'T42', 
  'RHIRGA1250', 'RHIRGA672', 'RHIRGA504', 'RHIRGA336', 'RHIRGA168', 'RHIRGA84', 'RHIRGA42', 
  'RPAR', 'PAR', 'diffPAR', 'maaPAR', 
  'tsoil_humus', 'tsoil_A', 'tsoil_B1', 'tsoil_B2', 'tsoil_C1',
  'wsoil_humus', 'wsoil_A', 'wsoil_B1', 'wsoil_B2', 'wsoil_C1']

colnames = copy.copy(varnames)
colnames.insert(0, 'timestamp')
colnames.insert(1, 'day_of_year')
combined_data = pd.DataFrame(columns = colnames)

# fetch and dump data
# an url sample
# url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variable=Pamb0&table=HYY_META&from=2016-04-01 00:00:00&to=2016-04-02 00:00:00&quality=ANY&averaging=30MIN&type=ARITHMETIC'
flag_valid_response = True
for var in varnames:
    url = 'http://avaa.tdata.fi/palvelut/smeardata.jsp?variable=' + var + '&table=HYY_META&from=' + start_dt + \
      '&to=' + end_dt + '&quality=ANY&averaging=30MIN&type=ARITHMETIC'
    print('Fetching variable ' + var + ' from SMEAR (http://avaa.tdata.fi/) ...')
    response = requests.get(url, verify=True) # Verify is check SSL certificate
    if response.status_code != 200:
        print('Status:', response.status_code, ' No response from the request for variable ' + var)
        flag_valid_response = False
    requested_data = response.text
    # print(data)
    output_fname = temp_dir + var + '.txt'
    fo = open(output_fname, 'w')
    fo.write(requested_data)
    fo.close()
    # reading data in as data table
    fetched_datatable = np.genfromtxt(output_fname, delimiter='"', dtype=(['float64', 'S64']), 
      names=[var, 'timestamp'], usecols=[0,1], invalid_raise=False)
    fetched_datatable[var][ fetched_datatable[var]==0. ] = np.nan
    combined_data[var] = fetched_datatable[var]
    # fill timestamps and convert to day of year
    if var == varnames[0]:
        combined_data['timestamp'] = np.empty(fetched_datatable.shape[0], dtype='|S32')
        for i in range(fetched_datatable.size):
            dt_str = str(datetime.datetime.strptime(fetched_datatable['timestamp'][i], '%b %d, %Y %I:%M:%S %p'))
            converted_doy = (datetime.datetime.strptime(fetched_datatable['timestamp'][i], '%b %d, %Y %I:%M:%S %p')
                - datetime.datetime(2016, 1, 1, 0, 0) ).total_seconds() / 86400.
            combined_data.set_value(i, 'timestamp', "'" + dt_str + "'") # use single quotation mark
            combined_data.set_value(i, 'day_of_year', converted_doy)
            del(dt_str, converted_doy)
        print('Time stamps parsed. ')
    del(url, response, requested_data, output_fname, fo, fetched_datatable)

combined_data.to_csv(output_dir + '/hyy_met_data.csv', sep=',', na_rep='NaN', index=False)  # do not output 'row name'
print('Tabulated data written to ' +  output_dir + '/hyy_met_data.csv')
print('Done.')
