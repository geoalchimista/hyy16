# -*- coding: utf-8 -*-
"""
Hyytiälä pine forest, Apr-now 2016
Raw chamber measurements data processing
Run in python 2.7; does not support python 3.5.

Revision
--------
04/12/2016
- Fitting plots output changed to multipage PDFs. 

04/20/2016
- Sanity check for fluxes added. Large fluxes will be set NaN. 

05/02/2016
- Adapted to the latest chamber schedules implemented on 28 Apr 2016. 
- Output IQR of conc's during closure periods as diagnostics for instrument drift. 

05/04/2016
- Chamber sensor data (PAR & temp) incorporated. 
- Time lag for leaf chambers are directly calculated from flow rates and included in the output 

05/05/2016
- Baseline endpoints for fitting changed from means to medians, for robustness against outliers. 
- Time lag for leaf chambers are marked on the conc fitting plots. 
- Fitting plots directories changed to their upper directory, because only one file is generated every day. 

05/06/2016
- Use gapfilled and downsampled (1 min step) flow data. 
- Leaf area of the aspen chamber reverted to 350 cm^2 to scale it with the fluxes from other chambers. 
Please multiply a factor of 350 when dealing with the data from the aspen chamber. 

05/24/2016
- Leaf chamber labels corrected. 

05/25/2016
- Daily plots changed to 3 columns. Fluxes from the large or extra large leaf chamber are plotted separately.

05/26/2016
- Leaf chamber schedule corrected. 'ch_no' now corresponds to the actually chambers. 

05/27/2016
- Flow rates recorded by Honeywell(R) flowmeters are actually in 'standard liter per minute'. Corrected.

06/09/2016
- Found errors in chamber labeling. Corrected them. 
- The meaning of shoot_no updated: 
  * shoot_no == 0: no chamber connected
  * shoot_no > 0: a leaf chamber
  * shoot_no == -1: a soil chamber

06/14/2016
- XL chamber volume (4.7 L) corrected. 

07/05/2016
- The time range of fitting plots for leaf chambers is extended to better show misidentified time lags. 
- Suppress all RuntimeWarning because of the annoying numpy "Mean of empty slice" warning
- Suppress UserWarning from matplotlib tight_layout()

07/06/2016
- Leaf areas and chamber volumes for 'Aux3' and 'BG' controlled chambers have been updated. 
- PAR assignment in June 03-15, 2016 has been corrected. 
- Manual water vapor correction has been turned off. 

07/07/2016
- A nonlinear time-lag optimization option has been implemented. [needs improvement of robustness]
- Output data columns changed: 'time_lag' -> 't_lag_nom'.
- New columns added: 't_lag_co2', 'status_tlag'.

07/28/2016
- Soil chamber #3 is actually a very large chamber, not the Li-Cor chamber. Its volume has been corrected.
  I do not have an estimate of its area, so for now I use 0.5 m^2 suggested by Linda Kooijmans as a placeholder. 

08/03/2016
- Soil chamber #3 bottom area and volume are updated with measurements provided by Ilona Ylivinkka at the site. 

08/05/2016
- Soil chamber #3 flow rates have been corrected with a function `func_flow_lsc`. 

09/06/2016
- Light/dark test on soil chamber 2: 08:21--14:57 UTC+2. 

09/20/2016
- Leaf area values updated. 
"""

import scipy.constants.constants as sci_consts    # scientific constants
import numpy as np
from scipy import stats, signal, optimize
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import os, glob, datetime, linecache, copy

plt.rcParams.update({'mathtext.default': 'regular'})  # san-serif math

import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)
# suppress the annoying numpy runtime warning of "mean of empty slice"
warnings.simplefilter('ignore', category=UserWarning)
# suppress the annoying matplotlib tight_layout user warning

print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))

'''
Configuration
-------------
'''
flag_run_current_day = False
# if False, run it over all the data; if True, run only the current whole day of data

flag_fitting_plots = False
flag_save_pdf_plots = False  # save plots into multipage pdf for each day, instead of png  
# warning: saving pdf plots slows the processing significantly

flag_use_gapfilled_flow_data = True  # use gapfilled flow data

flag_manual_water_correction = False
# automatic water vapor correction on the instrument is turned on, no need to correct manually

flag_new_data_generated = False # do not modify this # if it becomes True, that means new data are generated

flag_timelag_optmz = True

flag_test_mode = False  


'''
Functions
---------
'''
def IQR_func(x):
    if np.sum(np.isfinite(x)) > 0:
        q1, q3 = np.nanpercentile(x, [25,75])
        return(q3 - q1)
    else:
        return(np.nan)

'''
Physical constants
------------------
'''
p_std = sci_consts.atm    # 1 standard atm pressure in Pascal
# p_atm = p_std    # station atmospheric temperature, use measured values
R_gas = sci_consts.R
T_0 = sci_consts.zero_Celsius
air_conc_std = p_std / R_gas / T_0    # air concentration (mol m^-3) at STP

# define the start datetime of Mac convention, for converting QCL data time stamps to day of year (fractional)
mac_sec_start = datetime.datetime(1904,1,1,0,0)
mac_sec_2016 = (datetime.datetime(2016,1,1,0,0) - mac_sec_start).total_seconds()

'''
Chamber constants
-----------------
'''

''' variables 'A_ch' and 'V_ch' will be deprecated
# 6 chambers in total, including 3 leaf chambers, 3 soil chambers
# the third soil chamber was installed 04/28/2016 17:15
# only 2 soil chambers were effectively running before that
A_ch = np.zeros(6) * np.nan
V_ch = np.zeros(6) * np.nan

# leaf chambers
# lc_ht = 0.; lc_rad = 0. # V_ch[0:3] = lc_ht * sci_consts.pi * lc_rad**2
V_ch[0:3] = np.array([1.8, 1.8, 3.5]) * 1e-3
A_ch[0:3] = np.array([350., 350., 350.]) * 1e-4  # leaf areas in m^2

# soil chambers
A_ch[3:] = 317.8e-4  # m^2, LI-8100-104 long term chamber
col_ht = 2e-2  # collar height in m
V_ch[3:] = 4076.1e-6 + col_ht * A_ch[3:]  # m^3
'''

# leaf chamber constants
V_lc_small = 1.8e-3  # small chamber 1.8 L
V_lc_large = 3.5e-3  # large chamber 3.5 L
V_lc_XL = 4.7e-3  # extra large chamber 4.7 L
V_lc_slide = 1.0e-3  # slide chamber 1.0 L

# soil chamber constants
A_sc = 317.8e-4  # m^2, LI-8100-104 long term chamber
col_ht = 2e-2  # collar height in m
V_sc = 4076.1e-6 + col_ht * A_sc  # m^3

# transparent soil chambers
A_tsc = (22. - 2 * 0.3) ** 2 * np.pi / 4. * 1e-4
V_tsc = (7.25 + 4.) * 1e-2 * A_tsc

# large soil chamber (soil chamber #3)
A_lsc = 0.4 * 0.8 # m^2
V_lsc = 0.4 * 0.8 * (0.17 + 0.10) # 80. * 1e-3   # m^3
sd_V_lsc = 0.4 * 0.8 * 0.02  # m^3  
# std dev of volume has not been used yet, will be incorporated in future calculations


def chamber_lookup_table_func(doy):
    '''
    Return a chamber meta information look-up table (pandas.DataFrame)
    '''
    if doy < 103. + 11./24.:  # from the beginning to 04/13/2016 11:00 (UTC+2)
        chlut = pd.DataFrame([1,2,3,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-B', 'LC-S-A', 'LC-L-A', 'SC1', 'SC2']
        chlut['shoot_no'] = [3, 1, 2, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 5, 4]
        chlut['TC_no'] = [2, 1, 3, 4, 5]
        chlut['PAR_no'] = [2, 1, 1, -1, -1]
    elif (doy >= 103. + 11./24.) and (doy < 111.5):
        # from 04/13/2016 11:00 (UTC+2) to roughly 04/21/2016 12:00 (UTC+2)
        chlut = pd.DataFrame([1,2,3,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-B', 'LC-S-A', 'LC-L-A', 'SC1', 'SC2']
        chlut['shoot_no'] = [6, 4, 5, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 5, 4]
        chlut['TC_no'] = [2, 1, 3, 4, 5]
        chlut['PAR_no'] = [2, 1, 1, -1, -1]
        if doy >= 108. + 8./24. + 25./1440.:
            # from 04/18/2016 08:25 (UTC+2) to 04/20/2016 13:40 (UTC+2)
            # testing 'LC-XL' at Aux4 channel
            # LC-XL had rubber material parts during this period, so label it 'LC-XL+rubber'
            chlut = chlut.set_value(0, 'ch_label', 'LC-XL+rubber')
            chlut = chlut.set_value(0, 'shoot_no', 7)
    elif (doy >= 111.5) and (doy < 119.):
        # from 04/21/2016 12:00 (UTC+2) to 04/28/2016 17:00 (UTC+2)
        chlut = pd.DataFrame([1,2,3,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-A', 'LC-S-B', 'LC-XL', 'SC1', 'SC2']
        chlut['shoot_no'] = [9, 8, 10, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 5, 4]
        chlut['TC_no'] = [2, 1, 3, 4, 5]
        chlut['PAR_no'] = [2, 2, 1, -1, -1]
        if doy >= 112. + 8.5/24.:  # an aspen bud is placed in the XL chamber on 04/22/2016 08:29 (UTC+2)
            chlut = chlut.set_value(2, 'shoot_no', 11)
    elif (doy >= 119.) and (doy < 140. + 11.5/24.):
        # from 04/29/2016 00:00 (UTC+2) to 05/20/2016 11:30 (UTC+2)
        chlut = pd.DataFrame([1,2,3,6,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-A', 'LC-S-B', 'LC-XL', 'SC3', 'SC1', 'SC2']
        chlut['shoot_no'] = [9, 8, 11, -1, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 4, 5, 4] # 'SC3' does not have its own flowmeter
        chlut['TC_no'] = [2, 1, 3, 6, 4, 5]  # 'SC3' indeed has its own thermocouple
        chlut['PAR_no'] = [2, 2, 1, -1, -1, -1]
    elif (doy >= 140. + 11.5/24.) and (doy < 145. + (10.+11./60.)/24.):
        # from 05/20/2016 11:30 (UTC+2) to 05/25/2016 10:11 (UTC+2)
        # 'S-A' and 'S-B' are swapped, the places of other sensors are assumed unchanged
        chlut = pd.DataFrame([1,2,3,6,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-B', 'LC-S-A', 'LC-XL', 'SC3', 'SC1', 'SC2']
        chlut['shoot_no'] = [9, 8, 11, -1, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 4, 5, 4] # 'SC3' does not have its own flowmeter
        chlut['TC_no'] = [2, 1, 3, 6, 4, 5]  # 'SC3' indeed has its own thermocouple
        chlut['PAR_no'] = [2, 2, 1, -1, -1, -1]
    elif (doy >= 145. + (10.+11./60.)/24.) & (doy < 154. + (14.+8./60.)/24.):
        # from 05/25/2016 10:11 (UTC+2) to 06/03/2016 14:08 (UTC+2)
        chlut = pd.DataFrame([1,2,3,6,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-A', 'LC-S-B', 'LC-XL', 'SC3', 'SC1', 'SC2']
        chlut['shoot_no'] = [12, 8, 11, -1, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 4, 5, 4] # 'SC3' does not have its own flowmeter
        chlut['TC_no'] = [2, 1, 3, 6, 4, 5]  # 'SC3' indeed has its own thermocouple
        chlut['PAR_no'] = [2, 2, 1, -1, -1, -1]
    elif (doy >= 154. + (14.+8./60.)/24.) and (doy < 166.5):
        # 'LC-S-A' changed to 'LC-slide' 06/03/2016 14:08 (UTC+2); other sensors unchanged
        chlut = pd.DataFrame([1,2,3,6,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-Slide', 'LC-S-B', 'LC-XL', 'SC3', 'SC1', 'SC2']
        chlut['shoot_no'] = [13, 8, 11, -1, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 4, 5, 4] # 'SC3' does not have its own flowmeter
        chlut['TC_no'] = [2, 1, 3, 6, 4, 5]  # 'SC3' indeed has its own thermocouple
        chlut['PAR_no'] = [1, 2, 1, -1, -1, -1]
        if doy >= 158. + 9./24.: # an aspen branch was placed in the chamber 06/07/2016 09:00 (UTC+2) 
            chlut = chlut.set_value(0, 'shoot_no', 14)
    elif doy >= 166.5: 
        # after 06/15/2016 12:00 UTC+2
        # 'LC-Slide' (first in the sequence) --> 'LC-S-A', activated by 'Aux4' valve
        # 'LC-XL' (3rd in the sequence) --> 'LC-Slide', activated by 'BG' valve
        # leaf areas and chamber volumes have also been changed; see the control statement for 'A_ch' and 'V_ch'
        chlut = pd.DataFrame([1,2,3,6,4,5], columns=['ch_no'])
        chlut['ch_label'] = ['LC-S-A', 'LC-S-B', 'LC-Slide', 'SC3', 'SC1', 'SC2']
        chlut['shoot_no'] = [12, 8, 15, -1, -1, -1]
        chlut['flowmeter_no'] = [1, 2, 3, 4, 5, 4] # 'SC3' does not have its own flowmeter
        chlut['TC_no'] = [2, 1, 3, 6, 4, 5]  # 'SC3' indeed has its own thermocouple
        chlut['PAR_no'] = [2, 2, 1, -1, -1, -1]
        if doy >= 175. + 10./24.: # chamber 'S-B' moved to another pine branch at 06/24/2016 10:05 (UTC+2)
            chlut = chlut.set_value(1, 'shoot_no', 16)
            chlut = chlut.set_value(1, 'PAR_no', 1) # PAR channel was also changed
        if (doy >= 241.5) and (doy <= 255. + 11./24.):
            # from 08/29/2016 morning to 09/12/2016 10:55 (UTC+2) some blank measurements were made
            chlut = chlut.set_value(2, 'shoot_no', 17)  # no. 17 is for the blank on the slide chamber
            if (doy >= 245. + (9. + 40./60.) / 24.) and (doy <= 252. + (9. + 31./60.) / 24.):
                # from 09/02/2016 09:40 (UTC+2) and 09/09/2016 09:31 (UTC+2)
                chlut = chlut.set_value(0, 'shoot_no', 19)  # no. 19 is for the blank on chamber S-A
            else:
                chlut = chlut.set_value(1, 'shoot_no', 18)  # no. 18 is for the blank on chamber S-B

    # --------------------
    # add chamber sequence
    # --------------------
    # a new control statement
    if doy < 119.:
        '''
        Chamber sampling sequence (before 04/28/2016 17:00 UTC+2)
        Loop through 1-hour period
        Sampling sequence:
        00:00-06:00: 0.5 m line, and every 6 hours a background measurement
        06:00-08:30: cylinder 1 (port 10)
        08:30-11:00: cylinder 2 (port 12)
        # --- leaf chamber 1 ---
        11:00-12:00: open leaf chamber 1 (small-B; shoots changed after 04/21/2016)
        12:00-16:00: closed leaf chamber 1
        16:00-19:00: open leaf chamber 1
        # --- leaf chamber 2 ---
        19:00-20:00: open leaf chamber 2 (small-A; shoots changed after 04/21/2016)
        20:00-24:00: closed leaf chamber 2
        24:00-27:00: open leaf chamber 2
        # --- leaf chamber 3 ---
        27:00-28:00: open leaf chamber 3 (large)
        28:00-32:00: closed leaf chamber 3
        32:00-35:00: open leaf chamber 3
        35:00-37:00: 125 m
        37:00-39:00: 4 m
        39:00-41:30: cylinder 2 (port 12)
        41:30-43:00: 0.5 m 
        # --- soil chamber 1 ---
        43:00-50:00: closed soil chamber 1
        50:00-52:00: 0.5 m
        # --- soil chamber 2 ---
        52:00-59:00: closed soil chamber 2
        59:00-00:00: 0.5 m
        '''
        chlut['ch_start'] = np.array([11., 19., 27., 41.5, 50.]) / 1440.
        chlut['ch_o_b'] = np.array([0., 0., 0., 0., 0.]) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.array([1., 1., 1., 1.5, 2.]) / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.array([5., 5., 5., 8.5, 9.]) / 1440. # with respect to ch_start
        chlut['ch_end'] = np.array([8., 8., 8., 10.5, 10.]) / 1440. # with respect to ch_start 
        '''
        Note 1: For leaf chambers, because there are about 2 minutes of time lag in the sampling lines, 
        for the opening period after chamber closure, do not use all 3 minutes of them. 
        Instead, use only the measurements before the pressure spikes induced by line switching. 
        The time when pressure spike occurs is fixed (8 min after switching to the current chamber line), 
        therefore the time lag should not be added to it. 
        Note 2: Time lag shall be added to the schedules later (see 'time_lag_in_day' variable). 
        Note 3: For soil chambers, no time lag is assumed. 
        '''
    else:
        '''
        Chamber sampling sequence (after 04/28/2016 17:00 UTC+2)
        Loop through 1.5-hour period
        Sampling sequence:
        00:00-06:00: 0.5 m line, and every 6 hours a background measurement
        06:00-09:00: cylinder 1 (port 10)
        09:00-12:00: cylinder 2 (port 12)
        # --- leaf chamber 1 ---
        12:00-13:00: open leaf chamber 1 (small-B)
        13:00-17:00: closed leaf chamber 1
        17:00-21:00: open leaf chamber 1
        # --- leaf chamber 2 ---
        21:00-22:00: open leaf chamber 2 (small-A)
        22:00-26:00: closed leaf chamber 2
        26:00-30:00: open leaf chamber 2
        # --- leaf chamber 3 ---
        30:00-31:00: open leaf chamber 3 (large)
        31:00-35:00: closed leaf chamber 3
        35:00-39:00: open leaf chamber 3
        39:00-42:00: cylinder 2 (port 12)
        42:00-46:00: 125 m
        46:00-49:00: 4 m
        49:00-51:00: 0.5 m
        # --- soil chamber 3 ---
        51:00-59:00: closed soil chamber 3 (new in place!)
        59:00-62:00: 0.5 m
        62:00-65:00: cylinder 2 (port 12)
        65:00-68:00: 0.5 m
        # --- soil chamber 1 ---
        68:00-76:00: closed soil chamber 1
        76:00-79:00: 0.5 m
        # --- soil chamber 2 ---
        79:00-87:00: closed soil chamber 2
        87:00-90:00: 0.5 m
        '''
        # note, the variables describing the schedule is following ch_no 1-2-3-6-4-5
        # soil chamber 3 comes before soil chambers 1 and 2
        chlut['ch_start'] = np.array([12., 21., 30., 49., 65., 76.]) / 1440.
        chlut['ch_o_b'] = np.array([0., 0., 0., 0., 0., 0.]) / 1440. # with respect to ch_start
        chlut['ch_cls'] = np.array([1., 1., 1., 2., 3., 3.]) / 1440. # with respect to ch_start
        chlut['ch_o_a'] = np.array([5., 5., 5., 10., 11., 11.]) / 1440. # with respect to ch_start
        chlut['ch_end'] = np.array([9., 9., 9., 13., 14., 14.]) / 1440. # with respect to ch_start
        '''
        Note 1: For leaf chambers, because there are about 2 minutes of time lag in the sampling lines, 
        for the opening period after chamber closure, do not use all 4 minutes of them. 
        Instead, use only the measurements before the pressure spikes induced by line switching. 
        The time when pressure spike occurs is fixed (9 min after switching to the current chamber line), 
        therefore the time lag should not be added to it. 
        Note 2: Time lag shall be added to the schedules later (see 'time_lag_in_day' variable). 
        Note 3: For soil chambers, no time lag is assumed. 
        '''
    # --------------------------------------------
    # add leaf and soil areas, and chamber volumes
    # --------------------------------------------
    # a new control statement
    if doy < 119.:
        chlut['A_ch'] = np.array([130., 170., 350., 317.8, 317.8]) * 1e-4
        chlut['V_ch'] = np.array([V_lc_small, V_lc_small, V_lc_large, V_sc, V_sc])
        if (doy >= 108. + 8./24. + 25./1440.) and (doy < 110. + 13./24. + 40./1440.):
            # from 04/18/2016 08:25 (UTC+2) to 04/20/2016 13:40 (UTC+2)
            # testing 'LC-XL' at Aux4 channel
            chlut = chlut.set_value(0, 'A_ch', 130e-4)
            chlut = chlut.set_value(0, 'V_ch', V_lc_XL)
        if doy > 111.5:
            # chamber 'XL' changed to the aspen bud
            chlut = chlut.set_value(2, 'A_ch', 1e-4)
            chlut = chlut.set_value(2, 'V_ch', V_lc_XL)
    elif doy >= 119.:
        # soil chamber 3 added (a very large chamber)
        chlut['A_ch'] = np.array([130e-4, 170e-4, 1e-4, A_lsc, 317.8e-4, 317.8e-4]) 
        chlut['V_ch'] = np.array([V_lc_small, V_lc_small, V_lc_XL, V_lsc, V_sc, V_sc])
        if doy >= 131.5:
            # soil chamber 2 replaced with the transparent chamber
            # installation happened between 7:47 and 13:19 05/11/2016 UTC+2
            # note: soil chamber 2 (ch #5) is the last one in the sequence
            chlut = chlut.set_value(5, 'A_ch', A_tsc)
            chlut = chlut.set_value(5, 'V_ch', V_tsc)
        if doy >= 135.5:
            # aspen bud leafed out on 05/15/2016
            # the most recent leaf area measure is 600 cm^2
            chlut = chlut.set_value(2, 'A_ch', np.interp(doy, [135.5, 145.5], [1e-4, 930e-4]) )
        if doy >= 145. + (10.+11./60.)/24.:
            # shoots in chamber 'S-A' changed on 05/25/2016 10:11 (UTC+2)
            chlut = chlut.set_value(0, 'A_ch', 300e-4)
        if doy >= 154. + (14.+8./60.)/24.:
            # 'LC-S-A' changed to 'LC-slide' 06/03/2016 14:08 (UTC+2)
            chlut = chlut.set_value(0, 'A_ch', 190e-4)
            chlut = chlut.set_value(0, 'V_ch', V_lc_slide)
        if doy >= 166.5: 
            # after 06/15/2016 12:00 UTC+2
            # 'LC-Slide' (first in the sequence) --> 'LC-S-A', activated by 'Aux4' valve
            # 'LC-XL' (3rd in the sequence) --> 'LC-Slide', activated by 'BG' valve
            chlut = chlut.set_value(0, 'A_ch', 300e-4)
            chlut = chlut.set_value(0, 'V_ch', V_lc_small)
            chlut = chlut.set_value(2, 'A_ch', 190e-4)
            chlut = chlut.set_value(2, 'V_ch', V_lc_slide)
        if doy >= 175. + 10./24.: # chamber 'S-B' moved to another pine branch at 06/24/2016 10:05 (UTC+2)
            chlut = chlut.set_value(1, 'A_ch', 250e-4)

    # -----------------------------------------------------
    # set shoot_no to zero for absolutely no chamber period
    # -----------------------------------------------------
    # a new control statement
    if (doy > 110. + (13.+40./60.)/24.) and (doy < 111. + (13.+15./60.)/24.):
        # from 04/20/2016 13:40 (UTC+2) to 04/21/2016 13:15 (UTC+2), no chamber at Aux4 channel
        # set 'shoot_no' to zero but keep the chamber label unchanged
        chlut = chlut.set_value(0, 'shoot_no', 0)
    if (doy > 111. + (8.+20./60.)/24.) and (doy < 111. + (13.+15./60.)/24.):
        # from 04/21/2016 08:20 (UTC+2) to 04/21/2016 13:15 (UTC+2), no chamber at Aux3 channel
        # set 'shoot_no' to zero but keep the chamber label unchanged
        chlut = chlut.set_value(1, 'shoot_no', 0)
    if (doy > 111. + (8.+20./60.)/24.) and (doy < 111. + (12.+20./60.)/24.):
        # from 04/20/2016 13:40 (UTC+2) to 04/21/2016 13:15 (UTC+2), no chamber at backgr channel
        # set 'shoot_no' to zero but keep the chamber label unchanged
        chlut = chlut.set_value(2, 'shoot_no', 0)
    if (doy > 154. + (13.+40./60.)/24.) and (doy < 154. + (14.+8./60.)/24.):
        # from 06/03/2016 13:45 (UTC+2) to 06/03/2016 14:08 (UTC+2), no chamber at Aux4 channel
        # however, the replacement happened at when Aux4-chamber was being measured, so toss at least that data point
        chlut = chlut.set_value(0, 'shoot_no', 0)
    if (doy > 158. + (9./24.)) and (doy < 158. + (10./24.)):
        # from 06/07/2016 09:00 (UTC+2) to 06/07/2016 10:00 (UTC+2), no chamber at Aux4 channel
        chlut = chlut.set_value(0, 'shoot_no', 0)
    return(chlut)



# the timelag optimization function to fit
def func_conc_resid(t_lag, time, conc, t_turnover, 
                    dt_open_before = 60., dt_close = 240., dt_open_after = 180., 
                    dt_left_margin = 0., dt_right_margin = 0., 
                    closure_period_only = False):
    """
    parameter to optimize: t_lag - time lag, in sec
    inputs
    * time - time since switching to chamber line, in sec
    * conc - concentration (in its most convenient unit)
    * t_turnover - turnover time, `V_ch_mol` (in mol) divided by `f_ch` (in mol/s)
    """
    # all index arrays should only contain the indices of finite `conc` values
    _ind_chb = np.where( (time >= t_lag + dt_left_margin) & (time < t_lag + dt_open_before - dt_right_margin) & 
                        np.isfinite(time) & np.isfinite(conc) )
    _ind_chc = np.where( (time >= t_lag + dt_open_before + dt_left_margin) & 
                        (time < t_lag + dt_open_before + dt_close - dt_right_margin) & 
                        np.isfinite(time) & np.isfinite(conc) )
    _ind_cha = np.where( (time >= t_lag + dt_open_before + dt_close + dt_left_margin) & 
                        (time < t_lag + dt_open_before + dt_close + dt_open_after - dt_right_margin) & 
                        np.isfinite(time) & np.isfinite(conc) )
    
    _median_chb = np.nanmedian( conc[_ind_chb] )
    _median_cha = np.nanmedian( conc[_ind_cha] )
    _t_mid_chb = np.nanmedian( time[_ind_chb] )
    _t_mid_cha = np.nanmedian( time[_ind_cha] )
    
    # baseline
    _k_bl = (_median_cha - _median_chb) / (_t_mid_cha - _t_mid_chb)
    _b_bl = _median_chb - _k_bl * _t_mid_chb
    _conc_bl = _k_bl * time + _b_bl
    
    _x_obs = np.exp( - (time[_ind_chc] - t_lag - dt_open_before) / t_turnover )
    _y_obs = conc[_ind_chc] - _conc_bl[_ind_chc]
    
    if _x_obs.size == 0: return(np.nan)
    # if no valid observations in chamber closure period, return NaN value
    # this will terminate the optimization procedure, and returns a 'status code' of 1 in `optimize.minimize`

    _slope, _intercept, _r_value, _p_value, _sd_slope = stats.linregress( _x_obs, _y_obs )
    _y_fitted = _slope * _x_obs + _intercept

    if closure_period_only:
        MSR = np.nansum((_y_fitted - _y_obs)**2) / (_ind_chc[0].size - 2) # mean squared residual
    else:
        '''
        MSR = ( np.nansum((_y_fitted - _y_obs)**2) + np.nansum((conc[_ind_chb] - _median_chb)**2) + \
              np.nansum((conc[_ind_cha] - _median_cha)**2) ) / (_ind_chc[0].size - 2 + _ind_chb[0].size - 1 + _ind_cha[0].size - 1)
        '''
        _conc_fitted = _slope * np.exp( - (time - t_lag - dt_open_before) / t_turnover ) + _intercept + _conc_bl
        _conc_fitted[ (time < t_lag + dt_open_before) & (time > t_lag + dt_open_before + dt_close) ] = _conc_bl[ (time < t_lag + dt_open_before) & (time > t_lag + dt_open_before + dt_close) ]
        # MSR = np.nansum((conc - _conc_fitted)**2) / (np.sum(np.isfinite(conc - _conc_fitted)) - 4.)  # degree of freedom = 4
        resid = conc - _conc_fitted
        resid_trunc = resid[ time <= t_lag + dt_open_before + dt_close ]  # do not include the chamber open period after closure
        MSR = np.nansum(resid_trunc**2) / (np.sum(np.isfinite(resid_trunc)) - 3.)  # degree of freedom = 3
    return(MSR)

# a function to interpolate the flow rates in the large soil chamber (soil chamber #3)
def func_flow_lsc(day_of_year):
    """
    measured values 
    datetime                  | doy_int | flow rates
    2016-04-21 12:52 (UTC+2)  |   112   |  3.75 slpm
    2016-07-04 ??             |   186   |  2.65 slpm
    2016-07-07 11:40 (UTC+2)  |   189   |  3.19 slpm
    2016-07-07 11:52 (UTC+2)  |   189   |  4.00 slpm
    """
    doy_lsc = [111.+12./24.+52./1440., 185.5, 188.+11./24.+40./1440., 188.+11./24.+52./1440., ]
    flow_lsc = [3.75, 2.65, 3.19, 4.00, ]
    flow_interp = np.interp(day_of_year, doy_lsc, flow_lsc)
    return(flow_interp)

'''
Working directories
-------------------
'''
data_dir = '../../Volumes/Perovskite/projects/ulli_lab/2016_Finland/'    # if working on my hard drive
plot_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/plots/'
output_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/processed/chflux/'

qcl_dir = data_dir + '/QCL_data/'  # QCL data
# met_dir = data_dir + '/met_data/'  # Met data on my hard drive
met_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/met_data/'
flo_dir = qcl_dir + '/Flowmeter/'  # flow rates data
sensor_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/sensor_data/corrected/'

if flag_run_current_day:
    data_dir = '/Users/wusun/Dropbox/QCLdata/'
    qcl_dir = '/Users/wusun/Dropbox/QCLdata/'
    flo_dir = qcl_dir + '/Flowmeter/'

flo_gapfilled_dir = '/Users/wusun/Dropbox/Projects/hyytiala_2016/flow_data/'  # gapfilled flow data

'''
Reading raw data files
----------------------
'''
# Note: by default, both met data and QCL data are on Finnish non-daylight-saving time (UTC+2)
# This is different from last year (2015) when QCL data was on UTC time.

met_fname = met_dir + '/hyy_met_data.csv'  # use the data fetched by scripts
qcl_flist = glob.glob(qcl_dir + '/*.str')
sensor_flist = glob.glob(sensor_dir + '/*.csv')

if not len(qcl_flist):
    print('Cannot find QCL data file!')

# use new header for met data
met_data = np.genfromtxt(met_fname, skip_header=1, delimiter=',',
    names=['timestamp','day_of_year', 'p_atm',
    'T_air_125m','T_air_67m','T_air_50m','T_air_34m','T_air_17m','T_air_8m','T_air_4m',
    'RH_125m','RH_67m','RH_50m','RH_34m','RH_17m','RH_8m','RH_4m',
    'PAR_reflected', 'PAR', 'PAR_diffuse', 'PAR_below', 
    'T_soil_surf','T_soil_A','T_soil_B1','T_soil_B2','T_soil_C1',
    'w_soil_surf','w_soil_A','w_soil_B1','w_soil_B2','w_soil_C1',], 
    invalid_raise=False, dtype=None)

print(str(met_data.size) + ' lines read from the SMEAR met data file')
# note: precipitation values are not included
# download precipitation data separately. they should be summed over the 30 min period, not averaged

met_doy = copy.copy(met_data['day_of_year'])
met_doy_utc = met_doy - 2./24.  # UTC time
# from examining the SMEAR data, it seems they are not on daylight saving time

# linearly interpolate p_atm and T_air over time for data processing
p_atm = np.interp(met_doy, met_doy[ np.isfinite(met_data['p_atm'])], met_data[ np.isfinite(met_data['p_atm']) ]['p_atm'] )
T_air_4m = np.interp(met_doy, met_doy[ np.isfinite(met_data['T_air_4m'])], met_data[ np.isfinite(met_data['T_air_4m']) ]['T_air_4m'] ) # for soil chamber air temp
T_air_8m = np.interp(met_doy, met_doy[ np.isfinite(met_data['T_air_8m'])], met_data[ np.isfinite(met_data['T_air_8m']) ]['T_air_8m'] ) # for lower canopy leaf chamber air temp
T_air_17m = np.interp(met_doy, met_doy[ np.isfinite(met_data['T_air_17m'])], met_data[ np.isfinite(met_data['T_air_17m']) ]['T_air_17m'] ) # for upper canopy leaf chamber air temp

# new: read flowmeter data tables
# columns: 0 - time, 1 - outflow?, 2 to 6 - inflows
# chamber assignments (starting from col2)
# before Apr 21: flow1 - 'LC-S-B'; flow2 - 'LC-S-A'; ch3 - 'LC-L-A' or 'LC-XL'; ch4 - 'SC-1'; ch5 - 'SC-2'; flow for ch6 is missing, use ch5 - 'SC-3'
# after Apr 21: flow2 - 'LC-S-B'; flow1 - 'LC-S-A'; ch3 - 'LC-XL'; ch4 - 'SC-1'; ch5 - 'SC-2'; flow for ch6 is missing, use ch5 - 'SC-3'
if flag_use_gapfilled_flow_data:
    print('Reading gapfilled flow data...')
    flow_data = np.array([])
    flo_flist = glob.glob(flo_gapfilled_dir + '/*.csv')
    for i in range(len(flo_flist)):
        flo_fname = flo_flist[i]
        flow_data_loaded = np.genfromtxt(flo_fname, skip_header=1, invalid_raise=False, delimiter=',')
        if flow_data.size > 0:
            flow_data = np.concatenate((flow_data, flow_data_loaded))
        else: flow_data = flow_data_loaded
        del(flow_data_loaded)
    flow_doy = flow_data[:,0]
else:
    print('Reading raw flow data (not gapfilled)... ')
    flow_data = np.array([])
    for i in range(40,400):
        flo_fname = flo_dir + '/data_' + '%d' % i + '.dat'
        if os.path.isfile(flo_fname) == False:
            continue
        flow_data_loaded = np.genfromtxt(flo_fname, skip_header=0, invalid_raise=False)
        if flow_data.size > 0:
            flow_data = np.concatenate((flow_data, flow_data_loaded))
        else: flow_data = flow_data_loaded
        del(flow_data_loaded)
    flow_doy = (flow_data[:,0] - mac_sec_2016) / 86400.  # convert seconds to day of year
    # remove NaNs from flow data by interpolation
    for i in range(2,7):
        flow_data[:,i] = np.interp(flow_doy, flow_doy[np.isfinite(flow_data[:,i])], flow_data[ np.isfinite(flow_data[:,i]) ,i])

# local time is UTC+2
doy_today = (datetime.datetime.utcnow() - datetime.datetime(2016,1,1)).total_seconds() / 86400. + 2./24.

# starting date: 04/07/2016
# doy_start = 97.  # fractional DOY
if flag_run_current_day:
    doy_start = np.int(doy_today - 3.)
    # doy_start = 97. # test
else: doy_start = 97.
doy_end = np.int(doy_today)
# doy_start = doy_end - 3. # test

# daily processing starts
for doy in np.arange(doy_start, doy_end):
    # note, 'doy' here is the fractional DOY, which is no larger than integer DOY (Julian day number)
    run_date_str = (datetime.datetime(2016,1,1) + datetime.timedelta(doy+0.5)).strftime("%y%m%d")
    # reading chamber sensor data
    current_sensor_files = [s for s in sensor_flist if run_date_str in s]
    n_sensor_files = len(current_sensor_files)
    sensor_data = np.array([])
    if n_sensor_files > 0:
        for i in range(n_sensor_files):
            sensor_data_loaded = np.genfromtxt(current_sensor_files[i], delimiter=',',
                names=['day_of_year','PAR_1','PAR_2','T_amb','T_ch1','T_ch2','T_ch3','T_ch4','T_ch5','T_ch6'],
                dtype='f8', invalid_raise=False)
            if sensor_data.size:
                sensor_data = np.concatenate((sensor_data, sensor_data_loaded))
            else:
                sensor_data = np.copy(sensor_data_loaded)
            del(sensor_data_loaded)
        print(str(sensor_data.size) + ' lines read from chamber sensor data. ')
    else:
        print('No sensor data for chambers are found. Will use met data for approximation. ')
    # reading QCL data
    current_qcl_files = [s for s in qcl_flist if run_date_str in s]
    # check QCL data existence; skip if not exist
    n_qcl_files = len(current_qcl_files)
    if n_qcl_files == 0:
        print ('QCL data file not found on day ' + run_date_str)
        continue
    print('Reading raw data files for day 20' + run_date_str + '...')
    # read QCL data
    qcl_data = np.array([])
    for loop_num in range(n_qcl_files):
        qcl_data_loaded = np.genfromtxt(current_qcl_files[loop_num], skip_header=1, names=['time', 'cos', 'co2_2', 'co2', 'h2o', 'co', 'cos_2', 'co2_3'], 
            invalid_raise=False)
        # print(qcl_data_loaded.size) # for test
        if qcl_data.size:
            qcl_data = np.concatenate((qcl_data, qcl_data_loaded))
        else:
            qcl_data = qcl_data_loaded
    print(str(qcl_data.size) + ' lines read from QCL data file(s) on day 20' + run_date_str)
    
    # legacy code; this won't be executed
    if flag_manual_water_correction:
        # --- QCL measurements calibration ----
        # apply water vapor corrections to QCL data (from Linda Kooijmans)
        # '1e-7' converts H2O to percentage of air
        # updated 24 Nov 2015
        qcl_data['cos'] = qcl_data['cos'] / (1 + 0.029 * qcl_data['h2o'] * 1e-7)
        qcl_data['co2'] = qcl_data['co2'] / (1 - 0.0145 * qcl_data['h2o'] * 1e-7)
        qcl_data['co'] = qcl_data['co'] / (1 - 0.009 * qcl_data['h2o'] * 1e-7)
        # slopes and intercepts of calibration curves (from Linda Kooijmans)
        # for COS, currently no calibration-correction
        cal_slp_cos = 1. #1.01
        cal_slp_co2 = 1.038 # 1.0385
        cal_slp_co = 0.975
        cal_int_cos = 0. # 14.931
        cal_int_co2 = -0.839 # -0.847
        cal_int_co = 2.173 # 2.115
        
        qcl_data['cos'] = (cal_int_cos + (qcl_data['cos'] * 1e3) * cal_slp_cos) * 1e-3
        qcl_data['co2'] = (cal_int_co2 + (qcl_data['co2'] * 1e-3) * cal_slp_co2) * 1e3
        qcl_data['co'] = cal_int_co + qcl_data['co'] * cal_slp_co
        # --- end of QCL calibration ---

    ## define the start datetime of Mac convention, for converting QCL data time stamps to day of year (fractional)
    #mac_sec_start = datetime.datetime(1904,1,1,0,0)
    #mac_sec_2016 = (datetime.datetime(2016,1,1,0,0) - mac_sec_start).total_seconds()
    qcl_doy = (qcl_data['time'] - mac_sec_2016) / 86400.
    # note: qcl_doy is fractional, e.g. doy of Jan 1st is a number between 0 and 1; integer doy is `numpy.ceil(doy)'
    doy_int = np.int(np.ceil(np.median(qcl_doy)))    # the integer doy  # a number
    
    chlut_today = chamber_lookup_table_func(doy + 0.5)  # get today's chamber lookup table
    ch_start = chlut_today['ch_start'].values
    ch_o_b = chlut_today['ch_o_b'].values # with respect to ch_start
    ch_cls = chlut_today['ch_cls'].values # with respect to ch_start
    ch_o_a = chlut_today['ch_o_a'].values # with respect to ch_start
    ch_end = chlut_today['ch_end'].values # with respect to ch_start 
    if doy <= 118.: 
        smpl_seq_len = 1. / 24.    # length of one full sampling sequence is 1/24 day or 1 hour before and on 04/28/2016
    elif doy > 118.: 
        smpl_seq_len = 1.5 / 24.   # starting 04/29/2016, a full cycle is 1.5 hour
    
    n_ch = chlut_today.shape[0]
    n_loop_per_day = np.int(1. / smpl_seq_len)  # one hour per sampling sequence
    n_smpl_per_day = n_ch * n_loop_per_day
    ch_no = np.tile(chlut_today['ch_no'].values, n_loop_per_day)
    
    # an exception on 04/19/2016: data after 18:00 are corrupt
    if doy == 109.: 
        n_smpl_per_day = n_ch * 18
        ch_no = ch_no[0:n_smpl_per_day]
    # an exception on 04/28/2016: only use data before 17:00
    if doy == 118.: 
        n_smpl_per_day = n_ch * 17
        ch_no = ch_no[0:n_smpl_per_day]
    
    # define processed output variables
    ch_time = np.zeros(n_smpl_per_day)  # time stamps of chamber measurements
    p_ch = np.zeros(n_smpl_per_day) * np.nan  # chamber pressure, approximated by ambient pressure
    T_ch = np.zeros(n_smpl_per_day) * np.nan  # chamber air temperature, approximated by air temp at 4.2 m
    f_ch = np.zeros(n_smpl_per_day) * np.nan  # chamber flowrate, mol s-1, need to correct to outlet flowrate?
    f_ch_lpm = np.zeros(n_smpl_per_day) * np.nan   # chamber flowrate, standard liter per min (measured)
    time_lag_nominal = np.zeros(n_smpl_per_day) * np.nan  # save nominal time lag (sec) as a diagnostic
    time_lag_co2 = np.zeros(n_smpl_per_day) * np.nan  # save optimized co2 time lag (sec) as a diagnostic
    status_time_lag_co2 = np.zeros(n_smpl_per_day, dtype='int') - 1  # status code for time lag optimization; initial value -1
    V_ch_mol = np.zeros(n_smpl_per_day) * np.nan    # chamber 'volume' in mol of air
    T_soil_vavg = np.zeros(n_smpl_per_day) * np.nan  # vertical average of soil temp
    T_soil_surf = np.zeros(n_smpl_per_day) * np.nan
    T_soil_A = np.zeros(n_smpl_per_day) * np.nan
    w_soil_vavg = np.zeros(n_smpl_per_day) * np.nan  # vertical average of soil water content
    w_soil_surf = np.zeros(n_smpl_per_day) * np.nan
    w_soil_A = np.zeros(n_smpl_per_day) * np.nan
    PAR = np.zeros(n_smpl_per_day) * np.nan  # PAR at 18 m, from the met data
    PAR_bc = np.zeros(n_smpl_per_day) * np.nan  # below canopy PAR, from the met data
    PAR_SB = np.zeros(n_smpl_per_day) * np.nan  # PAR at the small-B leaf chamber
    PAR_SA = np.zeros(n_smpl_per_day) * np.nan  # PAR at the small-A leaf chamber
    PAR_L = np.zeros(n_smpl_per_day) * np.nan  # PAR at the large leaf chamber
    
    cos_chb = np.zeros(n_smpl_per_day) * np.nan  # chamber conc before closure
    co_chb = np.zeros(n_smpl_per_day) * np.nan
    co2_chb = np.zeros(n_smpl_per_day) * np.nan  
    h2o_chb = np.zeros(n_smpl_per_day) * np.nan
    '''
    cos_chc = np.zeros(n_smpl_per_day) * np.nan  # chamber conc during closure
    co_chc = np.zeros(n_smpl_per_day) * np.nan
    co2_chc = np.zeros(n_smpl_per_day) * np.nan  
    h2o_chc = np.zeros(n_smpl_per_day) * np.nan
    cos_chs = np.zeros(n_smpl_per_day) * np.nan  # chamber conc at the start of the closure period
    co_chs = np.zeros(n_smpl_per_day) * np.nan
    co2_chs = np.zeros(n_smpl_per_day) * np.nan  
    h2o_chs = np.zeros(n_smpl_per_day) * np.nan
    '''
    cos_chc_iqr = np.zeros(n_smpl_per_day) * np.nan  # interquartile range of chamber conc during closure
    co_chc_iqr = np.zeros(n_smpl_per_day) * np.nan
    co2_chc_iqr = np.zeros(n_smpl_per_day) * np.nan
    h2o_chc_iqr = np.zeros(n_smpl_per_day) * np.nan

    cos_cha = np.zeros(n_smpl_per_day) * np.nan  # chamber conc after closure
    co_cha = np.zeros(n_smpl_per_day) * np.nan
    co2_cha = np.zeros(n_smpl_per_day) * np.nan  
    h2o_cha = np.zeros(n_smpl_per_day) * np.nan
    
    fcos = np.zeros(n_smpl_per_day) * np.nan # fluxes of chambers
    fco = np.zeros(n_smpl_per_day) * np.nan
    fco2 = np.zeros(n_smpl_per_day) * np.nan
    fh2o = np.zeros(n_smpl_per_day) * np.nan
    sd_fcos = np.zeros(n_smpl_per_day) * np.nan  # error estimates for fluxes
    sd_fco = np.zeros(n_smpl_per_day) * np.nan
    sd_fco2 = np.zeros(n_smpl_per_day) * np.nan
    sd_fh2o = np.zeros(n_smpl_per_day) * np.nan
    
    # save fitting diagnostics to a separate file
    k_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # slopes
    b_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # intercepts
    r_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # r values
    p_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # p values
    rmse_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # root mean square error of fitted concentrations
    delta_fit = np.zeros((n_smpl_per_day, 4)) * np.nan  # fitted changes of concentration during the closure period
    
    # meta data added 05/26/2016
    ch_labels = list()
    shoot_no = np.zeros(n_smpl_per_day, dtype='int') - 1  # '-1' is used as NaN
    A_ch = np.zeros(n_smpl_per_day) * np.nan
    V_ch = np.zeros(n_smpl_per_day) * np.nan
    
    # for fitting plots
    if flag_save_pdf_plots:
        save_path = plot_dir + '/fitting/'
    else:
        save_path = plot_dir + '/fitting/20' + run_date_str + '/'

    if not os.path.exists(save_path): os.makedirs(save_path)
    # if save plots as pdf, create the pdf now
    if flag_save_pdf_plots:
        pdf_to_save = PdfPages(save_path + '/chfit_20' + run_date_str + '.pdf')

    for loop_num in range(n_smpl_per_day):
        '''
        Indices for flux calculation
        * '_chb' - before closure
        * '_chc' - during closure
        * '_chs' - starting 1 min of the closure  # disabled
        * '_cha' - after closure
        '''
        ch_time[loop_num] = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch ] + (ch_cls[ loop_num % n_ch  ] + ch_o_a[ loop_num % n_ch ]) / 2.
        t_start = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch  ]
        t_o_b = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch  ] + ch_o_b[ loop_num % n_ch  ]
        t_cls = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch  ] + ch_cls[ loop_num % n_ch  ]
        t_o_a = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch  ] + ch_o_a[ loop_num % n_ch  ]
        t_end = doy_int - 1. + loop_num//n_ch * smpl_seq_len + ch_start[ loop_num % n_ch  ] + ch_end[ loop_num % n_ch  ]
        '''
        time_lag_in_day = 0.  # time lag between QCL and chamber control, not used
        dt_lmargin = 0./1440. # set a delta_t safe margin to avoid asynchronicity and pressure-induced fluctuation
        dt_rmargin = 0./1440.
        dt_spike = 0.
        # dt_spike = 20./86400.  # disabled # set a plus 20-sec offset for chamber closing time # no need
        # to avoid the pressure-induced spike, used only in fitting
        '''
        # chamber pressure and temperature, extracted from the met table
        met_loc = np.argmin( np.abs(met_doy - ch_time[loop_num]) )
        p_ch[loop_num] = p_atm[met_loc] * 1e2
        '''
        # deprecated 05/04/2016
        if ch_no[loop_num] <= 3:
            T_ch[loop_num] = T_air_17m[met_loc]
        else:
            T_ch[loop_num] = T_air_4m[met_loc]
        '''
        # soil chamber test on 09/06/2016 from 08:21 to 14:57
        if (ch_time[loop_num] > 249.+8./24.) and (ch_time[loop_num] < 249.+15./24.):
            if (ch_no[loop_num] == 4) or (ch_no[loop_num] == 6):
                ch_no[loop_num] = 5

        # get the current chamber's meta info
        chlut_current = chamber_lookup_table_func(ch_time[loop_num])
        chlut_current = chlut_current[ chlut_current['ch_no'] == ch_no[loop_num] ]
        A_ch[loop_num] = chlut_current['A_ch'].values[0]
        V_ch[loop_num] = chlut_current['V_ch'].values[0]
        ch_labels.append( chlut_current['ch_label'].values[0] ) # 'ch_labels' is a list! not an array
        shoot_no[loop_num] = chlut_current['shoot_no'].values[0]

        # mark light or dark chamber test on 09/06/2016 from 08:21 to 14:57
        if (ch_time[loop_num] > 249.+8./24.) and (ch_time[loop_num] < 249.+15./24.) and (ch_no[loop_num] == 5):
            if (loop_num + loop_num//6) % 2 == 0:
                dark_or_light_string = '-dark'
            else:
                dark_or_light_string = '-light'
            ch_labels[loop_num] = ch_labels[loop_num] + dark_or_light_string

        # use chamber temperatures from thermocouples whenever available
        # added 05/04/2016
        sensor_loc = np.where( (sensor_data['day_of_year'] > t_start) & (sensor_data['day_of_year'] < t_end) )[0]
        if sensor_loc.size > 0:
            chtemp_ref_name = 'T_ch' + str( chlut_current['TC_no'].values[0] ) # referencing column name for chamber temp
            T_ch[loop_num] = np.nanmean(sensor_data[chtemp_ref_name][sensor_loc])
            # however, if chamber temp sensor data are NaNs, T_ch would still be NaN
            # if T_ch is smaller than -273.15 C, which happened since Aug 31, 2016 for leaf chambers, set to NaN
            if T_ch[loop_num] < -273.15: T_ch[loop_num] = np.nan # added on 5 Sep 2016 PDT
        if np.isnan(T_ch[loop_num]):
            # if no valid number is extracted from sensor data
            # use met data as a substitute
            if ch_no[loop_num] <= 3:
                T_ch[loop_num] = T_air_17m[met_loc]
            else:
                T_ch[loop_num] = T_air_4m[met_loc]
        # PAR near leaf chambers
        if sensor_loc.size > 0:
            # get 'PAR_*' referencing names and average PAR values
            if chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-S-A' ")['PAR_no'].size > 0:
                PAR_SA_ref_name = 'PAR_' + str(chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-S-A' ")['PAR_no'].values[0])
                PAR_SA[loop_num] = np.nanmean(sensor_data[PAR_SA_ref_name][sensor_loc])
            if chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-S-B' or ch_label == 'LC-XL+rubber' ")['PAR_no'].size > 0:
                PAR_SB_ref_name = 'PAR_' + str(chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-S-B' or ch_label == 'LC-XL+rubber' ")['PAR_no'].values[0])
                PAR_SB[loop_num] = np.nanmean(sensor_data[PAR_SB_ref_name][sensor_loc])
            if chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-XL' ")['PAR_no'].size > 0:
                PAR_L_ref_name = 'PAR_' + str(chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-XL' ")['PAR_no'].values[0])
                PAR_L[loop_num] = np.nanmean(sensor_data[PAR_L_ref_name][sensor_loc])
            if chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-Slide' ")['PAR_no'].size > 0:
                # note: 'LC-Slide' was always on the same tree as 'LC-XL', even if they co-existed for a while
                # therefore, it is reasonable to override 'LC-XL' with 'LC-Slide', whenever the latter one was present. 
                PAR_L_ref_name = 'PAR_' + str(chamber_lookup_table_func(ch_time[loop_num]).query(" ch_label == 'LC-Slide' ")['PAR_no'].values[0])
                PAR_L[loop_num] = np.nanmean(sensor_data[PAR_L_ref_name][sensor_loc])
        # soil vertically averaged temp and water content
        T_soil_vavg[loop_num] = np.nanmean((met_data[met_loc]['T_soil_surf'], met_data[met_loc]['T_soil_A'], \
            met_data[met_loc]['T_soil_B1'], met_data[met_loc]['T_soil_B2'], met_data[met_loc]['T_soil_C1']))
        w_soil_vavg[loop_num] = np.nanmean((met_data[met_loc]['w_soil_surf'], met_data[met_loc]['w_soil_A'], \
            met_data[met_loc]['w_soil_B1'], met_data[met_loc]['w_soil_B2'], met_data[met_loc]['w_soil_C1']))
        T_soil_surf[loop_num] = met_data[met_loc]['T_soil_surf']
        T_soil_A[loop_num] = met_data[met_loc]['T_soil_A']
        w_soil_surf[loop_num] = met_data[met_loc]['w_soil_surf']
        w_soil_A[loop_num] = met_data[met_loc]['w_soil_A']
        #
        # chamber flow rate (in L/min and in mol/s)
        # now use measured values
        # if no flow data found within the range of chamber measurements, use the nearest point of flow data
        ind_chflow = np.where((flow_doy > t_cls) & (flow_doy < t_o_a))
        ind_chflow_nearest = np.argmin( np.abs(flow_doy - t_cls) )
        flowmeter_no = chlut_current['flowmeter_no'].values[0]
        if ind_chflow[0].size > 0:
            f_ch_lpm[loop_num] = np.nanmean( flow_data[ind_chflow, flowmeter_no+1] )
            if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.): 
                f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, flowmeter_no+1]
            '''
            # remove this part
            if ch_no[loop_num] <=3:
                f_ch_lpm[loop_num] = np.nanmean(flow_data[ind_chflow, ch_no[loop_num]+1])
                if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.): f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, ch_no[loop_num]+1]
            elif ch_no[loop_num] == 4:
                f_ch_lpm[loop_num] = np.nanmean(flow_data[ind_chflow, 6])
                if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.): f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 6]
            elif ch_no[loop_num] == 5:
                f_ch_lpm[loop_num] = np.nanmean(flow_data[ind_chflow, 5])
                if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.): f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 5]
            elif ch_no[loop_num] == 6:
                f_ch_lpm[loop_num] = np.nanmean(flow_data[ind_chflow, 6])
                if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.): f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 6]
            '''
        else: 
            f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, flowmeter_no+1]
            '''
            # remove this part
            if ch_no[loop_num] <= 3: f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, ch_no[loop_num]+1]
            elif ch_no[loop_num] == 4: f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 6]
            elif ch_no[loop_num] == 5: f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 5]
            elif ch_no[loop_num] == 6: f_ch_lpm[loop_num] = flow_data[ind_chflow_nearest, 6]
            '''
        # flow rates in the large soil chamber (soil chamber #3) were recorded separately; correct them
        # added on 5 August 2016
        if ch_no[loop_num] == 6:
            f_ch_lpm[loop_num] = func_flow_lsc(ch_time[loop_num])

        # Honeywell(R) flowmeters report flow rates in 'standard liter per minute' (TSI mass flow meter as well)
        # don't forget to convert 'standard liter per minute' to 'liter per minute'!!
        f_ch_lpm[loop_num] = f_ch_lpm[loop_num] * (T_0 + T_ch[loop_num]) / T_0
        #
        del(flowmeter_no)  # after extracting flow rates, delete this indicator variable
        # ensure a finite and positive number of f_ch_lpm; if not, revert to default values
        # default flow rates (leaf chambers 0.5 lpm, soil chambers 1.7 lpm)
        if np.isnan(f_ch_lpm[loop_num]) or (f_ch_lpm[loop_num] < 0.1):
            if ch_no[loop_num] <=3:
                f_ch_lpm[loop_num] = 0.50
            else:
                f_ch_lpm[loop_num] = 1.70
        # 'f_ch' is the flow rate in mol s^-1
        f_ch[loop_num] = f_ch_lpm[loop_num] * 1e-3 / 60. * air_conc_std * (p_ch[loop_num] / p_std) * T_0/(T_0 + T_ch[loop_num])
        # chamber volume in mol
        V_ch_mol[loop_num] = V_ch[loop_num] * air_conc_std * (p_ch[loop_num] / p_std) * T_0/(T_0 + T_ch[loop_num])
        #
        # set time lags and safety margins separately for leaf chambers and soil chambers
        if ch_no[loop_num] <= 3: # for leaf chambers
            # use time lag calculated from flow rates
            # inner diameter 4.3 mm (1/4 inch O.D. - 0.04 inch wall * 2 = 0.17 inch I.D.)
            # tubing length 60 m for leaf chambers small-A and small-B, and 62.5 m for the large leaf chamber
            tube_inner_diameter = 0.17 * 25.4 * 1e-3
            tube_length = 60. + (ch_no[loop_num] == 3) * 2.5
            time_lag_in_day = tube_inner_diameter**2 * np.pi/4. * tube_length * 1e3 / f_ch_lpm[loop_num] / 1440.
            # if ch_no[loop_num] < 3: time_lag_in_day *= 0.9 # test # tune down the time lag for chamber 1 and 2 to match data
            dt_lmargin = 15./86400.
            dt_rmargin = 15./86400.  # changed from 0 to 15 sec on 05/05/2016
            # time_lag_in_day = 2./1440.  # assume a time lag of 2 min for leaf chamber sampling lines # deprecated
        else:  # for soil chambers
            time_lag_in_day = 0./1440. # no time lag for soil chamber sampling lines
            dt_lmargin = 0.5/1440.
            dt_rmargin = 0.
        # save time lag in seconds as a diagnostic
        time_lag_nominal[loop_num] = time_lag_in_day * 86400.
        # the 'time lag in day' is used in identifying sub-periods of the sampling period

        # --- time lag optimization ---
        if flag_test_mode: print('TEST: loop_num = ' + str(loop_num))  # test
        if flag_timelag_optmz and (ch_no[loop_num] <= 3):
            # do time lag optimization
            time_lag_uplim = 180.  # upper bound of the time lag
            time_lag_lolim = 90. # time_lag_nominal[loop_num] * 0.5  # lower bound of the time lag
            if ch_time[loop_num] > 196.25: time_lag_lolim = 60. # after 6am on DOY 197, leaf chamber flow rates were adjusted to 0.9 lpm
            # '_trial': extract a trial period for the sampling on the current chamber according to the nominal time lag
            ind_ch_full_trial = np.where((qcl_doy > t_o_b) & (qcl_doy < t_end))
            ch_full_time_trial = (qcl_doy[ind_ch_full_trial] - t_start) * 86400.
            co2_full_trial = qcl_data[ind_ch_full_trial]['co2'] * 1e-3
            # initial guess of the time lag will be limited to no larger than 120 sec
            time_lag_guess = np.nanmin((time_lag_nominal[loop_num], 120.))
            time_lag_co2_results = optimize.minimize(func_conc_resid, x0=time_lag_guess, 
                        args=(ch_full_time_trial, co2_full_trial, V_ch_mol[loop_num] / f_ch[loop_num]), 
                        method='Nelder-Mead', options={'xtol': 1e-6, 'ftol': 1e-6})  # unbounded optimization
            time_lag_co2[loop_num] = time_lag_co2_results.x
            status_time_lag_co2[loop_num] = time_lag_co2_results.status
            '''
            # bounded optmz won't work for this non-smooth case
            time_lag_co2_bounded_results = optimize.minimize(func_conc_resid, x0=time_lag_guess, 
                        args=(ch_full_time_trial, co2_full_trial, V_ch_mol[loop_num] / f_ch[loop_num]), 
                        method='SLSQP', bounds=((time_lag_lolim, time_lag_uplim),), options={'disp':False, 'eps':5., 'ftol':1e-9}) # bounded optimization
            # Note: must set the step size 'eps' to >1 for non-trivial results, because the measurements are discrete (one data point every second). 
            if np.abs(time_lag_co2_bounded_results.x - time_lag_guess) < 1.:
                # if the bounded optimization does not work, use the unbounded optimization
                time_lag_co2[loop_num] = time_lag_co2_results.x
                status_time_lag_co2[loop_num] = time_lag_co2_results.status
                print('TEST: bounded optmz does NOT work. ')
            else:
                # if the bounded optimization works
                time_lag_co2[loop_num] = time_lag_co2_bounded_results.x
                status_time_lag_co2[loop_num] = time_lag_co2_bounded_results.status
                # print('TEST: bounded optmz works. ')
            '''
            # check whether the optimized time lag is in the reasonable range
            if (time_lag_co2[loop_num] > time_lag_uplim) or (time_lag_co2[loop_num] < time_lag_lolim) or np.isnan(time_lag_co2[loop_num]):
                # if the optimized time lag is outside of the bounds
                # revert to nominal time lag value or the upper bound of the time lag (whichever is smaller) and set the status code to 255
                time_lag_co2[loop_num] = np.nanmin((time_lag_nominal[loop_num], time_lag_uplim))
                time_lag_co2[loop_num] = np.nanmax((time_lag_co2[loop_num], time_lag_lolim))
                status_time_lag_co2[loop_num] = 255
            del(time_lag_guess, time_lag_co2_results, ind_ch_full_trial, ch_full_time_trial, co2_full_trial)
        else:
            # use nominal time lag
            time_lag_co2[loop_num] = time_lag_nominal[loop_num]

        if flag_test_mode:
            print('TEST: nominal time lag = ' + str(time_lag_nominal[loop_num]) )  # test
            print('TEST: optimized time lag = ' + str(time_lag_co2[loop_num]) )  # test
            print('TEST: optimization status = ' + str(status_time_lag_co2[loop_num]) )  # test
        # update the `time_lag_in_day` variable
        time_lag_in_day = time_lag_co2[loop_num] / 86400.
        # --- end of time lag optimization ---

        # indices for flux calculations
        # ind_ch_full = np.where((qcl_doy > t_o_b + time_lag_in_day) & (qcl_doy < t_end + time_lag_in_day))
        ind_ch_full = np.where((qcl_doy > t_o_b) & (qcl_doy < t_end + time_lag_in_day))  # 'ind_ch_full' index is only used for plotting
        # if on leaf chambers, the plotting time range is extended for 2 more minutes
        if ch_no[loop_num] <= 3:
            ind_ch_full = np.where((qcl_doy > t_o_b) & (qcl_doy < t_end + time_lag_in_day + 2./1440.))  # added on 5 Jul 2016

        ind_chb = np.where((qcl_doy > t_o_b + time_lag_in_day + dt_lmargin) & (qcl_doy < t_cls + time_lag_in_day - dt_rmargin))
        ind_chc = np.where((qcl_doy > t_cls + time_lag_in_day + dt_lmargin) & (qcl_doy < t_o_a + time_lag_in_day - dt_rmargin))
        # ind_chs = np.where((qcl_doy > t_cls + time_lag_in_day + dt_lmargin) & (qcl_doy < t_cls + time_lag_in_day + dt_lmargin + 1./1440.))
        # Note: after the line is switched, regardless of the time lag, the analyzer will sample the next line. 
        # This is the reason why a time lag is not added to the terminal time. 
        ind_cha = np.where((qcl_doy > t_o_a + time_lag_in_day + dt_lmargin) & (qcl_doy < t_end))
        # if on leaf chamber #3, for the opening period after closure, remove the first 30 sec + left safety margin
        if ch_no[loop_num] == 3:
            ind_cha = np.where((qcl_doy > t_o_a + time_lag_in_day + 30./86400. + dt_lmargin) & (qcl_doy < t_end))

        n_ind_chb = ind_chb[0].size
        n_ind_chc = ind_chc[0].size
        # n_ind_chs = ind_chs[0].size
        n_ind_cha = ind_cha[0].size
        
        # calculate fluxes
        if n_ind_chc >= 2. * 60.:    # need at least 2 min data in the closure period to do flux calculation
            flag_calc_flux = 1    # 0 - dont calc fluxes; 1 - calc fluxes
        else:
            flag_calc_flux = 0
        
        # define species for which to calculate flux
        species_list = ['cos', 'co', 'co2', 'h2o']
        conc_factor = [1e3, 1, 1e-3, 1e-6]  # multiplication factors for concentrations to be pptv, ppbv, ppmv, mmol mol^-1
        
        # avg conc's for output
        cos_chb[loop_num] = np.nanmean(qcl_data[ind_chb]['cos'] * conc_factor[0])
        #cos_chc[loop_num] = np.nanmean(qcl_data[ind_chc]['cos'] * conc_factor[0])
        #cos_chs[loop_num] = np.nanmean(qcl_data[ind_chs]['cos'] * conc_factor[0])
        cos_cha[loop_num] = np.nanmean(qcl_data[ind_cha]['cos'] * conc_factor[0])
        cos_chc_iqr[loop_num] = IQR_func(qcl_data[ind_chc]['cos'] * conc_factor[0])
        
        co_chb[loop_num] = np.nanmean(qcl_data[ind_chb]['co'] * conc_factor[1])
        #co_chc[loop_num] = np.nanmean(qcl_data[ind_chc]['co'] * conc_factor[1])
        #co_chs[loop_num] = np.nanmean(qcl_data[ind_chs]['co'] * conc_factor[1])
        co_cha[loop_num] = np.nanmean(qcl_data[ind_cha]['co'] * conc_factor[1])
        co_chc_iqr[loop_num] = IQR_func(qcl_data[ind_chc]['co'] * conc_factor[1])
        
        co2_chb[loop_num] = np.nanmean(qcl_data[ind_chb]['co2'] * conc_factor[2])
        #co2_chc[loop_num] = np.nanmean(qcl_data[ind_chc]['co2'] * conc_factor[2])
        #co2_chs[loop_num] = np.nanmean(qcl_data[ind_chs]['co2'] * conc_factor[2])
        co2_cha[loop_num] = np.nanmean(qcl_data[ind_cha]['co2'] * conc_factor[2])
        co2_chc_iqr[loop_num] = IQR_func(qcl_data[ind_chc]['co2'] * conc_factor[2])
        
        h2o_chb[loop_num] = np.nanmean(qcl_data[ind_chb]['h2o'] * conc_factor[3])
        #h2o_chc[loop_num] = np.nanmean(qcl_data[ind_chc]['h2o'] * conc_factor[3])
        #h2o_chs[loop_num] = np.nanmean(qcl_data[ind_chs]['h2o'] * conc_factor[3])
        h2o_cha[loop_num] = np.nanmean(qcl_data[ind_cha]['h2o'] * conc_factor[3])
        h2o_chc_iqr[loop_num] = IQR_func(qcl_data[ind_chc]['h2o'] * conc_factor[3])

        # fitted conc and baselines, saved for plotting purposes
        # only need two points to draw a line for each species
        conc_bl_pts = np.zeros((len(species_list), 2))
        t_bl_pts = np.zeros(2)
        
        conc_bl = np.zeros((len(species_list), n_ind_chc))  # fitted baselines
        
        conc_fitted = np.zeros((len(species_list), n_ind_chc))  # fitted concentrations during closure
        
        if flag_calc_flux:
            # loop through each species
            for spc_id in range(len(species_list)):
                # spc_id: 0 - COS, 1 - CO, 2 - CO2, 3 - H2O
                # extract closure segments and convert the DOY to seconds after 'ch_start' time for fitting
                ch_full_time = (qcl_doy[ind_ch_full] - t_start) * 86400.  # for plotting only
                chb_time = (qcl_doy[ind_chb] - t_start) * 86400.
                cha_time = (qcl_doy[ind_cha] - t_start) * 86400.
                
                chc_time = (qcl_doy[ind_chc] - t_start) * 86400.
                
                # conc of current species defined with 'spc_id'
                chc_conc = qcl_data[ind_chc][ species_list[spc_id] ] * conc_factor[spc_id] 
                
                # calculate baselines' slopes and intercepts
                # baseline end points changed from mean to medians (05/05/2016)
                median_chb_time = np.nanmedian(qcl_doy[ind_chb] - t_start) * 86400. # median time for LC open (before), sec
                median_cha_time = np.nanmedian(qcl_doy[ind_cha] - t_start) * 86400. # median time for LC open (after)
                median_chb_conc = np.nanmedian(qcl_data[ind_chb][ species_list[spc_id] ]) * conc_factor[spc_id]
                median_cha_conc = np.nanmedian(qcl_data[ind_cha][ species_list[spc_id] ]) * conc_factor[spc_id] 
                # if `median_cha_conc` is not a finite value, set it equal to `median_chb_conc`. Thus `k_bl` will be zero.
                if np.isnan(median_cha_conc): 
                    median_cha_conc = median_chb_conc
                    median_cha_time = chc_time[-1]
                k_bl = (median_cha_conc - median_chb_conc) / (median_cha_time - median_chb_time)
                b_bl = median_chb_conc - k_bl * median_chb_time
                # do not apply baseline correction for water (04/08/2016)
                # two reasons: (1) water conc drift is usually minimal; (2) condensation may happen in the line which creates biased baseline
                if spc_id == 3:
                    k_bl = 0.
                
                # subtract the baseline to correct for instrument drift (assumed linear)
                conc_bl = k_bl * chc_time + b_bl
                
                # linear fit, see the supplementary info of Sun et al. (2015) JGR-Biogeosci
                y_fit = (chc_conc - conc_bl) * f_ch[loop_num] / A_ch[loop_num]
                x_fit = np.exp(-f_ch[loop_num] / V_ch_mol[loop_num] * (chc_time - chc_time[0] + (time_lag_in_day+dt_lmargin)*8.64e4))
                slope, intercept, r_value, p_value, sd_slope = stats.linregress(x_fit, y_fit)
                flux_ch = - slope
                sd_flux_ch = np.abs(sd_slope)
                '''
                # for H2O, if there is a huge spike immediately after the chamber closes, it will take a while to relax; in this case, simply leave out the first 1.5 min
                if spc_id == 3 and  np.nanmax(chc_conc[0:30]) - mean_chb_conc > 5.:
                    slope, intercept, r_value, p_value, sd_slope = stats.linregress(x_fit[90:], y_fit[90:])
                    flux_ch = - slope
                    sd_flux_ch = np.abs(sd_slope)
                '''
                # sanity check 1  # an upper bound (total flux = flux density * area <= 5)
                if np.abs(flux_ch * A_ch[loop_num]) > 5.: flux_ch = np.nan; sd_flux_ch = np.nan
                # sanity check 2: if IQR_conc > 300, instrument drift must have happened and hence the data shall be discarded 
                if (cos_chc_iqr[loop_num] > 300.) or (co2_chc_iqr[loop_num] > 300.):
                    flux_ch = np.nan; sd_flux_ch = np.nan
                #
                # save the fitted conc values 
                conc_fitted[spc_id, :] = (slope * x_fit + intercept) * A_ch[loop_num] / f_ch[loop_num] + conc_bl
                # save the fitting diagnostics
                k_fit[loop_num, spc_id] = slope
                b_fit[loop_num, spc_id] = intercept
                r_fit[loop_num, spc_id] = r_value
                p_fit[loop_num, spc_id] = p_value
                rmse_fit[loop_num, spc_id] = np.sqrt(np.nanmean((conc_fitted[spc_id, :] - chc_conc) ** 2))
                delta_fit[loop_num, spc_id] = (conc_fitted[spc_id, -1] - conc_bl[-1]) - (conc_fitted[spc_id, 0] - conc_bl[0])
                '''
                # print for test
                if loop_num == 0 and spc_id == 0:
                    print slope, intercept, r_value, p_value, sd_slope
                '''
                
                if spc_id == 0:
                    fcos[loop_num] = flux_ch
                    sd_fcos[loop_num] = sd_flux_ch
                elif spc_id == 1:
                    fco[loop_num] = flux_ch
                    sd_fco[loop_num] = sd_flux_ch
                elif spc_id == 2:
                    fco2[loop_num] = flux_ch
                    sd_fco2[loop_num] = sd_flux_ch
                elif spc_id == 3:
                    fh2o[loop_num] = flux_ch
                    sd_fh2o[loop_num] = sd_flux_ch
                
                # save the baseline conc's
                # baseline end points changed from mean to medians (05/05/2016)
                conc_bl_pts[spc_id, :] = median_chb_conc, median_cha_conc
                # do not apply baseline correction for water (04/08/2016)
                if spc_id == 3:
                    conc_bl_pts[spc_id, :] = median_chb_conc, median_chb_conc
            
            # used for plotting the baseline
            t_bl_pts[:] = median_chb_time, median_cha_time
            
        else:
            fcos[loop_num] = np.nan
            fco[loop_num] = np.nan
            fco2[loop_num] = np.nan
            fh2o[loop_num] = np.nan
            sd_fcos[loop_num] = np.nan
            sd_fco[loop_num] = np.nan
            sd_fco2[loop_num] = np.nan
            sd_fh2o[loop_num] = np.nan
            
        # generate fitting plots
        if flag_calc_flux and flag_fitting_plots:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(8,12))
            cos_full = qcl_data[ind_ch_full]['cos'] * conc_factor[0]
            co_full = qcl_data[ind_ch_full]['co'] * conc_factor[1]
            co2_full = qcl_data[ind_ch_full]['co2'] * conc_factor[2]
            h2o_full = qcl_data[ind_ch_full]['h2o'] * conc_factor[3] 
            
            ax1.plot(ch_full_time, cos_full, 'k-')
            ax1.set_ylabel('COS (pptrv)')
            ax2.plot(ch_full_time, co_full, 'k-')
            ax2.set_ylabel('CO (ppbv)')
            ax3.plot(ch_full_time, co2_full, 'k-')
            ax3.set_ylabel('CO$_2$ (ppmv)')
            ax4.plot(ch_full_time, h2o_full, 'k-')
            if ch_no[loop_num] <= 3:
                if doy <= 118.:
                    ax4.set_xlim( (60,540) )  # if leaf chambers
                else:
                    ax4.set_xlim( (60,660) )  # if leaf chambers
            else:
                if doy <= 118.:
                    ax4.set_xlim( (0,660) )  # if soil chambers
                else:
                    ax4.set_xlim( (0,900) )  # if soil chambers
            ax4.xaxis.set_minor_locator( AutoMinorLocator() )
            ax4.set_ylabel('H$_2$O (ppthv)')
            ax4.set_xlabel('Time since switched to the chamber line (sec)')
            
            ch_time_str = (datetime.datetime(2016, 1, 1, 0, 0) + datetime.timedelta(days=t_start)).strftime('%d %b %Y, %H:%M')
            ax1.set_title('ch #' + '%d' % ch_no[loop_num] + ' switch time: ' + ch_time_str + ' UTC+2, no. ' + str(loop_num+1).zfill(3) + '\n' + \
              'F$_{COS}$ = ' + '%.2f' % fcos[loop_num] + '\tF$_{CO}$ = ' + '%.2f' % fco[loop_num] + \
              '  \tF$_{CO2}$ = ' + '%.2f' % fco2[loop_num] + '\tF$_{H2O}$ = ' + '%.2f' % fh2o[loop_num])
            ax2.set_title('nominal time lag = ' + '%.2f' % time_lag_nominal[loop_num] + '; optimized time lag = ' + '%.2f' % time_lag_co2[loop_num])
            del(ch_time_str)
            
            # chamber opening before closure
            ax1.plot(chb_time, qcl_data[ind_chb]['cos'] * conc_factor[0], 'r.')
            ax2.plot(chb_time, qcl_data[ind_chb]['co'] * conc_factor[1], 'r.')
            ax3.plot(chb_time, qcl_data[ind_chb]['co2'] * conc_factor[2], 'r.')
            ax4.plot(chb_time, qcl_data[ind_chb]['h2o'] * conc_factor[3], 'r.')
            
            # chamber closure
            ax1.plot(chc_time, qcl_data[ind_chc]['cos'] * conc_factor[0], 'g.')
            ax2.plot(chc_time, qcl_data[ind_chc]['co'] * conc_factor[1], 'g.')
            ax3.plot(chc_time, qcl_data[ind_chc]['co2'] * conc_factor[2], 'g.')
            ax4.plot(chc_time, qcl_data[ind_chc]['h2o'] * conc_factor[3], 'g.')
            
            # chamber opening after closure
            if cha_time.size > 0:
            	ax1.plot(cha_time, qcl_data[ind_cha]['cos'] * conc_factor[0], 'r.')
            	ax2.plot(cha_time, qcl_data[ind_cha]['co'] * conc_factor[1], 'r.')
            	ax3.plot(cha_time, qcl_data[ind_cha]['co2'] * conc_factor[2], 'r.')
            	ax4.plot(cha_time, qcl_data[ind_cha]['h2o'] * conc_factor[3], 'r.')
            
            # baselines
            # COS
            ax1.plot(t_bl_pts, conc_bl_pts[0,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            ax1.plot(t_bl_pts, conc_bl_pts[0,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            # CO
            ax2.plot(t_bl_pts, conc_bl_pts[1,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            ax2.plot(t_bl_pts, conc_bl_pts[1,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            # CO2
            ax3.plot(t_bl_pts, conc_bl_pts[2,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            ax3.plot(t_bl_pts, conc_bl_pts[2,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            # H2O
            ax4.plot(t_bl_pts, conc_bl_pts[3,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            ax4.plot(t_bl_pts, conc_bl_pts[3,], 'kx--', linewidth=1.5, markeredgewidth=1.25)
            
            # fitted lines
            ax1.plot(chc_time, conc_fitted[0,], '-', color='violet', linewidth=1.5)
            ax2.plot(chc_time, conc_fitted[1,], '-', color='violet', linewidth=1.5)
            ax3.plot(chc_time, conc_fitted[2,], '-', color='violet', linewidth=1.5)
            ax4.plot(chc_time, conc_fitted[3,], '-', color='violet', linewidth=1.5)
            
            # plot timelag point for leaf chambers (added 05/05/2016)
            if ch_no[loop_num] <= 3:
                # time lag lines
                ax1.axvline(time_lag_co2[loop_num], color='b', linestyle='dashed', )
                ax2.axvline(time_lag_co2[loop_num], color='b', linestyle='dashed', )
                ax3.axvline(time_lag_co2[loop_num], color='b', linestyle='dashed', )
                ax4.axvline(time_lag_co2[loop_num], color='b', linestyle='dashed', )
                # chamber closing time lines
                ax1.axvline(time_lag_co2[loop_num] + ch_cls[loop_num % n_ch]*86400., color='r', linestyle='dashed', )
                ax2.axvline(time_lag_co2[loop_num] + ch_cls[loop_num % n_ch]*86400., color='r', linestyle='dashed', )
                ax3.axvline(time_lag_co2[loop_num] + ch_cls[loop_num % n_ch]*86400., color='r', linestyle='dashed', )
                ax4.axvline(time_lag_co2[loop_num] + ch_cls[loop_num % n_ch]*86400., color='r', linestyle='dashed', )
            # set y axes ranges
            q1_cos, q3_cos = np.percentile(cos_full, [25,75])
            iqr_cos = q3_cos - q1_cos
            q1_co, q3_co = np.percentile(co_full, [25,75])
            iqr_co = q3_co - q1_co
            q1_co2, q3_co2 = np.percentile(co2_full, [25,75])
            iqr_co2 = q3_co2 - q1_co2
            q1_h2o, q3_h2o = np.percentile(h2o_full, [25,75])
            iqr_h2o = q3_h2o - q1_h2o
            
            cos_uplim = np.nanmax( (q3_cos + 1.5*iqr_cos, np.percentile(cos_full, 95)) )
            cos_lolim = np.nanmin( (q1_cos - 1.5*iqr_cos, np.percentile(cos_full, 5)) )
            co_uplim = np.nanmax( (q3_co + 1.5*iqr_co, np.percentile(co_full, 95)) )
            co_lolim = np.nanmin( (q1_co - 1.5*iqr_co, np.percentile(co_full, 5)) )
            co2_uplim = np.nanmax( (q3_co2 + 1.5*iqr_co2, np.percentile(co2_full, 95)) )
            co2_lolim = np.nanmin( (q1_co2 - 1.5*iqr_co2, np.percentile(co2_full, 5)) )
            h2o_uplim = np.nanmax( (q3_h2o + 1.5*iqr_h2o, np.percentile(h2o_full, 95)) )
            h2o_lolim = np.nanmin( (q1_h2o - 1.5*iqr_h2o, np.percentile(h2o_full, 5)) )
            
            ax1.set_ylim( (cos_lolim, cos_uplim) )
            ax2.set_ylim( (co_lolim, co_uplim) )
            ax3.set_ylim( (co2_lolim, co2_uplim) )
            ax4.set_ylim( (h2o_lolim, h2o_uplim) )            
            
            if not flag_save_pdf_plots:
                plt.savefig(save_path + 'chfit_20' + run_date_str + '_' + str(loop_num+1).zfill(3) + '.png', bbox_inches='tight')
            else:
                plt.savefig(pdf_to_save, format='pdf', bbox_inches='tight')

            # important!!! release the memory after figure is saved
            fig.clf()
            plt.close()
            # plot 'em, to be continued   
    
    # out of the hourly loop
    # close pdf pages
    if flag_save_pdf_plots:
        pdf_to_save.close()
        del(pdf_to_save)
    # generate daily plots
    hr_local = (ch_time - np.round(ch_time[0])) * 24.
    # color_set=['springgreen', 'limegreen', 'darkgreen', 'tan', 'peru', 'saddlebrown']
    color_set=['limegreen', 'seagreen', 'darkgreen', 'tan', 'peru', 'saddlebrown']
    if doy < 108.:
        # before 18 Apr 2016 (not including that day)
        labels = ['LC-S-B', 'LC-S-A', 'LC-L-A', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 108.):
        # on 18 Apr 2016 
        labels = ['LC-Aux4', 'LC-S-A', 'LC-L-A', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 109.) or (doy == 110.):
        # on 19 and 20 Apr 2016 
        labels = ['LC-XL', 'LC-S-A', 'LC-L-A', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 111.):
        # on 21 Apr 2016
        labels = ['LC-Aux4', 'LC-Aux3', 'LC-Bg', 'SC-1', 'SC-2', 'SC-3']
    elif (doy >= 112.) and (doy < 140.):
        # 22 Apr to 20 May 2016 (not including 20 May 2016)
        labels = ['LC-S-A', 'LC-S-B', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 140.):
        # on 20 May 2016
        labels = ['LC-Aux4', 'LC-Aux3', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy >= 141.) and (doy < 145. ):
        # 21 May to 25 May 2016 (not including 25 May 2016)
        labels = ['LC-S-B', 'LC-S-A', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 145.):
        # on 25 May 2016
        labels = ['LC-Aux4', 'LC-Aux3', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy >= 146.) and (doy < 154.):
        # 26 May to 3 Jun 2016 (not including 3 Jun 2016)
        labels = ['LC-S-A', 'LC-S-B', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 154.):
        labels = ['LC-Aux4', 'LC-S-B', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy > 154.) and (doy < 166.):
        labels = ['LC-Slide', 'LC-S-B', 'LC-XL', 'SC-1', 'SC-2', 'SC-3']
    elif (doy == 166.): 
        labels = ['LC-Aux4', 'LC-S-B', 'LC-Bg', 'SC-1', 'SC-2', 'SC-3']
    elif (doy > 166.): 
        labels = ['LC-S-A', 'LC-S-B', 'LC-Slide', 'SC-1', 'SC-2', 'SC-3']
    else:
        # default
        labels = ['LC-Aux4', 'LC-Aux3', 'LC-Bg', 'SC-1', 'SC-2', 'SC-3']

    fig_dailych, ax_dailych = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(18,12))
    for i in range(1,3):
        ax_dailych[0,0].plot(hr_local[ch_no==i], fcos[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[1,0].plot(hr_local[ch_no==i], fco[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[2,0].plot(hr_local[ch_no==i], fco2[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[3,0].plot(hr_local[ch_no==i], fh2o[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
    # ch3: a separate column
    ax_dailych[0,1].plot(hr_local[ch_no==3], fcos[ch_no==3], 'o-', c=color_set[2], markeredgecolor='None', label=labels[2], lw=1.5)
    ax_dailych[1,1].plot(hr_local[ch_no==3], fco[ch_no==3], 'o-', c=color_set[2], markeredgecolor='None', label=labels[2], lw=1.5)
    ax_dailych[2,1].plot(hr_local[ch_no==3], fco2[ch_no==3], 'o-', c=color_set[2], markeredgecolor='None', label=labels[2], lw=1.5)
    ax_dailych[3,1].plot(hr_local[ch_no==3], fh2o[ch_no==3], 'o-', c=color_set[2], markeredgecolor='None', label=labels[2], lw=1.5)
    for i in range(4, n_ch+1):  # if DOY > 118., plot chamber no. 6 (soil chamber #3)
        ax_dailych[0,2].plot(hr_local[ch_no==i], fcos[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[1,2].plot(hr_local[ch_no==i], fco[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[2,2].plot(hr_local[ch_no==i], fco2[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
        ax_dailych[3,2].plot(hr_local[ch_no==i], fh2o[ch_no==i], 'o-', c=color_set[i-1], markeredgecolor='None', label=labels[i-1], lw=1.5)
    # add the legend
    fig_dailych.legend( (ax_dailych[0,0].lines + ax_dailych[0,1].lines + ax_dailych[0,2].lines), labels[0:n_ch+1], 'upper center', ncol=n_ch, fontsize=12, frameon=False, framealpha=0.5)
    # add the errorbars after the legend, otherwise it'll mess up the labeling
    for i in range(1,3):
        ax_dailych[0,0].errorbar(hr_local[ch_no==i], fcos[ch_no==i], yerr=sd_fcos[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[1,0].errorbar(hr_local[ch_no==i], fco[ch_no==i], yerr=sd_fco[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[2,0].errorbar(hr_local[ch_no==i], fco2[ch_no==i], yerr=sd_fco2[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[3,0].errorbar(hr_local[ch_no==i], fh2o[ch_no==i], yerr=sd_fh2o[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
    # ch3: a separate column
    ax_dailych[0,1].errorbar(hr_local[ch_no==3], fcos[ch_no==3], yerr=sd_fcos[ch_no==3], c=color_set[2], fmt='', linestyle='None', capsize=0)
    ax_dailych[1,1].errorbar(hr_local[ch_no==3], fco[ch_no==3], yerr=sd_fco[ch_no==3], c=color_set[2], fmt='', linestyle='None', capsize=0)
    ax_dailych[2,1].errorbar(hr_local[ch_no==3], fco2[ch_no==3], yerr=sd_fco2[ch_no==3], c=color_set[2], fmt='', linestyle='None', capsize=0)
    ax_dailych[3,1].errorbar(hr_local[ch_no==3], fh2o[ch_no==3], yerr=sd_fh2o[ch_no==3], c=color_set[2], fmt='', linestyle='None', capsize=0)
    for i in range(4, n_ch+1):
        ax_dailych[0,2].errorbar(hr_local[ch_no==i], fcos[ch_no==i], yerr=sd_fcos[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[1,2].errorbar(hr_local[ch_no==i], fco[ch_no==i], yerr=sd_fco[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[2,2].errorbar(hr_local[ch_no==i], fco2[ch_no==i], yerr=sd_fco2[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
        ax_dailych[3,2].errorbar(hr_local[ch_no==i], fh2o[ch_no==i], yerr=sd_fh2o[ch_no==i], c=color_set[i-1], fmt='', linestyle='None', capsize=0)
    #
    ax_dailych[0,0].set_ylabel('F$_{COS}$ (pmol m$^{-2}$ s$^{-1}$)')
    ax_dailych[1,0].set_ylabel('F$_{CO}$ (nmol m$^{-2}$ s$^{-1}$)') 
    ax_dailych[2,0].set_ylabel('F$_{CO2}$ ($\mu$mol m$^{-2}$ s$^{-1}$)')
    ax_dailych[3,0].set_ylabel('F$_{H2O}$ (mmol m$^{-2}$ s$^{-1}$)')
    ax_dailych[3,0].set_xlabel('Hour (Finnish Standard Time, UTC+2)')
    ax_dailych[3,1].set_xlabel('Hour (Finnish Standard Time, UTC+2)')
    ax_dailych[3,2].set_xlabel('Hour (Finnish Standard Time, UTC+2)')
    ax_dailych[0,0].set_title('DOY ' + '%d' % doy_int, loc='left')
    # ax_dailych[0,0].set_title('DOY ' + '%d' % doy_int + '\n leaf chambers - greenish (light to dark): ch1, ch2, ch3;\n soil chambers - lightbrown: ch4; darkbrown: ch5')
    ax_dailych[3,0].set_xlim((0,24))
    ax_dailych[3,0].xaxis.set_major_locator(MultipleLocator(2))
    ax_dailych[3,0].xaxis.set_minor_locator(MultipleLocator(1))
    for i in range(4):
        for j in range(3):
            ax_dailych[i,j].grid(color='gray')
    fig_dailych.tight_layout()
    fig_dailych.savefig(plot_dir + '/daily_plots/daily_chflux_20' + run_date_str + '.png')
    
    # filter out extreme conc measurements
    # this is not intended for data filtering. this is just to make the output work. 
    if (doy == 186.):
        cos_chb[np.abs(cos_chb) > 1e5] = np.nan
        co_chb[np.abs(co_chb) > 1e5] = np.nan
        cos_cha[np.abs(cos_cha) > 1e5] = np.nan
        co_cha[np.abs(co_cha) > 1e5] = np.nan

    # output to file
    output_file_path = output_dir + 'hyy_ch_' + run_date_str + '.txt'
    fo = open(output_file_path, 'w')
    
    header = ['doy_utc', 'doy_local', \
    'ch_no', 'ch_label', 'shoot_no', 'A_ch', 'V_ch', \
    'cos_chb', 'co_chb', 'co2_chb', 'h2o_chb', \
    'cos_cha', 'co_cha', 'co2_cha', 'h2o_cha', \
    'cos_chc_iqr', 'co_chc_iqr', 'co2_chc_iqr', 'h2o_chc_iqr', \
    'fcos', 'fco', 'fco2', 'fh2o', \
    'sd_fcos', 'sd_fco', 'sd_fco2', 'sd_fh2o', \
    'p_ch', 'flow_lpm', 't_lag_nom', 't_lag_co2', 'status_tlag', 'T_ch', \
    'T_soil_surf', 'T_soil_A', 'T_soil_vavg', \
    'w_soil_surf', 'w_soil_A', 'w_soil_vavg', \
    'PAR_SA', 'PAR_SB', 'PAR_L',]
    for i in range(len(header)): header[i] = ' ' * (12 - len(header[i])) + header[i]
    header[2] = ' ' * 3 + 'ch_no'
    header[3] = ' ' * 6 + 'ch_label'
    # fo.write( string.join(header) + '\n' )  # obsolete
    fo.write( ''.join(header) + '\n' )    # print the header
    # note: the default time is no longer UTC, it's Finnish winter time (UTC+2)
    for m in range(n_smpl_per_day):
        line_to_write_float = [ ch_time[m]-2./24., ch_time[m], \
        ch_no[m], ch_labels[m], shoot_no[m], A_ch[m], V_ch[m], \
        cos_chb[m], co_chb[m], co2_chb[m], h2o_chb[m], \
        cos_cha[m], co_cha[m], co2_cha[m], h2o_cha[m], \
        cos_chc_iqr[m], co_chc_iqr[m], co2_chc_iqr[m], h2o_chc_iqr[m], \
        fcos[m], fco[m], fco2[m], fh2o[m], sd_fcos[m], sd_fco[m], sd_fco2[m], sd_fh2o[m], \
        p_ch[m]*1e-2, f_ch_lpm[m], time_lag_nominal[m], time_lag_co2[m], status_time_lag_co2[m], \
        T_ch[m], T_soil_surf[m], T_soil_A[m], T_soil_vavg[m], \
        w_soil_surf[m], w_soil_A[m], w_soil_vavg[m], PAR_SA[m], PAR_SB[m], PAR_L[m],]
        # convert float to str using 'map()'
        line_to_write_str = map(lambda n: '%12.7f'%n, line_to_write_float[0:2]) + \
            [' ' * 7, str(line_to_write_float[2])] + \
            [' ' * (14 - len(line_to_write_float[3])) , line_to_write_float[3]] + \
            [' ' * 10, str(line_to_write_float[4]).zfill(2)] + \
            map(lambda n: '%12.8f'%n, line_to_write_float[5:7]) + \
            map(lambda n: '%12.4f'%n, line_to_write_float[7:])
        # fo.write( string.join(line_to_write_str) + '\n' )    # obsolete
        fo.write(''.join(line_to_write_str) + '\n' )    # print the header
    
    fo.close()
    
    print('Raw data on the day ' + run_date_str + ' processed.')
    # when new data is output, set flag_new_data_generated = True
    flag_new_data_generated = True
    
    # write fitting diagnotics to file
    diag_file_path = output_dir + 'hyy_diag_' + run_date_str + '.txt'
    fd = open(diag_file_path, 'w')
    header_diag = ['doy_utc', 'doy_local', 'ch_no', \
    'k_cos', 'b_cos', 'r_cos', 'p_cos', 'rmse_cos', 'delta_cos', \
    'k_co', 'b_co', 'r_co', 'p_co', 'rmse_co', 'delta_co', \
    'k_co2', 'b_co2', 'r_co2', 'p_co2', 'rmse_co2', 'delta_co2', \
    'k_h2o', 'b_h2o', 'r_h2o', 'p_h2o', 'rmse_h2o', 'delta_h2o', ]
    for i in range(len(header_diag)): header_diag[i] = ' ' * (12 - len(header_diag[i])) + header_diag[i]
    header_diag[2] = ' ' * 3 + 'ch_no'
    fd.write( ''.join(header_diag) + '\n' )    # print the header
    for m in range(n_smpl_per_day):
        line_to_write_float = [ ch_time[m], ch_time[m]+2./24., ch_no[m], \
        k_fit[m,0], b_fit[m,0], r_fit[m,0], p_fit[m,0], rmse_fit[m,0], delta_fit[m,0],\
        k_fit[m,1], b_fit[m,1], r_fit[m,1], p_fit[m,1], rmse_fit[m,1], delta_fit[m,1],\
        k_fit[m,2], b_fit[m,2], r_fit[m,2], p_fit[m,2], rmse_fit[m,2], delta_fit[m,2],\
        k_fit[m,3], b_fit[m,3], r_fit[m,3], p_fit[m,3], rmse_fit[m,3], delta_fit[m,3],]
        # convert float to str using 'map()'
        line_to_write_str = map(lambda n: '%12.7f'%n, line_to_write_float[0:2]) + \
            [' ' * 7, str(line_to_write_float[2])] + map(lambda n: '%12.4f'%n, line_to_write_float[3:])
        fd.write(''.join(line_to_write_str) + '\n' )    # print the header
    fd.close()

print('Done.')

if flag_new_data_generated:
    # combine all the output files into one
    fo_list = glob.glob(output_dir + '/hyy_ch_16*.txt')
    combined_file_path = output_dir + '/combined/hyy_ch_combined.txt'

    with open(combined_file_path, 'w') as fc:
        fc.write(''.join(header) + '\n')
        for file_item in fo_list[0:len(fo_list)]:
            with open(file_item, 'r') as f_in:
                for line_read in f_in:
                    if 'doy_utc' not in line_read:
                        fc.write(line_read)
                f_in.close()
        fc.close()

    print('Combined processed data file written to ' + combined_file_path)

    # combine all fitting diagnostics files into one
    fo_list = glob.glob(output_dir + '/hyy_diag_16*.txt')
    combined_file_path = output_dir + '/combined/hyy_diag_combined.txt'

    with open(combined_file_path, 'w') as fc:
        fc.write(''.join(header_diag) + '\n')
        for file_item in fo_list[0:len(fo_list)]:
            with open(file_item, 'r') as f_in:
                for line_read in f_in:
                    if 'doy_utc' not in line_read:
                        fc.write(line_read)
                f_in.close()
        fc.close()

    print('Combined processed data file written to ' + combined_file_path)

print(datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
