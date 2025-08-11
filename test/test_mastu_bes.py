import copy

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import flap  # This loads the flap_defaults.cfg in the current working directory
import flap_mastu_bes

## To use a data source via flap.get, first, it has to be registered.

# Register MAST-U BES data source
flap_mastu_bes.register()

# Register MAST-U diagnostics data source
flap_mastu_bes.mastu_diagnostics.register()

######################  1  ######################

# Load a BES shot, all channels, in a single flap DataObject
test_shotnumber = 50615
d = flap.get_data(
    data_source='MAST_BES',
    exp_id=test_shotnumber,
    name='BES-*-*',
    object_name='BES'  # also load it to flap storage
)

# Get a single channel and plot it
plt.figure()
single_channel_name = 'BES-5-5'
d_single_channel = d.slice_data({'Signal name' : single_channel_name})
d_single_channel.plot()  # Without arguments, it plots against the 'Time' coordinate
plt.show()
## Alternatively, if we only need a single channel, this could have also been:
# d_single_channel = flap.get_data(data_source='MAST_BES', exp_id=test_shotnumber, name='BES-5-5',

# Get four consecutive channels in poloidal direction and plot them above each
# other
# Note: in this graph, the Y values are offset to provide legibility! Do not take these literally.
plt.figure()
d.slice_data({'Signal name' : 'BES-[1-4]-5'}).plot(plot_type='multi xy')
plt.show()

# Get four consecutive channels in radial direction and plot them
# But now, in separate plots, with shared x axes
plt.figure(figsize=(4,8))
gs = GridSpec(4, 1)
ax0 = plt.subplot(gs[0])
d.slice_data({'Signal name' : 'BES-5-1'}).plot()
plt.title('BES-5-1')

plt.subplot(gs[1], sharex=ax0)
d.slice_data({'Signal name' : 'BES-5-2'}).plot()
plt.title('BES-5-2')

plt.subplot(gs[2], sharex=ax0)
d.slice_data({'Signal name' : 'BES-5-3'}).plot()
plt.title('BES-5-3')

plt.subplot(gs[3], sharex=ax0)
d.slice_data({'Signal name' : 'BES-5-4'}).plot()
plt.title('BES-5-4')

plt.tight_layout()
plt.show()

######################  2  ######################

# Autocorrelation
print('Calculating ACF...')
autocorr = d_single_channel.ccf(coordinate='Time', options={'Normalize':True})
plt.figure()
autocorr.plot(axes=['Time lag'], options={'All points': True})
plt.title(f'Autocorrelation: {single_channel_name}')
plt.show()

# Cross-correlation
other_channel_name = 'BES-2-3'
d_other_channel = d.slice_data({'Signal name' :other_channel_name})
print('Calculating CCF...')
crosscorr = d_single_channel.ccf(ref=d_other_channel, coordinate='Time', options={'Normalize':True})
plt.figure()
crosscorr.plot(axes=['Time lag'], options={'All points': True})
plt.title(f'Cross-correlation: {single_channel_name} vs. {other_channel_name}')
plt.show()

######################  3  ######################

# So far, we have used the object-oriented approach to FLAP data objects.
# An alternative is to use the FLAP data store, this is illustrated here
# briefly.
#
# In this paradigm, the data objects are referred to by strings. No objects are
# passed around, rather, they are retreived from the flap data store.
#
# You can use whatever paradigm fits your workflow the best.

# List the data store contents:
flap.list_data_objects()

# Select a channel and store it in the flap store
flap.slice_data('BES', slicing={'Row' : 5, 'Column': 5}, output_name='BES_sliced-5-5')
# And another one
flap.slice_data('BES', slicing={'Row' : 2, 'Column': 3}, output_name='BES_sliced-2-3')

# Auto-power spectral density
flap.apsd('BES_sliced-5-5', coordinate='Time', output_name='APSD')
plt.figure()
flap.plot('APSD', axes='Frequency', options={'Log x' : True, 'Log y' : True, 'All points' : True})
plt.title('APSD: BES-5-5')
plt.show()

flap.cpsd('BES_sliced-5-5', ref='BES_sliced-2-3', coordinate='Time', output_name='CPSD')
plt.figure()
flap.plot('CPSD', axes='Frequency', options={'Log x' : True, 'Log y' : True, 'All points' : True})
plt.suptitle('CPSD: BES-5-5 vs. BES-2-3')
plt.show()

# Now we return to the object-oriented paradigm.

######################  4  ######################


## Getting diagnostics data from MAST-U diagnostics

# Note: proper coordinate mapping must be set in config to match flap conventions!

# lower super-X divertor D-alpha
d_XIM_DA_HL02_SXD = flap.get_data('MAST-U', test_shotnumber, name='XIM/DA/HL02/SXD')

# D_alpha - HU10-OSP (outer strike point)
d_XIM_DA_HU10_OSP = flap.get_data('MAST-U', test_shotnumber, name='XIM/DA/HU10/OSP')

# high-frequency magnetics
d_ACQ216_202_CH02 = flap.get_data('MAST-U', test_shotnumber, name='XMC/ACQ216_202/CH02')

# low-frequency magnetics
d_XMB_SANX13_01_CH81 = flap.get_data('MAST-U', test_shotnumber, name='XMB/SANX13-01/CH81')

# Plasma current
d_IP = flap.get_data('MAST-U', test_shotnumber, name='ip')

# SS Beam power
d_SS_POWER = flap.get_data('MAST-U', test_shotnumber, name='ANB-SS-POWER')

# Line-integrated density
d_xdc_ai_raw_density = flap.get_data('MAST-U', test_shotnumber, name='xdc/ai/raw/density')


# Plot most of them
plt.figure(figsize=(4,10))
gs = GridSpec(6, 1)

ax0 = plt.subplot(gs[0])
d_XIM_DA_HU10_OSP.plot()
plt.title('D_alpha - HU10-OSP')

plt.subplot(gs[1], sharex=ax0)
d_XMB_SANX13_01_CH81.plot()
plt.title('Low-frequency magnetics')

plt.subplot(gs[2], sharex=ax0)
d_ACQ216_202_CH02.plot()
plt.title('High-frequency magnetics')

plt.subplot(gs[3], sharex=ax0)
d_IP.plot()
plt.title('Plasma current')

plt.subplot(gs[4], sharex=ax0)
d_SS_POWER.plot()
plt.title('SS beam power')

plt.subplot(gs[5], sharex=ax0)
d_xdc_ai_raw_density.plot()
plt.title('Line-integrated density')

plt.tight_layout()
plt.xlim([0,1])
plt.show()

######################  5  ######################

# Cross-correlation of magnetics with BES channel
# Problem: the two signals have different temporal resolution
print("BES data step:", d_single_channel.get_coordinate_object('Time').step)
print("Magnetics step:", d_ACQ216_202_CH02.get_coordinate_object('Time').step)
# This will be solved via interpolation: we resample the magnetics data (it has a higher sample rate).

d_magnetics_resampled = d_ACQ216_202_CH02.slice_data({'Time' : d_single_channel}, options={'Interpolation' : 'Closest', 'Partial intervals' : False})
flap.list_data_objects(d_single_channel)
flap.list_data_objects(d_magnetics_resampled)

# They also have a different range.
# Let's restrict this to [0 s, 1 s] here.
d_single_channel_resampled = d_single_channel.slice_data({'Time' : flap.Intervals(0, 1)})
d_magnetic_interval = d_magnetics_resampled.slice_data({'Time' : flap.Intervals(0, 1)})

plt.figure()
mag_bes_ccf = d_magnetic_interval.ccf(ref=d_single_channel_resampled, coordinate='Time', options={'Normalize': True})
mag_bes_ccf.plot(axes=['Time lag'], options={'All points': True})
plt.title(f'Cross-correlation: magnetics vs. BES channel: {single_channel_name}')
plt.show()

######################  6  ######################

# Calculations with LF megnetics
# Problem: sampling is not equidistant in this signal, many FLAP functions will not work

# Demonstration:
try:
    d_XMB_SANX13_01_CH81.apsd()
    print('APSD success!')
except Exception as e:
    print(repr(e))

# We see that the Time coordinates are indeed not equidistant:
flap.list_data_objects(d_XMB_SANX13_01_CH81)

# This can be verified this way as well:
print(f'{d_XMB_SANX13_01_CH81.coordinates[0].unit.name=}')
print(f'{d_XMB_SANX13_01_CH81.coordinates[0].mode.equidistant=}')

# Why?
time = d_XMB_SANX13_01_CH81.coordinate('Time')[0]
steps = np.diff(time)
print(f'{np.all(np.isclose(steps,steps[0]))=}')
print(f'{steps.min()=:.10e}')
print(f'{steps.max()=:.10e}')

# Let's see the trend:
plt.figure(figsize=(15,5), dpi=100)
plt.scatter(time[:-1], steps)
plt.yscale('log')
plt.grid('both')
plt.xlabel('Time [s]')
plt.ylabel('Sampling interval [s]')
plt.show()


# During the main part of the shot, the sampling  is equidistant.
# We can convert the time coordinates in this range to equidistant.
from flap_mastu_bes.mastu_diagnostics import convert_to_equidistant

# Fix a time range where we think the sampling is equidistant
t_equi_range = [0, 1]
Mirnov_LF_data_sliced = d_XMB_SANX13_01_CH81.slice_data(
    slicing={'Time' : flap.Intervals(t_equi_range[0], t_equi_range[1])}
)

# Convert the Time coordinates to equidistant:
Mirnov_LF_data_sliced_equi = convert_to_equidistant(Mirnov_LF_data_sliced, 'Time')

# They are indeed equidistant now:
print(f"{Mirnov_LF_data_sliced_equi.get_coordinate_object('Time').mode.equidistant=}")

# Now, this can be used for APSD and other calculations.
try:
    Mirnov_LF_data_sliced_equi.apsd()
    print('APSD success!')
except Exception as e:
    print(repr(e))