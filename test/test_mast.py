# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:42:58 2023

@author: Zoletnik
"""

import matplotlib.pyplot as plt

import flap
import flap_mastu_bes

flap_mastu_bes.register()

plt.close('all')
# Set up Datapth in the flap_defaults.cfg file and set the working directory where the 
#Reading a simple channel. Offet is subtracted if aet in config file. 
#E.g. Offset timerange = [-0.1,-0.01]
d = flap.get_data('MAST_BES',exp_id=48193,name='BES-1-2')
# Printing the contents of the data object:
flap.list_data_objects(d)
#The simplest plot:
d.plot()
# Plotting all the points without reduction only 10 ms from 0.2s 
d.plot(slicing={'Time':flap.Intervals(0.2,0.21)},options={'All points':True})

plt.figure()
plt.subplot(2,2,1)
# Plotting one coordinate as a function of another:
d.plot(axes=['Sample','Time'])
plt.subplot(2,2,2)
# PLotting one signal as a function of another
d1 = flap.get_data('MAST_BES',exp_id=48193,name='BES-1-3')
d.plot(axes=d1)
# Resampling during data read:
d = flap.get_data('MAST_BES',exp_id=48193,name='BES-1-3',options={'Resample':1e3})  
plt.subplot(2,2,3)
d.plot()  
plt.subplot(2,2,4)
d.plot(axes=['__Data__','Time'])

plt.figure()
flap.get_data('MAST_BES',exp_id=48193,coordinates={'Time':[0.1,0.2]},name='BES-1-3').plot()
    
plt.figure()
plt.subplot(2,2,1)
d=flap.get_data('MAST_BES',exp_id=48193,name=['BES-1-*'],options={'Resample':1e3})
flap.list_data_objects(d)
pid = d.plot()
plt.subplot(2,2,2)
d.plot(plot_type='image',axes=['Time','Column'])

# Video of the full BES measurement.
plt.figure()
d=flap.get_data('MAST_BES',exp_id=48193,name=['BES*'],options={'Resample':1e2})
flap.list_data_objects()
d.plot(plot_type='anim-image',axes=['Column','Row','Time'])
   

