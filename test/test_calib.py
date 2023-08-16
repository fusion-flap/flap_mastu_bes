# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:14:47 2023

@author: Zoletnik
"""
import matplotlib.pyplot as plt
import numpy as np

import flap
import flap_mastu_bes

flap_mastu_bes.register()

plt.close('all')

d=flap.get_data('MAST_BES',exp_id=48235,name='BES*',options={'Resample':1e5},coordinates={'Time':[0.563,0.566]})
plt.figure(figsize=(8,8))
d.plot(plot_type='grid xy',axes=['Column','Row','Time'],options={'Y range':[0,np.max(d.data)]})
# plt.figure(figsize=(5,5))
# d.plot(plot_type='anim-image',axes=['Column','Row','Time'],options={'Waittime':0.05,'Video file':'48236_BES_0.82_0.84.avi','Z range':[0,1.6]})

d_cal=flap.get_data('MAST_BES',exp_id=48235,name='BES*',coordinates={'Time':[0.563,0.566]})
plt.figure(figsize=(5,5))
d_cal.plot(summing={'Row':'Sum','Column':'Sum'},axes='Time')
plt.title('Mean signal during calibration')

plt.figure(figsize=(5,5))
d_cal = d_cal.slice_data(summing={'Time':'Sum'})
d_cal.plot(axes=['Column','Row'],plot_type='image')
plt.title("Intensity during calibration")


d=flap.get_data('MAST_BES',exp_id=48235,name='BES*',options={'Resample':1e6},coordinates={'Time':[0.334,0.3346]})
d.data /= d_cal.data
d.data[:,0,0] = 0
d.data[:,7,0] = 0
d.data[:,0,7] = 0
d.data[:,7,7] = 0
flap.list_data_objects(d) 
plt.figure(figsize=(8,8))
d.plot(plot_type='grid xy',axes=['Column','Row','Time'],options={'Error':False,'Y range':[0,np.max(d.data)]})
plt.suptitle('48235 BES Calibrated')

plt.figure(figsize=(5,5))
d.plot(summing={'Time':'Sum'},axes=['Row','Column'],plot_type='image',options={'Aspect':'equal'})
plt.title('48235 mean calibrated BES (0.334-0.3346)')

plt.figure(figsize=(5,5))
d.plot(plot_type='anim-image',axes=['Row','Column','Time'],options={'Waittime':0.1,'Video file':'test.avi','Z range':[0,np.max(d.data)],'Aspect':'equal'})


