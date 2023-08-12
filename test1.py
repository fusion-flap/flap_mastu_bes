# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 21:14:47 2023

@author: Zoletnik
"""
import matplotlib.pyplot as plt

import flap
import flap_mastu_bes

flap_mastu_bes.register()

plt.close('all')

plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
d=flap.get_data('MAST_BES',exp_id=48193,name=['BES*'],options={'Resample':1e3})
pid = d.plot(plot_type='grid xy',axes=['Column','Row','Time'])
plt.subplot(2,2,2)
pid1 = d.plot(slicing={'Signal name':'BES-3-4'},axes='Time')
plt.subplot(2,2,3)
d.plot(slicing={'Row':2},plot_type='multi xy',axes=['Time','Column'])
plt.subplot(2,2,4)
d.plot(slicing={'Time':0.5},plot_type='image',axes=['Column','Row'])
d=flap.get_data('MAST_BES',exp_id=48189,name=['BES*'],options={'Resample':1e3})
d.plot(plot_type='grid xy',axes=['Column','Row','Time'],plot_id=pid)
d.plot(slicing={'Signal name':'BES-1-4'},axes='Time',plot_id=pid1)