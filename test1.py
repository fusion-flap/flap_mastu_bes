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

plt.figure(figsize=(5,5))

d=flap.get_data('MAST_BES',exp_id=48193,name=['BES-1-1'],options={'Resample':1e4}).plot(options={'All':True})
#plt.clf()
#d=flap.get_data('MAST_BES',exp_id=48193,coordinates={'Time':[0.8,0.82]},name=['BES*'],options={'Resample':1e5})
#d.plot(plot_type='anim-image',axes=['Column','Row','Time'],options={'Waittime':0.1,'Video file':'test.avi','Z range':[0,1.5]})
