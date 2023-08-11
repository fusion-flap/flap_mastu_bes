# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:42:58 2023

@author: Zoletnik
"""

import flap
import flap_mastu_bes

flap_mastu_bes.register()

#d = flap.get_data('MAST_BES',exp_id=48193,name='BES-1-2',coordinates={'Time':[-0.1,1]},options={'Offset':None})
d = flap.get_data('MAST_BES',exp_id=48193,name='BES-1-2')
flap.list_data_objects(d)
d.plot()