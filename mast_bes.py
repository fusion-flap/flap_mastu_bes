# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:18:09 2023

MAST and MAST Upgrade BES diagnostic FLAP module

@author: Sandor Zoletnik zoletnik.sandor@ek-cer.hu
"""
import os
import sys
import numpy as np
import copy
import h5py
import psutil

import flap
from flap_apdcam.apdcam_control import apdcam_channel_map,apdcam10g_channel_map

def get_camera_info(MAST_file):
    camera_type = None
    try:
        keys = MAST_file['devices']['d3_APDcamera'].attrs.keys()
        camera_type = 'APDCAM-1G'
        cam_object = 'd3_APDcamera'
    except KeyError:
        pass
    if (not camera_type):
        try:
            keys = MAST_file['devices']['d3_APDcamera10G'].attrs.keys()
            camera_type = 'APDCAM-10G'
            cam_object = 'd3_APDcamera10G'
        except KeyError:
            raise ValueError('Cannot determine APDCAM type.')
    data = {}
    for k in keys:
        data[k] = MAST_file['devices'][cam_object].attrs[k]
    data['APDCAM_bits'] = data['depth'][0] # This is not in the file, but normally set to this
    return camera_type,data

def apdcam_channel_list(camera_type,sensor_rotation):
    if (camera_type == 'APDCAM-1G'):
        if (sensor_rotation == False):
            chmap = apdcam_channel_map(sensor_rotation=0)
        else:
            chmap = apdcam_channel_map(sensor_rotation=90)
    elif (camera_type == 'APDCAM-10G'):
        chmap = apdcam10g_channel_map(camera_type='8x8',camera_version=1)
    else:
        raise ValueError("Internal error, unknown camera type.")
    return chmap
            

def get_data_mast_bes(exp_id=None, data_name=None, no_data=False, options=None, coordinates=None, data_source=None):
    """ 
    Data read function for MAST and MAST-U BES diagnostic.
    
    Parameters
    ----------
    exp_id: Shot number or string. If string it has to be in yyyymmdd.xxx format and this reads early 
            measurements in 2022-23.
    data_name: string or list of strings
               ADCxxx: ADC number. Unix style regular expressions are allowed:
                       ADC*
                       ADC[2-5]
                       Can also be a list of data names, eg. ['ADC1','ADC3']
               BES-r-c (string): APD pixel at row r, column c realtive to upper left corner as looking onto 
                       the detector.
    coordinates: List of flap.Coordinate() or a single flap.Coordinate
                 Defines read ranges. The following coordinates are interpreted:
                     'Sample': The read samples
                     'Time': The read times
                     Only a single equidistant range is interpreted. Use option "Resample"
                     to resample the signal to lower frequency than the original sampling frequency.
    options: dict
        'Scaling':  'Digit'
                    'Volt'
        'Datapath': Data path (string)
        'Resample': Resample to this frequency [Hz]. Only frequencies below the sampling frequency can be used.
                    The frequency will be rounded to the integer times the sampling frequency. 
                    Data will be averaged in blocks and the variance in blocks will be added as error.
        'Test measurement': 
   
    Return value
    ------------
    flap.DataObject:
        The output flap data object. The data dimension depends on the requested channels.
        If only 1 channel is requested: 1D
        If any of the channels is ADCxxx or the requested channels do not form a 2D array: 2D
        If the requested channels form a regular 2D subarray of it: 3D
            
    """ 
    
    if (exp_id is None):
        raise ValueError('exp_id should be set for MAST data.')
    default_options = {'Datapath': 'data',
                       'Scaling':'Digit',
                       'Offset timerange': [-0.1,-0.01],
                       'Resample' : None,
                       'Test measurement': False
                       }
    _options = flap.config.merge_options(default_options,options,data_source='MAST_BES')
    
    # Checking if data path/file exists
    datapath = _options['Datapath']
    if (type(exp_id) is int):
        if (_options['Test measurement']):
            datafile = os.path.join(datapath,'xbtz{:06d}.nc'.format(exp_id))
        else:    
            datafile = os.path.join(datapath,'xbt{:06d}.nc'.format(exp_id))
        temp_data = False
        if (not os.path.exists(datafile)):
            raise ValueError("Cannot find datafile: {:s}".format(datafile))
    elif (type(exp_id) is str):
        datadir = os.path.join(datapath,exp_id)
        temp_data = True
        if (not os.path.exists(datadir)):
            raise ValueError("Cannot find test data directory: {:s}".format(datafile))       
    else:
        raise ValueError("exp_id should be integer (normal shot) a string of format yyyymmdd.xxx (test shot).")

    if (not temp_data):
        MAST_file = h5py.File(datafile, "r")
        camera_type,camera_info = get_camera_info(MAST_file)
        if (camera_type == 'APDCAM-1G'):
            if (camera_info['genCameraSerial'][3] == 4):
                sensor_rotation = 0
            elif (camera_info['genCameraSerial'][3] == 5):
                sensor_rotation = 90
            else:
                raise ValueError('Unknown APDCAM-1G serial number: {:d}.'.format(camera_info['genCameraSerial'][3]))
        else:
            sensor_rotation = 0
        time_vector = MAST_file['time1'][()]
        camera_info['APDCAM_samplenumber'] = len(time_vector)
        camera_info['APDCAM_starttime'] = time_vector[0]
        camera_info['APDCAM_sampletime'] = (time_vector[-1] - time_vector[0]) / (len(time_vector - 1))
    else:
        raise NotImplementedError("Reading temporary MAST-U BES data is not implemented yet.")
    
    # Ensuring that the data name is a list
    if type(data_name) is not list:
        chspec = [data_name]
    else:
        chspec = data_name
    # Finding the desired channels
    chmap = apdcam_channel_list(camera_type,sensor_rotation)
    ch_names = []
    adc_list = []
    row_list = []
    col_list = []
    nrow = chmap.shape[0]
    ncol = chmap.shape[1]
    for ir in range(nrow):
        for ic in range(ncol):
            ch_names.append('BES-{:d}-{:d}'.format(ir + 1,ic + 1))
            row_list.append(ir + 1)
            col_list.append(ic + 1)
            ch_names.append('ADC{:d}'.format(chmap[ir,ic]))
            row_list.append(None)
            col_list.append(None)
            adc_list.append(chmap[ir,ic])
            adc_list.append(chmap[ir,ic])
    try:
        chname_proc, ch_index = flap.select_signals(ch_names,chspec)
    except ValueError as e:
        raise e     
    if (len(col_list) != 0):
        col_proc = [col_list[i] for i in ch_index]
    if (len(row_list) != 0):
        row_proc = [row_list[i] for i in ch_index]
    ADC_proc = [adc_list[i] for i in ch_index]

    # Determining the dimension of the output data array
    if (len(chname_proc) == 1):
        outdim = 1
    else:
        outdim = 3
        # If any of the channels has None in column or row, the output is 2D
        for c,r in zip(col_proc,row_proc):
            if ((c is None) or (r is None)):
                outdim = 2
                break
        if (outdim == 3):
            out_row_list = sorted(set(row_proc))
            out_col_list = sorted(set(col_proc))
            if ((len(out_col_list) * len(out_row_list) != len(chname_proc))
                or (len(out_col_list) == 1) or  (len(out_row_list) == 1)  
                ):
                outdim = 2
            else:
                out_col_index = []
                out_row_index = []
                for i in range(len(row_proc)):
                    out_row_index.append(out_row_list.index(row_proc[i]))
                    out_col_index.append(out_col_list.index(col_proc[i]))

    # Determining read sample range. Result will be read_samplerange and resample_binsize
    read_range = None
    read_samplerange = None
    if (coordinates is not None):
        if (type(coordinates) is not list):
             _coordinates = [coordinates]
        else:
            _coordinates = coordinates
        for coord in _coordinates:
            if (type(coord) is not flap.Coordinate):
                raise TypeError("Coordinate description should be flap.Coordinate.")
            if ((coord is None) or (coord.c_range is None)):
                continue
            if (coord.unit.name == 'Time'):
                if (coord.mode.equidistant):
                    read_range = [float(coord.c_range[0]),float(coord.c_range[1])]
                    if (read_range[1] <= read_range[0]):
                        raise ValueError("Invalid read timerange.")
                else:
                    raise NotImplementedError("Non-equidistant Time axis is not implemented yet.")
                break
            if coord.unit.name == 'Sample':
                if (coord.mode.equidistant):
                    read_samplerange = coord.c_range
                    if (read_samplerange[1] <= read_samplerange[0]):
                        raise ValueError("Invalid read samplerange.")

                else:
                    raise \
                        NotImplementedError("Non-equidistant Sample axis is not implemented yet.")
                break
    if ((read_range is None) and (read_samplerange is None)):
        read_samplerange = np.array([0,camera_info['APDCAM_samplenumber'] - 1])
    if (read_range is not None):
        read_range = np.array(read_range)
    if (read_samplerange is None):
        read_samplerange = np.rint((read_range - float(camera_info['APDCAM_starttime']))
                                   / float(camera_info['APDCAM_sampletime'])
                                   ).astype(int)
    else:
        read_samplerange = np.array(read_samplerange).astype(int)
    if ((read_samplerange[1] < 0) or (read_samplerange[0] >= camera_info['APDCAM_samplenumber'])):
        raise ValueError("No data in time range.")
    if (read_samplerange[0] < 0):
        read_samplerange[0] = 0
    if (read_samplerange[1] >= camera_info['APDCAM_samplenumber']):
        read_samplerange[1] = camera_info['APDCAM_samplenumber'] - 1
    read_range = float(camera_info['APDCAM_starttime']) \
                       + read_samplerange * float(camera_info['APDCAM_sampletime'])
    if (_options['Resample'] is not None):
        if (_options['Resample'] > 1 / camera_info['APDCAM_sampletime']):
            raise ValueError("Resampling frequency should be below the original sample frequency.")
        resample_binsize = int(round((1 / _options['Resample']) / float(camera_info['APDCAM_sampletime'])))

    # Determining data array type
    if (_options['Scaling'] == 'Volt'):
        scale_to_volts = True
        dtype = float
        data_unit = flap.Unit(name='Signal',unit='Volt')
        number_size = 8
    elif (_options['Scaling'] == 'Digit'):
        # Converting also Digits to float as offset subtraction results in floats
        scale_to_volts = False
        dtype = float
        data_unit = flap.Unit(name='Signal',unit='Digit')
        number_size = 8
    else:
        raise ValueError("Invalid option 'Scaling'. Should be 'Digit' or 'Volt'.")

    # Getting offset data
    offset_timerange = _options['Offset timerange']
    if (offset_timerange is not None):
        if (type(offset_timerange) is not list):
            raise ValueError("Invalid option 'Offset timerange'. Should be 2-element list.")
        if (len(offset_timerange) != 2):
            raise ValueError("Invalid option 'Offset timerange'. Should be 2-element list.")

        offset_samplerange = np.rint((np.array(offset_timerange) - float(camera_info['APDCAM_starttime']))
                                   / float(camera_info['APDCAM_sampletime'])).astype(int)
        if ((offset_samplerange[0] < 0) or (offset_samplerange[1] >= camera_info['APDCAM_samplenumber'])):
            raise ValueError("Offset timerange is out of measurement time.")
        offset_data = np.empty(len(ADC_proc), dtype='float')
        for i_ch,ch in enumerate(ADC_proc):
            d = np.array(MAST_file['xbt']['channel{:02d}'.format(ch)], dtype=float)[offset_samplerange[0] : offset_samplerange[1]]
            offset_data[i_ch] = np.mean(d)
        if (scale_to_volts):
            offset_data = ((2. ** camera_info['APDCAM_bits'] - 1) - offset_data) / (2. ** camera_info['APDCAM_bits'] - 1) * 2
        else:
            offset_data = (2. ** camera_info['APDCAM_bits'] - 1) - offset_data

    ndata_read = int(read_samplerange[1] - read_samplerange[0] + 1)
    if (_options['Resample'] is not None):
        ndata_out = int(ndata_read / resample_binsize) 
        ndata_read = ndata_out * resample_binsize
    else:
        ndata_out = ndata_read

    # Determining data array shape
    if (outdim == 1):
        data_shape = (ndata_out) 
    elif (outdim == 2):
        data_shape = (ndata_out,len(ADC_proc))
    else:
        data_shape = (ndata_out,len(out_row_list),len(out_col_list))
        
    if (no_data is False):
        if ndata_out * len(ADC_proc) * number_size > psutil.virtual_memory().available:
            raise MemoryError("Not enough memory for reading data")

        data_arr = np.empty(data_shape,dtype=dtype) 
        if (_options['Resample'] is not None):
            error_arr = np.empty(data_shape,dtype=dtype) 
        else:
            error_arr = None

        for i,ch in enumerate(ADC_proc):
            d = np.array(MAST_file['xbt']['channel{:02d}'.format(ch)], dtype=float)
            d = np.flip(d)
            d = d[read_samplerange[0]: read_samplerange[0] + ndata_read]
            if (scale_to_volts):
                d = ((2. ** camera_info['APDCAM_bits'] - 1) - d) / (2. ** camera_info['APDCAM_bits'] - 1) * 2
            else:
                d = (2. ** camera_info['APDCAM_bits'] - 1) - d
            if (offset_timerange is not None):
                d -= offset_data[i]
            if (_options['Resample'] is not None):
                d_resample = np.zeros(ndata_out,dtype=float)
                d_error = np.zeros(ndata_out,dtype=float)
                if (ndata_out > resample_binsize):
                    for i_slice in range(0,resample_binsize):
                        d_resample += d[slice(i_slice,len(d),resample_binsize)]
                        d_error += d[slice(i_slice,len(d),resample_binsize)] ** 2
                else:
                    for i_resamp in range(0,len(d_resample)):
                        d_resample[i_resamp] = np.sum(d[i_resamp * resample_binsize : (i_resamp + 1) * resample_binsize])
                        d_error[i_resamp] = np.sum(d[i_resamp * resample_binsize : (i_resamp + 1) * resample_binsize] ** 2)
                d_error = np.sqrt(d_error / resample_binsize - (d_resample / resample_binsize) ** 2)
                d = d_resample / resample_binsize
                if (outdim == 1):
                    error_arr = d_error
                elif (outdim == 2):
                    error_arr[:,i] = d_error
                else:
                    error_arr[:,out_row_index[i],out_col_index[i]] = d_error
            if (outdim == 1):
                data_arr = d
            elif (outdim == 2):
                data_arr[:,i] = d
            else:
                data_arr[:,out_row_index[i],out_col_index[i]] = d
    else:
        data_arr = None
    
    coord = []

    if (read_range is None):
        read_range = float(camera_info['APDCAM_starttime']) + read_samplerange * float(camera_info['APDCAM_sampletime'])
    if (_options['Resample'] is not None):
        tstart = read_range[0] + float(camera_info['APDCAM_sampletime']) * resample_binsize / 2
        tstep = float(camera_info['APDCAM_sampletime']) * resample_binsize
    else:
        tstart = read_range[0]
        tstep = float(camera_info['APDCAM_sampletime'])
    c_mode = flap.CoordinateMode(equidistant=True)
    coord.append(flap.Coordinate(name='Time',
                                 unit='Second',
                                 mode=c_mode,
                                 shape=ndata_out,
                                 start=tstart,
                                 step=tstep,
                                 dimension_list=[0]
                                 )
                 )
   
    c_mode = flap.CoordinateMode(equidistant=True)
    if (_options['Resample'] is not None):
        s_start = read_samplerange[0] + resample_binsize / 2
        s_step = resample_binsize
    else:
        s_start = read_samplerange[0]
        s_step = 1
    coord.append(flap.Coordinate(name='Sample',
                                 unit='n.a.',
                                 mode=c_mode,
                                 shape=ndata_out,
                                 start=s_start,
                                 step=s_step,
                                 dimension_list=[0]
                                 )
                 )

    if (outdim == 1):
        c_mode = flap.CoordinateMode(equidistant=False)
        coord.append(flap.Coordinate(name='ADC Channel',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[1],
                                     values=ADC_proc[0],
                                     dimension_list=[]
                                     )
                     )
        coord.append(flap.Coordinate(name='Signal name',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[1],
                                     values=chname_proc,
                                     dimension_list=[]
                                     )
                     )
    elif (outdim == 2):
        c_mode = flap.CoordinateMode(equidistant=False)
        coord.append(flap.Coordinate(name='ADC Channel',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[len(ADC_proc)],
                                     values=np.array(ADC_proc),
                                     dimension_list=[1]
                                     )
                     )
        coord.append(flap.Coordinate(name='Signal name',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[len(ADC_proc)],
                                     values=np.array(chname_proc),
                                     dimension_list=[1]
                                     )
                     )
        if ('out_row_list' in locals()):
            if (len(out_row_list) == 1):
                dimlist = []
            else:
                dimlist = [1]
            coord.append(flap.Coordinate(name='Row',
                                         unit='n.a.',
                                         mode=c_mode,
                                         shape=[len(out_row_list)],
                                         values=np.array(out_row_list),
                                         dimension_list=dimlist
                                         )
                         )
        if ('out_col_list' in locals()):
            if (len(out_col_list) == 1):
                dimlist = []
            else:
                dimlist = [1]
            coord.append(flap.Coordinate(name='Column',
                                         unit='n.a.',
                                         mode=c_mode,
                                         shape=[len(out_col_list)],
                                         values=np.array(out_col_list),
                                         dimension_list=dimlist
                                         )
                         )
    else:
        c_mode = flap.CoordinateMode(equidistant=False)
        c_array = np.ndarray((len(out_row_list),len(out_col_list)),dtype=int)
        for i in range(len(ADC_proc)):
            c_array[out_row_index[i],out_col_index[i]] = ADC_proc[i]
        coord.append(flap.Coordinate(name='ADC Channel',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=c_array.shape,
                                     values=c_array,
                                     dimension_list=[1,2]
                                     )
                     )
        maxlen = 0
        for s in chname_proc:
            if (len(s) > maxlen):
                maxlen = len(s)
        c_array = np.ndarray((len(out_row_list),len(out_col_list)),dtype='<U{:d}'.format(maxlen))
        for i in range(len(chname_proc)):
            c_array[out_row_index[i],out_col_index[i]] = chname_proc[i]
        coord.append(flap.Coordinate(name='Signal name',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=c_array.shape,
                                     values=c_array,
                                     dimension_list=[1,2]
                                     )
                     )
        coord.append(flap.Coordinate(name='Row',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[len(out_row_list)],
                                     values=np.array(out_row_list),
                                     dimension_list=[1]
                                     )
                     )
        coord.append(flap.Coordinate(name='Column',
                                     unit='n.a.',
                                     mode=c_mode,
                                     shape=[len(out_col_list)],
                                     values=np.array(out_col_list),
                                     dimension_list=[2]
                                     )
                     )
        
    data_title = "MAST BES data"
    if (data_arr.ndim == 1):
        data_title += ", " + chname_proc[0]
    d = flap.DataObject(data_array=data_arr,error=error_arr,data_unit=data_unit,
                        coordinates=coord, exp_id=exp_id,data_title=data_title,info=camera_info)
    return d
           


    
def register(data_source=None):
    flap.register_data_source('MAST_BES', get_data_func=get_data_mast_bes, add_coord_func=None)

