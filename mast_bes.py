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
import itertools
import warnings

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
        chmap = apdcam10g_channel_map(camera_type='8x8',camera_version=0)
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
               BES-r-c (string): APD pixel at row r, column c. Numbering
                                 convention is determined by the 'Use deployment
                                 channel numbering' option: see below.
    coordinates: List of flap.Coordinate() or a single flap.Coordinate
                 Defines read ranges. The following coordinates are interpreted:
                     'Sample': The read samples
                     'Time': The read times
                     Only a single equidistant range is interpreted. Use option "Resample"
                     to resample the signal to lower frequency than the original sampling frequency.
    options: dict
        'Scaling':  'Digit'
                    'Volt'
        'Cache directory' (str):
            - If provided, the given path will be used as the cache directory.
            - If not provided, and 'Download' data is True, the data will be
              downloaded into memory but not cached.
        'Calibrate intensity' (bool):
            - False: Do not apply any calibration
            - True: Apply calibration based on the given parameters.
        'Calibration exact coordinates' (list of str, default=['APDCam_viewRadius']):
            - Calibration coordinates that are to be matched exactly.
        'Calibration nearby coordinates' (list of str, default=['APDCam_biasSet0', 'APDCam_biasSet1', 'filter_temperature_read', 'SS_voltage_99']):
            - Calibration coordinates which will be used for selecting the
              closest match between those reference measurements that fit the
              exactly matced coordinates.
        'Calibration file' (str):
            - If given, this file is used as the reference measurement in the
              calibration procedure, without any matching. Cannot be used
              simultaneously with 'Calibration directory'.
        'Calibration directory' (str):
            - If given, the calibration reference files in this directory are used to
              construct a database, and the best match is selected based on the
              nearby and exactly matched coordinates given in the options.
              Cannot be used simultaneously with 'Calibration file'.
        'Calibration beam voltage reference' (float, default=1):
            - Only values higher than the reference will be considered when
              determining the 99th percentile voltage value that is considered as
              the nominal beam voltage. Dimension: Volts.
        'Download data' (bool):
            - False: Do not download anything, only use data from 'Datapath'.
            - True: Download data from source specified in elements 'Server' and 'Server
              port' of the '[pyUDA]' section in the options.
        'Datapath': Data path (string), only used if caching and downloading are
                    not enabled.
        'Server' (default=None): str, Server to download from.
        'Server port' (default=None): str, Server port to use.
        'Offset timerange': (default=[-0.1,-0.01]):
            - list of two values: [start, end]. The time range used for offset calculation.
            - None: No offset calculation.
        'Resample': Resample to this frequency [Hz]. Only frequencies below the sampling frequency can be used.
                    The frequency will be rounded to the integer times the sampling frequency.
                    Data will be averaged in blocks and the variance in blocks will be added as error.
        'Remove sharp peaks': (default=False):
            - True: Remove sharp peaks from the data automatically during import
              using the `remove_sharp_peaks` function in FLAP. Options for
              `remove_sharp_peaks` must be set in the [Denoising] section.
            - False: Do nothing.
        'Test measurement': bool, default=False
        'Use deployment numbering' : bool, default=True
            - False: Use channel numbering as the flap_apdcam library returns it
              (relative to the upper left corner of the detector viewed from the
              front)
            - True: Use channel numbering in the deployement configuration, if available

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
                       'Download data': False,
                       'Cache directory': None,
                       'Calibrate intensity': False,
                       'Calibration exact coordinates': ['APDCam_viewRadius'],
                       'Calibration nearby coordinates': ['APDCam_biasSet0', 'APDCam_biasSet1', 'filter_temperature_read', 'SS_voltage_99'],
                       'Calibration file': None,
                       'Calibration directory' : None,
                       'Calibration beam voltage reference': 1,
                       'Scaling':'Digit',
                       'Offset timerange': [-0.1,-0.01],
                       'Resample' : None,
                       'Remove sharp peaks': False,
                       'Test measurement': False,
                       'Use deployment channel numbering': True,
                       }
    _options = flap.config.merge_options(default_options,options,data_source='MAST_BES')

    if _options['Download data'] in [True, False]:
        download_file = _options['Download data']
    else:
        raise ValueError(f"Option 'Download data' has invalid value: {_options['Download data']}")
    
    if _options['Cache directory'] is None:
        cache_used = False
        if not download_file:
            # Use old behaviour, no cache, no download
            datapath = _options['Datapath']
        else:
            # Download but do not cache
            datapath = None
    else:
        # Use cache, not datapath
        cache_used = True
        datapath = _options['Cache directory']
        if not isinstance(datapath, str):
            raise ValueError(f"Invalid cache directory '{datapath}'.")
    
    if (_options['Calibration directory'] is not None) and (_options['Calibration file'] is not None):
        raise ValueError("Both of the options 'Calibration directory' and 'Calibration file' are set: these are mutually exclusive options, only one can be set.")

    def file_name_from_shot_number(shot_number, is_test_measurement=False):
        shotstring = str(shot_number).zfill(6)
        if not is_test_measurement:
            return f'xbt{shotstring}.nc'
        else:
            return f'xbtz{shotstring}.nc'

    def build_MAST_BES_download_path(shot_number, is_test_measurement=False):
        file_name = file_name_from_shot_number(shot_number, is_test_measurement)
        return f'$MAST_DATA/{shot_number}/LATEST/{file_name}'

    if (type(exp_id) is int):
        file_name = file_name_from_shot_number(
            exp_id,
            is_test_measurement=_options['Test measurement']
        )

        if cache_used:
            shotstring = str(exp_id).zfill(6)
            datafile = os.path.join(datapath, shotstring, file_name)

            # Decide whether cache exists and is valid
            print("Caching is enabled. Looking for cached file.")

            shot_folder = os.path.join(datapath, shotstring)
            if not (os.path.exists(shot_folder)):
                try:
                    os.mkdir(shot_folder)
                except Exception as e:
                    raise SystemError("The shot folder cannot be created. Cache directory might not be present.") from e

            if os.path.isfile(datafile):
                try:
                    test_open_MAST_file = h5py.File(datafile, "r")
                    test_open_MAST_file.close()
                    print(f"Using cached '{datafile}'")
                    download_file = False
                except:
                    print(f"Existing file '{datafile}' could not be opened, might be corrupt. Deleting and redownloading.")
                    os.remove(datafile)
            else:
                print(f"Could not find cached file '{datafile}'.")

        else:
            if datapath is not None:
                datafile = os.path.join(datapath, file_name)
                if (not os.path.exists(datafile)):
                    raise ValueError("Cannot find datafile: {:s}".format(datafile))
                else:
                    print(f"Using datafile '{datafile}'.")


        if download_file:
            default_pyuda_options = {
                'Server': None,
                'Server port': None,
            }

            pyuda_options = flap.config.merge_options(
                default_pyuda_options,
                {},
                data_source=data_source,
                section='pyUDA')
            
            print("Downloading via pyUDA...")
            try:
                if pyuda_options['Server port'] is None:
                    raise ValueError("Server port is None.")
                if pyuda_options['Server'] is None:
                    raise ValueError("Server is None.")

                print(f"Downloading from {pyuda_options['Server']}:{pyuda_options['Server port']}")

                print("Opening connection...")
                # This replicates the pyuda wrappers from _client.py in order to
                # enable saving a file to memory

                import cpyuda
                cpyuda.set_server_host_name(pyuda_options['Server'])
                cpyuda.set_server_port(pyuda_options['Server port'])

                print("Downloading...")
                source_file = build_MAST_BES_download_path(
                    exp_id,
                    is_test_measurement=_options['Test measurement']
                )

                result = cpyuda.get_data("bytes::read(path=%s)" % source_file, "")

                if cache_used:
                    with open(datafile, 'wb') as f_out:
                        result.data().tofile(f_out)
                    print(f"Saved to '{datafile}'.")
                else:
                    import io
                    datafile = io.BytesIO(result.data())
                    print("Storing downloaded file in memory, not writing to disk.")
                
                # Try to free up some memory
                del result

                print("Download complete.")
                cpyuda.close_connection()
                print("Connection closed.")

            except Exception as e:
                raise RuntimeError("pyUDA download failed.") from e

        temp_data = False

    elif (type(exp_id) is str):
        datadir = os.path.join(datapath, exp_id)
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
    if _options['Use deployment channel numbering']:
        if camera_type == 'APDCAM-10G' and np.all(camera_info['genCameraSerial'] == np.asarray([0,0,0,4], dtype='int8')):
            # Per the installation on MAST-U, as of the MU04 campaign
            chmap = chmap.T
        else:
            raise ValueError(f"For camera type {camera_type} and serial {camera_info['genCameraSerial']}, no deployment channel numbering is available. Set the option 'Use deployment channel numbering' to False, or implement a deployment channel numbering in 'mast_bes.py' for this configuration.")
    ch_names = []
    adc_list = []
    adc_rc_list = []
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
            adc_rc_list.append((ir,ic))
            adc_rc_list.append((ir,ic))
    try:
        chname_proc, ch_index = flap.select_signals(ch_names,chspec)
    except ValueError as e:
        raise e
    if (len(col_list) != 0):
        col_proc = [col_list[i] for i in ch_index]
    if (len(row_list) != 0):
        row_proc = [row_list[i] for i in ch_index]
    ADC_proc = [adc_list[i] for i in ch_index]
    ADC_rc_proc = [adc_rc_list[i] for i in ch_index]

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

    if _options['Remove sharp peaks']:
        print('Performing sharp peak removal...')
        d = d.remove_sharp_peaks()

    if _options['Calibrate intensity']:
        from .intensity_calibration import CalibrationReference, CalibrationDatabase
        calibration_filename = None
        calibration_cref = None
        if _options['Calibration directory'] is not None:
            calibration_dir = _options['Calibration directory']
            print(f'Using intensity calibration directory {calibration_dir} to find proper match')

            possible_calibration_filenames = [
                os.path.join(calibration_dir, node)
                for node
                in os.listdir(calibration_dir)
                if (os.path.isfile(os.path.join(calibration_dir, node)) and node.endswith('.json'))
            ]

            calibration_refs = []
            valid_calibration_filenames = []
            for fname in possible_calibration_filenames:
                try:
                    with open(fname, 'r') as f:
                        cref = CalibrationReference.from_json(f.read())
                        if cref.calibration_multiplier_matrix is None:
                            warnings.warn(f"Unable to import calibration reference file {fname}. (No calibration multiplier matrix found.)")
                        else:
                            calibration_refs.append(cref)
                            valid_calibration_filenames.append(fname)
                except Exception as e:
                    warnings.warn(f"Unable to import calibration reference file {fname}. ({e})")
                    
            actual_ref = CalibrationReference.from_MAST_U_shot(
                exp_id,
                beam_voltage_threshold_SI=_options['Calibration beam voltage reference'],
            )

            cdb = CalibrationDatabase(
                calibration_refs,
                exactly_matched_coordinates=_options['Calibration exact coordinates'],
                closely_matched_coordinates=_options['Calibration nearby coordinates'],
                exactly_matched_round_to_decimals=2,
                non_matched_references=[actual_ref],
            )

            match_dist, nearest_match_i, nearest_match = cdb.find_nearest_match(actual_ref)

            calibration_filename = valid_calibration_filenames[nearest_match_i]
            calibration_cref = nearest_match
            
            print(f'Found match in database ({match_dist:0.02f}): {calibration_filename}')

        elif _options['Calibration file'] is not None:
            calibration_filename = _options['Calibration file']
            with open(calibration_filename) as f:
                calibration_cref = CalibrationReference.from_json(f.read())
        else:
            raise ValueError(f"Option 'Calibrate intensity' is True, but neither option 'Calibration file' nor option 'Calibration directory' are set.")

        if (calibration_filename is not None) and (calibration_cref is not None):
            cref_chmap = np.array(calibration_cref.calibration_source['channel_map'])
            if np.all(chmap == cref_chmap):
                # calibration multiplier matrix array
                cmma = np.array(cref.calibration_multiplier_matrix)
                if outdim == 3:
                    if d.data.shape == (8,8):
                        d.data *= cmma
                    else:
                        out_row_list_i = np.array(out_row_list) - 1
                        out_col_list_i = np.array(out_col_list) - 1
                        d.data *= cmma[np.ix_(out_row_list_i, out_col_list_i)]
                elif outdim == 2:
                    d.data *= cmma[*np.array(ADC_rc_proc).T]
                    pass
                elif outdim == 1:
                    d.data *= cmma[*ADC_rc_proc[0]]
                else:
                    raise RuntimeError('This should never happen: outdim != 1, 2, or 3.')

                d.data_unit.unit = 'n.a.'
                print(f'Applied intensity calibration from {calibration_filename}')
            else:
                raise ValueError('Channel map in current data and reference calibration file do not match.')

    return d

def register(data_source=None):
    flap.register_data_source('MAST_BES', get_data_func=get_data_mast_bes, add_coord_func=None)

