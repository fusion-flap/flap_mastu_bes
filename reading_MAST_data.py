import numpy as np
import sys
sys.path.append('/data/MAST/mastda/mast_xbt/MAST data reading/func_utils')
import flap
import func_utils as fu
import matplotlib
import h5py 

@fu.stopwatch
def hdf5_to_flap(filename, options=None, calib_options=None):
    """
    filename : string.
        Name of hdf5 file to be converted to flap object.
            Should contain the file format. Example filename: xbt029937.nc
    options : dictionary.
        Sets desired data-preparation methods.
        'Calibration': if True, the data undergoes calibration based on 
        beam background intensity.
        'Peak trimming': if True, the neutron peaks in the data 
        will be trimmed.
    calib_options : dictionary.
        Sets the method for the case if calibration cannot be done 
        based on the data.
        (value) False: if calibration is not possible then returns 
        the raw data.
        'closest': not implemented yet. Would try to search for an optimal 
        calibration file, which is close to the data's shot time, 
        shot number and viewing radius.
        'specific': user provides the shot number as input for the 
        calibration file to be used.
    Raises:
    NotImplementedError: when user tries to use the 'closest' option for 
    calibration file replacement method.
    Returns:
    A flapDataobject with data from the hdf5 file, as well as the flap time 
    and channel Coordinate objects.
    """
    
    data_folder_path = '/data/MAST/mastda/mast_xbt/data/'
    calib_folder_path = '/data/MAST/mastda/mast_xbt/calibration/'
    calibration_vector = []
    peak_trimmed_no_arr = []
    default_calib_options = {'Choose calibration file':'specific'}
    calib_options = fu.merge_default_and_input_dict(calib_options, default_calib_options)
    
    def read_channel_data(input_hdf5):
        channels_list = []
        for key in input_hdf5['xbt'].keys():
            channel_data = np.array(input_hdf5['xbt'][key][()], dtype=float)
            channel_data = flip(channel_data)
            channel_data -= np.mean(channel_data[1000:5000])
            channels_list.append(channel_data)
        return channels_list
    
    def get_info(input_file):
        info_dict = {'d3_APDcamera': {'bias': input_file['devices']['d3_APDcamera'].attrs['bias'],
                                      'serial': input_file['devices']['d3_APDcamera'].attrs['serial']},
                     'd4_mirror': {'viewRadius': input_file['devices']['d4_mirror'].attrs['viewRadius']}}
        return info_dict
    
    def flip(chn_data):
        old_max = np.amax(chn_data)
        temparr = []
        for i in chn_data:
            temparr.append(old_max - i)
        return np.array(temparr)
            
    def reshape_(d1_array, shape=[8, 4]):
        #shape is 4, 8 starting at bottom right
        result = []
        for i in range(shape[1]):
            templist = []
            for j in range(shape[0]):
                templist.insert(0, d1_array[shape[0]*i + j])
            result.insert(0, templist)
        return np.array(result)
    
    def find_cutoff(chn_data):
        derivative_func = fu.derivative_function(chn_data) 
        deriv_stdev = np.std(derivative_func)
        peak_index_arr = []
        for i in range(len(derivative_func)):
            if derivative_func[i] > deriv_stdev*3:
                peak_index_arr.append(i)
        if len(cutoff_index) > 0:
            cutoff_index = peak_index_arr.pop()
        return cutoff_index
    
    def shot_indices(input_chn):
        base_cut = input_chn[100:10000]
        base_mean = np.mean(base_cut)
        base_std = np.std(base_cut)
        chn_len = int(len(input_chn))
        for i in range(chn_len):
            if input_chn[i] > (base_mean + 10*base_std):
                shot_start_index = i
                break 
        push_val = int(4e5) + shot_start_index 
        shot_end_index = push_val 
        i = push_val
        while shot_end_index == push_val: 
            if input_chn[i] < (base_mean + 10*base_std):
                shot_end_index = i
            i += 1
        return shot_start_index, shot_end_index
    
    def chn_calibration_val(chn_data):
        shot_start_index, cutoff_index = shot_indices(chn_data)
        temparr = []
        iter_index_no = int(2e3)
        beam_slice = chn_data[cutoff_index:(cutoff_index + iter_index_no)]
        beam_stdev = np.std(beam_slice)
        beam_mean = np.mean(beam_slice)
        #this loops cuts down the peaks
        for i in range(iter_index_no):
            if chn_data[cutoff_index + i] > (2*beam_stdev + beam_mean):
                iter_index_no += 1
            else: 
                temparr.append(chn_data[cutoff_index + i])
        temparr = np.array(temparr)
        calib_val = np.mean(temparr)
        return calib_val
    
    def calibration(raw_data):
        nonlocal calibration_vector, filename, MAST_file, calib_folder_path
        if can_calibrate_test(raw_data):
            calib_val_vec = []
            for i in raw_data:
                calib_val_vec.append(chn_calibration_val(i))
            calib_val_vec = np.array(fu.arr_normalize(calib_val_vec))
            result = []
            for channel, calib_val in zip(raw_data, calib_val_vec):
                result.append(channel/calib_val)
            calibration_vector = calib_val_vec
            create_calib_file(calib_folder_path+filename, calibration_vector, MAST_file)
            return np.array(result)
        else:
            nonlocal calib_options 
            if calib_options['Choose calibration file'] == False:
                return raw_data
            if calib_options['Choose calibration file'] == 'closest':
                raise NotImplementedError('"closest" option for channel calibration is not implemented yet')
            if calib_options['Choose calibration file'] == 'specific':
                print('Give shot number for calibration file: ')
                calib_shot_no = str(input())
                calib_filename = 'xbt'+calib_shot_no+'_calib.hdf5'
                calib_file = h5py.File(folder_path + calib_filename, "r")
                calib_val_vec = np.array(calib_file['Calibration values'][()])
                result = []
                for i in range(len(raw_data)):
                    result.append(raw_data[i]/calib_val_vec[i])
                calibration_vector = calib_val_vec
                return result

    def neutron_peak_trim(raw_data):
        result = []
        for i in raw_data:
            result.append(np.array(chn_npt(i)))
        print_trimmed_peak_no()
        return result
    
    def chn_npt(input_chn):
        
        base = input_chn[0:2500]
        base_deriv_func = fu.derivative_function(base)
        base_deriv_mean = np.mean(base_deriv_func)
        base_deriv_std = np.std(base_deriv_func)
        
        def peak_cutoff(shot_data):
            nonlocal peak_trimmed_no_arr 
            if type(shot_data) == list:
                shot_data = np.array(shot_data)
            shot_deriv_func = fu.derivative_function(shot_data)
            deriv_len = len(shot_deriv_func)
            result = [] #shot data without peaks
            peak_trimmed_no = 0
            peak_element_width = 5
            i = 0
                         
            def peaktest(shot_deriv_func, i):
                nonlocal base_deriv_mean, base_deriv_std, deriv_len, peak_element_width
                calib_coeff = 2
                if ((i+4) < deriv_len) and ((shot_deriv_func[i] and shot_deriv_func[i+1]) > (base_deriv_mean + calib_coeff*base_deriv_std) and
                     (shot_deriv_func[i+3] and shot_deriv_func[i+4]) < (base_deriv_mean - calib_coeff*base_deriv_std) and
                     (shot_deriv_func[i+2] < (base_deriv_mean + calib_coeff/2*base_deriv_std) and
                      shot_deriv_func[i+2] > (base_deriv_mean - calib_coeff/2*base_deriv_std))):
                    peak_element_width = 5
                    return True
                
                if ((i+3) < deriv_len) and ((shot_deriv_func[i] and shot_deriv_func[i+1]) > (base_deriv_mean + calib_coeff*base_deriv_std) and
                     (shot_deriv_func[i+2] and shot_deriv_func[i+3]) < (base_deriv_mean - calib_coeff*base_deriv_std)):
                    peak_element_width = 4
                    return True
                
                if (((i+2) < deriv_len) and (shot_deriv_func[i] > (base_deriv_mean + calib_coeff*base_deriv_std) and
                    (shot_deriv_func[i+1] < (base_deriv_mean + calib_coeff/2*base_deriv_std) and 
                      shot_deriv_func[i+1] > (base_deriv_mean - calib_coeff/2*base_deriv_std)) and 
                    shot_deriv_func[i+2] < (base_deriv_mean - calib_coeff*base_deriv_std))):
                    peak_element_width = 3
                    return True
                
                if (((i+1) < deriv_len) and(shot_deriv_func[i] > (base_deriv_mean + calib_coeff*base_deriv_std) and
                    shot_deriv_func[i+1] < (base_deriv_mean - calib_coeff*base_deriv_std))):
                    peak_element_width = 2
                    return True
                
                if True:
                    return False
            
            while i < len(shot_data):
                if peaktest(shot_deriv_func, i):
                    pre_peak_val = shot_data[i-1]
                    post_peak_val = shot_data[i+peak_element_width]
                    replace_vals = np.linspace(pre_peak_val, post_peak_val, (peak_element_width+2))
                    replace_vals = replace_vals.tolist()
                    result += replace_vals[1:-1]
                    i += peak_element_width
                    peak_trimmed_no += 1
                else:
                    result.append(shot_data[i])
                    i += 1
            result = np.array(result)
            peak_trimmed_no_arr.append(peak_trimmed_no)
            return result
        
        shot_start_index, shot_end_index = shot_indices(input_chn)
        trimmed_shot_data = peak_cutoff(input_chn[shot_start_index:shot_end_index])
        input_chn[shot_start_index:shot_end_index] = trimmed_shot_data
        return input_chn

    def data_prep(raw_data, options):
        default_options_dict = {'Calibration': True, 
                                'Peak trimming': True}
        func_dict = {'Calibration': calibration, 
                     'Peak trimming': neutron_peak_trim}
        final_options_dict = fu.merge_default_and_input_dict(options, default_options_dict)
        result = raw_data
        for key in final_options_dict:
            if final_options_dict[key]:
                result = func_dict[key](result)
        return result
        
    def print_trimmed_peak_no():
        nonlocal peak_trimmed_no_arr
        chn_no = 1
        for i in peak_trimmed_no_arr:
            print('Number of trimmed peaks in channel '+str(chn_no)+' is: '+str(i))
            chn_no += 1

    def create_calib_file(filename, calib_vec, input_file):
        filename = filename[0:-3]
        calib_file = h5py.File(filename+'_calib.hdf5','w')
        dataset = calib_file.create_dataset('Calibration values', data=calib_vec)
        dataset.attrs['APDcam_voltage'] = input_file['devices']['d3_APDcamera'].attrs['bias']
        dataset.attrs['viewRadius'] = input_file['devices']['d4_mirror'].attrs['viewRadius']
        dataset.attrs['date'] = input_file.attrs['date']
        dataset.attrs['shot'] = input_file.attrs['shot']
        dataset.attrs['time'] = input_file.attrs['time']
        calib_file.close()
    
    def can_calibrate_test(raw_data):
        tested_channel = raw_data[16]
        can_calibrate = False
        shot_start_index, cutoff_index = shot_indices(tested_channel)
        base_mean = np.mean(tested_channel[500:2000])
        base_std = np.std(tested_channel[500:2000])
        beam_mean = np.mean(tested_channel[cutoff_index:(cutoff_index+1500)])
        if beam_mean > (2*base_std + base_mean):
            can_calibrate = True
            return True
        else: 
            return False
    
    MAST_file = h5py.File(data_folder_path+filename, "r")
    time_vector = MAST_file['time1'][()]
    channel_data = np.array(read_channel_data(MAST_file))
    data_info = get_info(MAST_file)
    
    channel_data = data_prep(channel_data, options)
   
    if data_info['d3_APDcamera']['serial'] == 5:
        data_shape = [8, 4]
        grid_shape = [4, 8]
    else:
        data_shape = [4, 8]
        grid_shape = [8, 4]
    
    channel_data = reshape_(channel_data, data_shape)
    chan_num = np.arange(1, 33).reshape(grid_shape)
    
    time_divs = len(time_vector)
    dt = (time_vector[-1] - time_vector[0])/time_divs
    flap_time_coordinates = flap.Coordinate(name='Time', unit='sec',start=time_vector[0], step=dt, dimension_list=[2],
                                                  mode=flap.CoordinateMode(equidistant=True))
    flap_time_coordinates.shape = (time_divs)
    flap_channel_coords = flap.Coordinate(name='Channel', unit='', values=chan_num, dimension_list=[0, 1], 
                                                  mode=flap.CoordinateMode(equidistant=False))
    flap_channel_coords.shape = chan_num.shape
    channels_data_object = flap.DataObject(channel_data, coordinates=[flap_time_coordinates, flap_channel_coords], info=data_info)
    MAST_file.close()
    return channels_data_object, flap_time_coordinates, flap_channel_coords

test_obj, time_coords, chan_coords = hdf5_to_flap('xbt029937.nc')
