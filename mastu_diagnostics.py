import os
import io
import re

import numpy as np
import h5py

import flap

import ast

def is_array_equidistant(a, rtol, atol):
    d = np.diff(a)
    return np.allclose(d, d[0], rtol=rtol, atol=atol)

def get_data_mastu(
        exp_id,
        data_name,
        no_data=False,
        coordinates=None,
        data_source=None,
        options=None,
    ):
    """
    Data read function for general MAST-U diagnostics data.

    Parameters
    ----------
    exp_id : int
        Shot number. 
    data_name : str
        Access path of diagnosic trace. This will become the cache filename,
        with illegal characters being replaced by '-'.
    no_data : bool, optional, default=False
        Whether to skip the data itself.
    coordinates : List of flap.Coordinate() or a single flap.Coordinate, optional, default=None
        Defines read ranges. Not implemented.
    data_source : str, optional, default=None
        Data source name.
    options : dict, default=None
        - 'Detect equidistant coordinates' (bool, default=True):
            - Whether to try to detect equidistant coordinates automatically.
              Uses parameters 'rtol' and 'atol' for numerical precision.
        - 'Detect zero error' (bool, default=True):
            - Whether to try to detect zero error arrays to save space.  Uses
              parameters 'atol' for numerical precision.
        - 'atol' (float, default=1e-12): Absolute tolerance of comparison. See
          `np.isclose()` for details.
        - 'rtol' (float, default=1e-9): Relative tolerance of comparison. See
        `np.isclose()` for details.
        - 'Coordinate remapping' (dict, default={'names': {}, 'units': {}}):
            - A dictionary for mapping units and unit names of pyUDA coordinates to
            FLAP coorindates. Should be a valid Python dictionary, containing
            two sub-dictinaries with keys 'names' and 'units'. The keys of the
            sub-dictionaries are mappped to the associated values.
            E.g. {'names': {'time [TIME]' : 'Time'}, 'units': {'s': 'Second'}}.
        TODO

    Return value
    ------------
    flap.DataObject:
        The output flap data object.
    """

    if coordinates is not None:
        raise NotImplementedError('Passing read ranges is not implemented for MAST-U diagnostics data.')
    
    if not isinstance(exp_id, int):
        raise ValueError('Only integer exp_id, i.e. shot number is supported.')

    default_options = {
        'Download data': False,
        'Cache directory': None,
        'Resample' : None,
        'Detect equidistant coordinates' : True,
        'Detect zero error' : True,
        'atol' : 1e-12,
        'rtol' : 1e-9,
        'Coordinate remapping' : {'names': {}, 'units': {}},
    }

    _options = flap.config.merge_options(
        default_options,
        options,
        data_source='MAST-U'
    )

    atol = _options['atol']
    rtol = _options['rtol']
    detect_zero_error = _options['Detect zero error']
    detect_equidistant = _options['Detect equidistant coordinates']

    coord_remap = ast.literal_eval(_options['Coordinate remapping'])

    if _options['Download data'] in [True, False]:
        download_file = _options['Download data']
    else:
        raise ValueError(f"Option 'Download data' has invalid value: {_options['Download data']}")
    
    if _options['Cache directory'] is None:
        cache_used = False
        if not download_file:
            raise ValueError('Download and caching cannot both be disabled for MAST-U diagnotic data.')
        else:
            # Download but do not cache
            datapath = None
    else:
        # Use cache, not datapath
        cache_used = True
        datapath = _options['Cache directory']
        if not isinstance(datapath, str):
            raise ValueError(f"Invalid cache directory '{datapath}'.")

    shotstring = str(exp_id).zfill(6)

    # https://stackoverflow.com/a/71199182
    file_name = re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "-", data_name) + '.hdf5'

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

            import pyuda
            pyuda.Client.server = pyuda_options['Server']
            pyuda.Client.port = pyuda_options['Server port']
            client = pyuda.Client()

            print("Downloading...")

            # Only pyuda.Signal objects are supported as of now
            result = client.get(data_name, exp_id)
            if not isinstance(result, pyuda.Signal):
                raise ValueError('Only Signal type objects are implemented to be imported as FLAP objects.')

            if not cache_used:
                datafile = io.BytesIO()

            # elkészíteni a h5py objektumot
            with h5py.File(datafile, 'w') as f:
                dset = f.create_dataset("data", data=result.data)
                dset.attrs['pyuda_type'] = 'Signal'
                dset.attrs['description'] = result.description
                dset.attrs['label'] = result.label
                dset.attrs['rank'] = result.rank
                dset.attrs['time_index'] = result.time_index
                dset.attrs['unit'] = result.units

                dset_errors = f.create_dataset("errors", data=result.errors)
                dims_grp = f.create_group('dims', track_order=True)
                for dim in result.dims:
                    dim_dset = dims_grp.create_dataset(dim.label, data=dim.data)
                    dim_dset.attrs['unit'] = dim.units

            print("Download complete.")
            client.close_connection()
            print("Connection closed.")

        except Exception as e:
            raise RuntimeError("pyUDA download failed.") from e
    
    # Read the data into flap

    with h5py.File(datafile) as file:
        if file['data'].attrs['pyuda_type'] != 'Signal':
            raise ValueError("Only pyUDA 'Signal' data type is supported.")

        # The data values themselves
        data_array = file['data'][()]

        # Potentially detect zero error in order to avoid storing it
        if detect_zero_error:
            if np.allclose(file['errors'], 0, atol=atol):
                error = None
            else:
                error = file['errors']
        else:
            error = file['errors']
        
        # Unit of the data
        data_unit = flap.Unit(name='Signal', unit=file['data'].attrs['unit'])

        # Convert the stored dimensions to flap.Coordinates
        # Potentially detect if they are equidistant
        coord = []
        for i, dim_label in enumerate(file['dims'].keys()):
            dim_data = file['dims'][dim_label][()]

            if detect_equidistant:
                if is_array_equidistant(dim_data, atol=atol, rtol=rtol):
                    c_mode = flap.CoordinateMode(equidistant=True)
                else:
                    c_mode = flap.CoordinateMode(equidistant=False)
            else:
                c_mode = flap.CoordinateMode(equidistant=False)
            
            if c_mode.equidistant:
                start = dim_data[0] 
                step = np.mean(np.diff(dim_data))
                values = None
                value_index = None
            else:
                start = None
                step = None
                values = dim_data
                value_index = None

            coord_name = dim_label
            if coord_name in coord_remap['names'].keys():
                coord_name = coord_remap['names'][coord_name]

            coord_unit = file['dims'][dim_label].attrs['unit']
            if coord_unit in coord_remap['units'].keys():
                coord_unit = coord_remap['units'][coord_unit]

            coord.append(flap.Coordinate(
                name=coord_name,
                unit=coord_unit,
                mode=c_mode,
                shape=len(dim_data),
                start=start,
                step=step,
                values=values,
                value_index=value_index,
                dimension_list=[i],
            ))
        
        # Build data title
        data_title = f'MAST diagnostics data, {data_name}'

        additional_label_and_description = [
            str(ld) for ld in [
                file['data'].attrs['label'],
                file['data'].attrs['description'],
                ]
            if ld not in ['', None]
        ]

        if len(additional_label_and_description) > 0:
            data_title += f' ({', '.join(additional_label_and_description)})'

        # Build the data object
        d = flap.DataObject(
            data_array=data_array,
            error=error,
            data_unit=data_unit,
            coordinates=coord,
            exp_id=exp_id,
            data_title=data_title,
            data_source=data_source,
        )

    return d

def register(data_source=None):
    flap.register_data_source('MAST-U', get_data_func=get_data_mastu, add_coord_func=None)