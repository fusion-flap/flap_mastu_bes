from operator import itemgetter
import os

import h5py
import numpy as np
from scipy.spatial import KDTree

import flap

def file_name_from_shot_number(shot_number):
    shotstring = str(shot_number).zfill(6)
    return f'xbt{shotstring}.nc'

class CalibrationReference:
    """Object representing an intensity calibration refererence.
        
    The representation is aimed at providing a possibility to compare the
    circumstances of a measurement to other measurements, which can be used as
    calibration reference measurements.  Automatic construction from a MAST-U
    shot is recommended via the `.from_MAST_U_shot()` class method. Saving and
    loading to and from a JSON file is also supported via the appropriate
    methods.

    Parameters
    ----------
    coordinates : dict
        Calibration coordinates. The keys are the coordinate names, the
        values are the dimensionful coordinate values.
    info : dict, optional (default=None)
        Dictionary containing additonal non-coordinate information and metadata.
    calibration_multiplier_matrix : list of list of float, optional (default=None)
        Calibration multiplier matrix. A BES measurement can be multiplied
        by this matrix to perform the intensity calibration along the
        channels.
    calibration_source : dict, optional (default=None)
        Dictionary containing information about the source of information
        used to create the calibration multiplier matrix.
    """
    def __init__(self,
                 coordinates,
                 info=None,
                 calibration_multiplier_matrix=None,
                 calibration_source=None,
                ):
        if not isinstance(coordinates, dict):
            raise ValueError("Argument 'coordinates' must be a dictionary.")
            
        if ((info is not None) and (not isinstance(info, dict))):
            raise ValueError("Argument 'info' must be None or a dictionary.")

        self.coordinates_dimful = coordinates
        self.info = info

        # These are only used if this will be used as a reference measurement
        self.calibration_multiplier_matrix = calibration_multiplier_matrix
        self.calibration_source = calibration_source

    def __repr__(self):
        return f"CalibrationCoordinates(coordinates={repr(self.coordinates_dimful)}, info={repr(self.info)}, calibration_multiplier_matrix={repr(self.calibration_multiplier_matrix)}, calibration_source={repr(self.calibration_source)}"

    def to_json(self):
        import json
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, json_string):
        """Construct a CalibrationReference object from a JSON string.

        Parameters
        ----------
        json_string : str
            JSON representation of the object. (Usually have been created via
            the `.to_json()` method.)

        Returns
        -------
        CalibratioReference
            The resulting CalibrationReference object.
        """
        import json
        attrs = json.loads(json_string)
        if not isinstance(attrs, dict):
            raise ValueError('Invalid JSON: must parse to a dict.')

        if 'info' not in attrs.keys():
            raise ValueError("Invalid JSON: must contain an 'info' key.")
        
        if 'coordinates_dimful' not in attrs.keys():
            raise ValueError("Invalid JSON: must contain a 'coordinates_dimful' key.")

        if 'calibration_multiplier_matrix' in attrs.keys():
            calibration_multiplier_matrix = attrs['calibration_multiplier_matrix']
        else:
            calibration_multiplier_matrix = None
            
        if 'calibration_source' in attrs.keys():
            calibration_source = attrs['calibration_source']
        else:
            calibration_source = None
        
        return cls(
            coordinates=attrs['coordinates_dimful'],
            info=attrs['info'],
            calibration_multiplier_matrix=calibration_multiplier_matrix,
            calibration_source=calibration_source,
        )

    @classmethod
    def from_MAST_U_shot(cls,
                         shotnumber,
                         beam_voltage_threshold_SI=1.0,
                         flap_getdata_options_voltage=None,
        ):
        """Automatically create a CalibrationReference object from a MAST-U shot.

        Calibration coordinates are:

        - APDCAM view radius [1 coordinate]:

          - 'APDCam_viewRadius' (float)

        - APDCAM set bias voltage [2 coordinates, only first two are set on MAST-U APDCam]:

          - 'APDCam_biasSet0' (float)

          - 'APDCam_biasSet1' (float)

        - Filter temperature reading [1 coordinate]

          - 'filter_temperature_read' (int)

        - SS beam voltage (99% percentile of signal values over voltage
          threshold) [1 coordinate]

          - 'SS_voltage_99' (float, rounded to a single digit)

        Parameters
        ----------
        shotnumber : int
            The MAST-U shot number to use.
        beam_voltage_threshold_SI : float, optional (default=1.0)
            Voltage threshold over which the beam is considered to be 'on'. The
            values above this are used to calculate the reference (or 'target')
            SS beam voltage
        flap_getdata_options_voltage : dict, optional (default=None)
            A dictionary of options to be passed on to the flap.get_data() call
            when loading the SS voltage measurement data.

        Returns
        -------
        CalibrationReference
            The CalibrationReference object containing the calibration
            coordinates of the given shot. The calibration multiplier matrix is
            not created automatically, but can be created separately, via the
            appropriate method.
        """
        if flap_getdata_options_voltage is None:
            flap_getdata_options_voltage = {}

        try:
            d_SS_VOLTAGE = flap.get_data('MAST-U', shotnumber, name='ANB/SS/VOLTAGE', options=flap_getdata_options_voltage)
        except Exception as e:
            raise ValueError(f'Shot {shotnumber} has no SS beam voltage data.') from e
        
        SS_VOLTAGE_99 = np.percentile(d_SS_VOLTAGE.data[d_SS_VOLTAGE.data > beam_voltage_threshold_SI], 99)
        
        shotstring = str(shotnumber).zfill(6)
        
        try:
            shots_folder_bes = flap.config.interpret_config_value(flap.config.get('Module MAST_BES', 'Cache directory'))
        except ValueError:
            shots_folder_bes = None
            
        if shots_folder_bes is not None:
            # Do this to trigger downloading
            _ = flap.get_data('MAST_BES', shotnumber, 'BES-5-5', options={'Calibrate intensity' : False})

            # Filter temperature is not available in the DataObject.info
            nc_file = os.path.join(shots_folder_bes, shotstring, file_name_from_shot_number(shotnumber))
                               
            with h5py.File(nc_file) as f:
                apdcam_viewRadius = f['devices']['d3_APDcamera10G'].attrs['viewRadius']
                apdcam_biasSet = f['devices']['d3_APDcamera10G'].attrs['biasSet']
                filter_temperature_read = f['devices']['d8_filterheater'].attrs['temperature_read']
        else:
            raise NotImplementedError('Only loading from cache directory directly is supported.')
        
        coordinates_dimful = {
            'APDCam_viewRadius': float(apdcam_viewRadius[0]),
            'APDCam_biasSet0': float(apdcam_biasSet[0]),
            'APDCam_biasSet1': float(apdcam_biasSet[1]),
            'filter_temperature_read' : int(filter_temperature_read[0]),
            'SS_voltage_99' : np.round(SS_VOLTAGE_99, decimals=1),
        }
    
        return cls(
            coordinates=coordinates_dimful,
            info={'shot' : shotnumber, 'beam_voltage_threshold_SI' : beam_voltage_threshold_SI},
            calibration_multiplier_matrix=None,  # Created in a separate method, only if needed
            calibration_source=None,  # Created in a separate method, only if needed
        )

    def create_calibration_matrix(self,
                                  time_interval_s,
                                  method='mean',
        ):
        """Create a calibration multiplier matrix from a measurement, associated
        to the CalibrationReference object.

        A calibration matrix is used to compensate for the relative differences
        of measured voltage levels corresponding to the same intensity on
        different BES channels.
        
        The calibration matrix is created by calculating an aggregate (e.g.
        mean) value in the provided time interval for each channel separately, then
        calculating a multiplier for each channel that brings the different
        voltages to the same level along all channels, which is the maximum
        value measured during the interval along all channels. Naturally, this
        method assumes that the actual intensity is identical along the
        channels.

        ----------
        time_interval_s : list of float
            Time interval from the measurement to be used as the basis of
            calibration.
        method : str, optional (default='mean')
            Aggregation method to be used. Currently only method 'mean' is
            supported.
        """

        shot = self.info['shot']
        d_reference = flap.get_data(
            'MAST_BES',
            exp_id=shot,
            name='BES-*-*',
            coordinates={'Time' : time_interval_s},
            options={'Calibrate intensity' : False},  # !!!
        )

        channel_map = d_reference.coordinate('ADC Channel')[0][0]

        self.calibration_source = {
            'time_interval_s' : list(time_interval_s),
            'channel_map' : channel_map.tolist(),
        }

        if method == 'mean':
            cal_data = d_reference.slice_data(summing={'Sample' : 'Mean'})
        else:
            raise ValueError("Only method 'mean' is supported currently.")

        # Convert ndarray to list so it can be serialized later
        self.calibration_multiplier_matrix = (np.max(cal_data.data) / cal_data.data).tolist()
    
class CalibrationDatabase():
    """Calibration database for storing CalibrationReference objects and finding
    the closest match.

    The database of CalibrationReference object can be queried, via
    `find_nearest_match()`, by matching along the calibration coordinates. Two
    types of calibration coordinates are considered: exactly and closely matched
    coordinates. These are to be provided separately and explicitly. Only these
    coordinates are considered in a query.

    Parameters
    ----------
    calibration_reference_list : list of CalibrationReference
        The list of CalibrationReference objects to query against. A set of K-d
        trees are created for each unique combination of exactly matched
        coordinates for quick lookup.
    closely_matched_coordinates : list of str
        List of coordinates to be matched not exactly but only in an Euclidean
        nearest neighbor sense. The nearest neighbor calculations are performed
        in a dimensionless space where all the coordinates have been normalized
        to the 0-1 range by the observed minima and maxima, to avoid improper
        matched that would occur due to different magnitude of the coordinates.
    exactly_matched_coordinates : list of str, optional (default=None)
        List of coordinates to be matched exactly. A nearest neighbor lookup is
        only performed if the database contains at least one
        CalibrationReference object for which all exactly matched coordinates
        are identical with those of the query.
    exactly_matched_round_to_decimals : int, optional (default=2)
        Exactly matched coordinates are rounded to the number of decimals given
        here to avoid floating point errors.
    non_matched_references : list of CalibrationReference, optional (default=None)
        In case the possible range of coordinates in available observations is
        outside the range of coordinates in the calibration database, additional
        reference objects can be supplied to avoid inaccurate lookups. These
        objects are only used to set the coordinate value minima and maxima used
        in the nondimensionalisation and are not queried.
    """
    def __init__(
            self,
            calibration_reference_list,
            closely_matched_coordinates,
            exactly_matched_coordinates=None,
            exactly_matched_round_to_decimals=2,
            non_matched_references=None
        ):
        if len(calibration_reference_list) == 0:
            raise ValueError('calibration_reference_list is an empty list.')

        self.calibration_reference_list = calibration_reference_list
        self.closely_matched_coordinates = closely_matched_coordinates
        self.exactly_matched_round_to_decimals = exactly_matched_round_to_decimals
        
        if exactly_matched_coordinates is None:
            self.exactly_matched_coordinates = []
        else:
            self.exactly_matched_coordinates = exactly_matched_coordinates

        self.matched_coordinates = self.closely_matched_coordinates + self.exactly_matched_coordinates

        self.closely_matched_coordinates_n = len(self.closely_matched_coordinates)
        self.exactly_matched_coordinates_n = len(self.exactly_matched_coordinates)

        coordinates_dimful_stacked = np.vstack([itemgetter(*self.matched_coordinates)(calibration_data.coordinates_dimful)
                                                for calibration_data
                                                in self.calibration_reference_list])
        if non_matched_references is not None:
            nm_coordinates_dimful_stacked = np.vstack([itemgetter(*self.matched_coordinates)(nm_data.coordinates_dimful)
                                                    for nm_data
                                                    in non_matched_references])

        # Prepend an index column
        coordinates_dimful_stacked = np.hstack(
            (np.atleast_2d(np.array(range(0, len(self.calibration_reference_list)))).T,
             coordinates_dimful_stacked)
        )

        # Determine the existing combinations of the exactly matched coordinates
        if self.exactly_matched_coordinates_n > 0:
            # Round the exactly matched coordinates to avoid floating point errors
            coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):] = np.round(
                coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):],
                decimals=self.exactly_matched_round_to_decimals,
            )

            # Do the same for the non-matched references
            if non_matched_references is not None:
                nm_coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):] = np.round(
                    nm_coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):],
                    decimals=self.exactly_matched_round_to_decimals,
                )

            # Find the available unique combinations of the exactly matched coordinates
            unique_exactly_matchables = [
                tuple(row)
                for row
                in np.unique(coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):], axis=0)]
        else:
            unique_exactly_matchables = [tuple()]

        # Build a database: dict indexed by the unique combinations of the exactly matchable coordinates
        # This database is always recreated at runtime: this is not ideal, but
        # for the current database size, it is fast enough. In the future, it could be serialized.
        
        self.database = {}

        # Create the database entries for the unique combinations of the exactly matchable coordinates
        for uem in unique_exactly_matchables:
            # Are there even any exactly matchable coordinates?
            if self.exactly_matched_coordinates_n > 0:
                # Filter the original list to matches
                reduced = coordinates_dimful_stacked[
                    np.all(
                        coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):] == uem,
                        axis=1,
                    )
                ][:, :self.closely_matched_coordinates_n + 1]  # indices and all closely matched coords

                # Do the same reduction for the non-matched references
                if non_matched_references is not None:
                    nm_reduced = nm_coordinates_dimful_stacked[
                        np.all(
                            nm_coordinates_dimful_stacked[:, -(self.exactly_matched_coordinates_n):] == uem,
                            axis=1,
                        )
                    ][:, :self.closely_matched_coordinates_n]  # no index here, only closely matched coords

            else:
                reduced = coordinates_dimful_stacked
                nm_reduced = nm_coordinates_dimful_stacked

            index = np.asarray(reduced[:, 0], dtype='int')

            # Drop the index column
            reduced = reduced[:, 1:]

            # Dimensionless, closely matched coordinates
            
            # Calculate scaling ranges
            scale_min = reduced.min(axis=0)
            scale_max = reduced.max(axis=0)

            if (non_matched_references is not None) and (len(nm_reduced) > 0):
                # Take the non-matched references into account as well (this is their only purpose)
                scale_min = np.min([scale_min, np.min(nm_reduced, axis=0)], axis=0)
                scale_max = np.max([scale_max, np.max(nm_reduced, axis=0)], axis=0)

            # Filter coordinates where there is no actual range taken by the values
            # to avoid division by 0 
            indices_with_range = ~(scale_min == scale_max)

            if np.sum(indices_with_range) == 0:
                # No coordinates with range
                if len(reduced) == 1:
                    # No problem, only one reference, this we can match
                    self.database[uem] = {
                        'db_index_list' : list(index),
                        'coordinates_with_range' : list(np.array(closely_matched_coordinates)[indices_with_range]),
                        'scale_min' : None,
                        'scale_max' : None,
                        'kdtree_dimless' : None,
                    }
                else:
                    # Otherwise, no match can be determined, since there are no
                    # coordinates with any range among the reference values
                    raise ValueError(f"The coordinates to be closely matched are all identical for the exactly matched combination {uem}, for reference instances with indices: {list(index)}.")
            else:
                coords_nontrivial_dimless = (reduced[:, indices_with_range] - scale_min[np.newaxis, indices_with_range]) / (scale_max[np.newaxis, indices_with_range] - scale_min[np.newaxis, indices_with_range])
                
                kdtree_dimless = KDTree(coords_nontrivial_dimless)

                self.database[uem] = {
                    'db_index_list' : list(index),
                    'coordinates_with_range' : list(np.array(closely_matched_coordinates)[indices_with_range]),
                    'scale_min' : list(scale_min[indices_with_range]),
                    'scale_max' : list(scale_max[indices_with_range]),
                    'kdtree_dimless' : kdtree_dimless,
                }

    def find_nearest_match(self, query_calibration_data):
        """Find the nearest match to a given CalibrationReference object in the
        database.

        Exactly matched coordinates must be identical to the coordinates of at
        least one object in the database for nearest-neighbor lookup to be
        performed.

        Parameters
        ----------
        query_calibration_data : CalibrationReference
            The object to find the closest match to.

        Returns
        -------
        dist_scaled : float
            Distance from the nearest neighbor scaled to the [0,1] range, where
            0 is an exact match and 1 is the largest possible distance.
        
        original_index : int
            The index of the query result in the original list provided during
            the construction of the database.
        
        query_result : CalibrationReference
            The best match to `query_calibration_data` in the database.
        """
        # First, check the coordinates to be matched exactly
        if self.exactly_matched_coordinates_n > 0:
            exactly_coords = tuple(query_calibration_data.coordinates_dimful[emc] for emc in self.exactly_matched_coordinates)
        else:
            exactly_coords = tuple()

        if exactly_coords not in self.database:
            raise ValueError(f"Coordinates {tuple(self.exactly_matched_coordinates)}={exactly_coords} could not be matched exactly in the current database.")
            
        # Then, find the nearest match among the instances where the other
        # coordinates are matched exactly

        # Get everything needed to perform the search
        matched_db_entry = self.database[exactly_coords]

        db_index_list = matched_db_entry['db_index_list']
        kdtree_dimless = matched_db_entry['kdtree_dimless']

        if kdtree_dimless is None:
            # Only a single reference matches
            if len(db_index_list) == 1:
                return -1, db_index_list[0], self.calibration_reference_list[db_index_list[0]]
            else:
                raise RuntimeError("Cannot determine match: no K-d tree for exactly matched coordinates, but len(db_index_list) > 1")
            
        scale_min = np.array(matched_db_entry['scale_min'])
        scale_max = np.array(matched_db_entry['scale_max'])
        coordinates_with_range = matched_db_entry['coordinates_with_range']

        # We only need the coordinates with range
        query_reduced_only_range = np.array([query_calibration_data.coordinates_dimful[cwr] for cwr in coordinates_with_range])

        # Scale it appropriately
        query_coords_nontrivial_dimless = (query_reduced_only_range - scale_min) / (scale_max - scale_min)

        # Find the match
        dist, nearest_i = kdtree_dimless.query(query_coords_nontrivial_dimless)

        # Index in the original list
        original_index = db_index_list[nearest_i]

        # Scale the distance to between 0 and 1:
        max_dist = np.sqrt(kdtree_dimless.m)
        dist_scaled = dist / max_dist

        return dist_scaled, original_index, self.calibration_reference_list[original_index]