# This file is part of pyESVEP for running ESVEP model
# Copyright 2018 Radoslaw Guzinski and contributors listed in the README.md file.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on May 30 2018
@author: Radoslaw Guzinski (rmgu@dhigroup.com)
'''

from collections import OrderedDict
from os.path import splitext, dirname, exists
from os import mkdir

import numpy as np
import ast
import gdal
from netCDF4 import Dataset

from pyTSEB.PyTSEB import PyTSEB, S_N, S_P, S_A
from pyTSEB import resistances as res
from pyTSEB import meteo_utils as met
from pyTSEB import net_radiation as rad
from pyMETRIC import METRIC, endmember_search

VI_MAX = 0.95

# Set the landcover classes that will be used for searching endmembers 
TREE_SEARCH_IGBP = [res.CONIFER_E,
                    res.BROADLEAVED_E,
                    res.CONIFER_D,
                    res.BROADLEAVED_D,
                    res.FOREST_MIXED,
                    res.SAVANNA_WOODY,
                    res.GRASS,
                    res.SHRUB_C,
                    res.SHRUB_O, 
                    res.CROP_MOSAIC,
                    res.BARREN]

HERBACEOUS_SEARCH = [res.SAVANNA,
                     res.GRASS,
                     res.CROP,
                     res.CROP_MOSAIC,
                     res.BARREN]


class PyMETRIC(PyTSEB):

    def __init__(self, parameters):
        if "use_METRIC_resistance" not in parameters.keys():
            parameters["use_METRIC_resistance"] = 1
        if "tall_reference" not in parameters.keys():
            parameters["tall_reference"] = 0
        if "ET_bare_soil" not in parameters.keys():
            parameters["ET_bare_soil"] = 0
        if "VI" not in parameters.keys():
            parameters["VI"] = ''
        if "endmember_search" not in parameters.keys():
            parameters["endmember_search"] = 0
        
        
        parameters["resistance_form"] = 0
        super().__init__(parameters)
        
        self.use_METRIC_resistance = int(self.p['use_METRIC_resistance'])
        self.tall_reference = int(self.p['tall_reference'])
        self.endmember_search = int(self.p['endmember_search'])

    def _get_input_structure(self):
        ''' Input fields' names for METRIC model.  Only relevant for image processing mode.

        Parameters
        ----------
        None

        Returns
        -------
        outputStructure: string ordered dict
            Names (keys) and descriptions (values) of TSEB_2T input fields.
        '''


        input_fields = super()._get_input_structure()
        del input_fields["leaf_width"]
        del input_fields["alpha_PT"]
        del input_fields["KN_b"]
        del input_fields["KN_c"]
        del input_fields["KN_C_dash"]
        
        input_fields["VI"] = "Vegetation Index"
        input_fields["ETrF_bare"] = "Reference ET for bare soil"
        input_fields["tall_reference"] = "Predominant land cover are trees"
        input_fields['alt'] = "Digital Elevation Model"
        input_fields['endmember_search'] = "Endmember search algorithm"
        return input_fields

    def _set_special_model_input(self, field, dims):
        ''' Special processing for setting certain input fields. Only relevant for image processing
        mode.

        Parameters
        ----------
        field : string
            The name of the input field for which the special processing is needed.
        dims : int list
            The dimensions of the output parameter array.

        Returns
        -------
        success : boolean
            True is the parameter was succefully set, false otherwise.
        array : float array
            The set parameter array.
        '''

        # Those fields are optional for METRIC.
        if field in ["x_LAD", "f_c", "w_C"]:
            success, val = self._set_param_array(field, dims)
            if not success:
                val = np.ones(dims)
                success = True
        else:
            success = False
            val = None
        return success, val

    def _get_output_structure(self):
        ''' Output fields' names for METRIC model.

        Parameters
        ----------
        None

        Returns
        -------
        output_structure: ordered dict
            Names of the output fields as keys and instructions on whether the output
            should be saved to file as values.
        '''

        output_structure = OrderedDict([
            # Energy fluxes
            ('R_n1', S_P),   # net radiation reaching the surface
            ('R_ns1', S_A),  # net shortwave radiation reaching the surface
            ('R_nl1', S_A),  # net longwave radiation reaching the surface
            ('H1', S_P),  # total sensible heat flux (W/m^2)
            ('LE1', S_P),  # total latent heat flux (W/m^2)
            ('G1', S_P),  # ground heat flux (W/m^2)
            # temperatures (might not be accurate)
            ('T_sd', S_A),  # End-member temperature of dry soil (Kelvin)
            ('T_vw', S_A),  # End-member temperature of well-watered vegetation (Kelvin)
            # resistances
            ('R_A1', S_A),  # Aerodynamic resistance to heat transport (s m-1)
            # miscaleneous
            ('omega0', S_N),  # nadir view vegetation clumping factor
            ('L', S_A),  # Monin Obukhov Length
            ('theta_s1', S_N),  # Sun zenith angle
            ('F', S_N),  # Leaf Area Index
            ('z_0M', S_N),  # Aerodynamic roughness length for momentum trasport (m)
            ('d_0', S_N),  # Zero-plane displacement height (m)
            ('Skyl', S_N),
            ('L', S_A),  # Monin Obukhov Length at time t1
            ('u_friction', S_A), # Friction velocity
            ('flag', S_A),  # Quality flag
            ('n_iterations', S_N), # Number of iterations before model converged to stable value
            ('ET0', S_A)])  

        return output_structure

    def process_local_image(self):
        ''' Runs pyMETRIC for all the pixel in an image'''

        #======================================
        # Process the input

        # Create an input dictionary
        in_data = dict()

        if 'subset' in self.p:
            subset = ast.literal_eval(self.p['subset'])
        else:
            subset = []
        
        # Open the LST data according to the model
        try:
            fid = gdal.Open(self.p['T_R1'], gdal.GA_ReadOnly)
            prj = fid.GetProjection()
            geo = fid.GetGeoTransform()
            if subset:                   
                in_data['T_R1'] = fid.GetRasterBand(1).ReadAsArray(subset[0],
                                                        subset[1],
                                                        subset[2],
                                                        subset[3])
                geo = [geo[0]+subset[0]*geo[1], geo[1], geo[2], 
                       geo[3]+subset[1]*geo[5], geo[4], geo[5]]
            else:
                in_data['T_R1'] = fid.GetRasterBand(1).ReadAsArray()
            dims = np.shape(in_data['T_R1'])

        except:
            print('Error reading sunrise LST file ' + str(self.p['T_R0']))
            fid = None
            return
                 
        # Read the image mosaic and get the LAI
        success, in_data['LAI'] = self._open_GDAL_image(
            self.p['LAI'], dims, 'Leaf Area Index', subset)
        if not success:
            return
        # Read the image mosaic and get the Vegetation Index
        success, in_data['VI'] = self._open_GDAL_image(
            self.p['VI'], dims, 'Vegetation Index', subset)
        if not success:
            return


        # Read the fractional cover data
        success, in_data['f_c'] = self._open_GDAL_image(
            self.p['f_c'], dims, 'Fractional Cover', subset)
        if not success:
            return
        # Read the Canopy Height data
        success, in_data['h_C'] = self._open_GDAL_image(
            self.p['h_C'], dims, 'Canopy Height', subset)
        if not success:
            return
        # Read the canopy witdth ratio
        success, in_data['w_C'] = self._open_GDAL_image(
            self.p['w_C'], dims, 'Canopy Width Ratio', subset)
        if not success:
            return
        # Read landcover
        success, in_data['landcover'] = self._open_GDAL_image(
            self.p['landcover'], dims, 'Landcover', subset)
        if not success:
            return
        # Read leaf angle distribution
        success, in_data['x_LAD'] = self._open_GDAL_image(
            self.p['x_LAD'], dims, 'Leaf Angle Distribution', subset)
        if not success:
            return
        # Read digital terrain model
        success, in_data['alt'] = self._open_GDAL_image(
            self.p['alt'], dims, 'Digital Terrain Model', subset)
        if not success:
            return
        
        # Read spectral properties
        success, in_data['rho_vis_C'] = self._open_GDAL_image(
            self.p['rho_vis_C'], dims, 'Leaf PAR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_vis_C'] = self._open_GDAL_image(
            self.p['tau_vis_C'], dims, 'Leaf PAR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_nir_C'] = self._open_GDAL_image(
            self.p['rho_nir_C'], dims, 'Leaf NIR Reflectance', subset)
        if not success:
            return
        success, in_data['tau_nir_C'] = self._open_GDAL_image(
            self.p['tau_nir_C'], dims, 'Leaf NIR Transmitance', subset)
        if not success:
            return
        success, in_data['rho_vis_S'] = self._open_GDAL_image(
            self.p['rho_vis_S'], dims, 'Soil PAR Reflectance', subset)
        if not success:
            return
        success, in_data['rho_nir_S'] = self._open_GDAL_image(
            self.p['rho_nir_S'], dims, 'Soil NIR Reflectance', subset)
        if not success:
            return
        success, in_data['emis_C'] = self._open_GDAL_image(
            self.p['emis_C'], dims, 'Leaf Emissivity', subset)
        if not success:
            return
        success, in_data['emis_S'] = self._open_GDAL_image(
            self.p['emis_S'], dims, 'Soil Emissivity', subset)
        if not success:
            return
        
        # Calculate illumination conditions
        success, lat = self._open_GDAL_image(self.p['lat'], dims, 'Latitude', subset)
        if not success:
            return
        success, lon = self._open_GDAL_image(self.p['lon'], dims, 'Longitude', subset)
        if not success:
            return
        success, stdlon = self._open_GDAL_image(self.p['stdlon'], dims, 'Standard Longitude', subset)
        if not success:
            return
        success, in_data['time'] = self._open_GDAL_image(self.p['time'], dims, 'Time', subset)
        if not success:
            return
        success, doy = self._open_GDAL_image(self.p['DOY'], dims, 'DOY', subset)
        if not success:
            return
        in_data['SZA'], in_data['SAA'] = met.calc_sun_angles(
            lat, lon, stdlon, doy, in_data['time'])
        
        del lat, lon, stdlon, doy
        
        # Wind speed
        success, in_data['u'] = self._open_GDAL_image(
            self.p['u'], dims, 'Wind speed', subset)
        if not success:
            return
        # Vapour pressure
        success, in_data['ea'] = self._open_GDAL_image(
            self.p['ea'], dims, 'Vapour pressure', subset)
        if not success:
            return
        # Air pressure
        success, in_data['p'] = self._open_GDAL_image(
            self.p['p'], dims, 'Pressure', subset)
        # If pressure was not provided then estimate it based of altitude
        if not success:
            success, alt = self._open_GDAL_image(self.p['alt'], dims, 'Altitude', subset)
            if success:
                in_data['p'] = met.calc_pressure(alt)
            else:
                return
        success, in_data['S_dn'] = self._open_GDAL_image(
            self.p['S_dn'], dims, 'Shortwave irradiance', subset)
        if not success:
            return
        # Wind speed measurement height
        success, in_data['z_u'] = self._open_GDAL_image(
            self.p['z_u'], dims, 'Wind speed height', subset)
        if not success:
            return
        # Air temperature mesurement height
        success, in_data['z_T'] = self._open_GDAL_image(
            self.p['z_T'], dims, 'Air temperature height', subset)
        if not success:
            return
        # Soil roughness
        success, in_data['z0_soil'] = self._open_GDAL_image(
            self.p['z0_soil'], dims, 'Soil Roughness', subset)
        if not success:
            return 

        # Incoming long wave radiation
        success, in_data['T_A1'] = self._open_GDAL_image(
            self.p['T_A1'], dims, 'Air Temperature', subset)
        if not success:
            return
        success, in_data['L_dn'] = self._open_GDAL_image(
            self.p['L_dn'], dims, 'Longwave irradiance', subset)
        # If longwave irradiance was not provided then estimate it based on air
        # temperature and humidity
        if not success:
            emisAtm = rad.calc_emiss_atm(in_data['ea'], in_data['T_A1'])
            in_data['L_dn'] = emisAtm * met.calc_stephan_boltzmann(in_data['T_A1'])
            
        # Open the processing maks and get the id for the cells to process
        if self.p['input_mask'] != '0':
            success, mask = self._open_GDAL_image(
                self.p['input_mask'], dims, 'input mask', subset) 
            if not success:
                print ("Please set input_mask=0 for processing the whole image.")
                return
        # Otherwise create mask from landcover array
        else:
            mask = np.ones(dims)
            mask[np.logical_or.reduce((in_data['landcover'] == res.WATER, 
                                       in_data['landcover'] == res.URBAN,
                                       in_data['landcover'] == res.SNOW))] = 0
        
        mask[np.logical_or(~np.isfinite(in_data['VI']),
                           ~np.isfinite(in_data['T_R1']))] = 0            
        # Open the bare soil ET
        if str(self.p['ET_bare_soil']) != '0':
           success,  in_data['ET_bare_soil'] = self._open_GDAL_image(
                                                       self.p['ET_bare_soil'], 
                                                       dims, 
                                                       'ET_bare_soil', 
                                                       subset) 
           if not success:
                print ("Please set ET_bare_soil=0 for assuming zero ET for bare soil.")
                return
        # Otherwise assume ET soil = 0
        else:
            in_data['ET_bare_soil'] = np.zeros(dims)

        # Get the Soil Heat flux if G_form includes the option of measured G
        # Constant G or constant ratio of soil reaching radiation
        if self.G_form[0][0] == 0 or self.G_form[0][0] == 1:  
            success, self.G_form[1] = self._open_GDAL_image(
                                            self.G_form[1], dims, 'G', subset)
            if not success:
                return   
        # Santanello and Friedls G
        elif self.G_form[0][0] == 2:  
            # Set the time in the G_form flag to compute the Santanello and
            # Friedl G
            self.G_form[1] = in_data['time']
        # ASCE G ratios
        elif self.G_form[0][0] == 4:  
            # Set the time in the G_form flag to compute G using ASCE G ratios
            if self.tall_reference:
                self.G_form[1] = np.ones(dims) * 0.04
            else:
                self.G_form[1] = np.ones(dims) * 0.10
       
        del in_data['time']
        
        #======================================
        # Run the chosen model
        
        out_data = self.run_METRIC(in_data, mask)


        #======================================
        # Save output files

        # Output variables saved in images
        self.fields = ('H1', 'LE1', 'R_n1', 'G1')
        # Ancillary output variables
        self.anc_fields = (
            'R_ns1',
            'R_nl1',
            'u_friction',
            'L',
            'R_A1',
            'flag')
        
        outdir = dirname(self.p['output_file'])
        if not exists(outdir):
            mkdir(outdir)
        self._write_raster_output(
            self.p['output_file'],
            out_data,
            geo,
            prj,
            self.fields)
        outputfile = splitext(self.p['output_file'])[0] + '_ancillary' + \
                     splitext(self.p['output_file'])[1]
        self._write_raster_output(
            outputfile,
            out_data,
            geo,
            prj,
            self.anc_fields)
        print('Saved Files')
        
        return in_data, out_data


    def run_METRIC(self, in_data, mask=None):
        ''' Execute the routines to calculate energy fluxes.

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        mask : int array or None
            If None then fluxes will be calculated for all input points. Otherwise, fluxes will be
            calculated only for points for which mask is 1.

        Returns
        -------
        out_data : dict
            The output data from the model.
        '''

        print("Processing...")
        model_params = dict()
        
        if mask is None:
            mask = np.ones(in_data['LAI'].shape)

        # Create the output dictionary
        out_data = dict()
        for field in self._get_output_structure():
            out_data[field] = np.zeros(in_data['LAI'].shape) + np.NaN
            
        print('Estimating net shortwave radiation using Cambpell two layers approach')
        # Esimate diffuse and direct irradiance
        difvis, difnir, fvis, fnir = rad.calc_difuse_ratio(
            in_data['S_dn'], in_data['SZA'], press=in_data['p'])
        out_data['fvis'] = fvis
        out_data['fnir'] = fnir
        out_data['Skyl'] = difvis * fvis + difnir * fnir
        out_data['S_dn_dir'] = in_data['S_dn'] * (1.0 - out_data['Skyl'])
        out_data['S_dn_dif'] = in_data['S_dn'] * out_data['Skyl']
        
        del difvis, difnir, fvis, fnir
        
        aoi = np.zeros(in_data['T_R1'].shape, dtype=bool)

        if self.tall_reference:
            for lc in TREE_SEARCH_IGBP:
                aoi[np.logical_and(in_data['landcover']==lc, mask==1)] = 1
        else:
            for lc in HERBACEOUS_SEARCH:
                aoi[np.logical_and(in_data['landcover']==lc, mask==1)] = 1
        
        # ======================================
        # NEt radiation for bare soil
        noVegPixels = np.logical_and(in_data['LAI'] == 0, aoi == 1)
        # Calculate roughness
        out_data['z_0M'][noVegPixels] = in_data['z0_soil'][noVegPixels]
        out_data['d_0'][noVegPixels] = 0

        # Net shortwave radition for bare soil
        spectraGrdOSEB = out_data['fvis'] * \
            in_data['rho_vis_S'] + out_data['fnir'] * in_data['rho_nir_S']
        out_data['R_ns1'][noVegPixels] = (1. - spectraGrdOSEB[noVegPixels]) * \
            (out_data['S_dn_dir'][noVegPixels] + out_data['S_dn_dif'][noVegPixels])
     
        
        # ======================================
        # Then process vegetated cases
        # Calculate roughness
        i = ~noVegPixels
        out_data['z_0M'][i], out_data['d_0'][i] = \
            res.calc_roughness(in_data['LAI'][i],
                               in_data['h_C'][i],
                               w_C=in_data['w_C'][i],
                               landcover=in_data['landcover'][i],
                               f_c=in_data['f_c'][i])
        
        del in_data['h_C'], in_data['w_C'], in_data['landcover'], in_data['f_c']
        
        Sn_C1, Sn_S1 = rad.calc_Sn_Campbell(in_data['LAI'][i],
                                                      in_data['SZA'][i],
                                                      out_data['S_dn_dir'][i],
                                                      out_data['S_dn_dif'][i],
                                                      out_data['fvis'][i],
                                                      out_data['fnir'][i],
                                                      in_data['rho_vis_C'][i],
                                                      in_data['tau_vis_C'][i],
                                                      in_data['rho_nir_C'][i],
                                                      in_data['tau_nir_C'][i],
                                                      in_data['rho_vis_S'][i],
                                                      in_data['rho_nir_S'][i],
                                                      x_LAD=in_data['x_LAD'][i])
        
        out_data['R_ns1'][i] = Sn_C1 + Sn_S1
        del Sn_C1, Sn_S1, in_data['LAI'], in_data['x_LAD']
        del in_data['rho_vis_C'], in_data['tau_vis_C'], in_data['rho_nir_C'], in_data['tau_nir_C']
        del in_data['rho_vis_S'], in_data['rho_nir_S']
        
        out_data['emiss'] = in_data['VI']*in_data['emis_C'] + (1 -in_data['VI']) *in_data['emis_S']
        
        del in_data['emis_C'], in_data['emis_S']

        out_data['albedo'] = 1 - out_data['R_ns1']/in_data['S_dn']

        print('Automatic search of METRIC hot and cold pixels')
        # Find hot and cold endmembers in the Area of Interest
        gamma_w = met.calc_lapse_rate_moist(in_data['T_A1'],
                                            in_data['ea'],
                                            in_data['p'])
        
        Tr_datum = in_data['T_R1'] + gamma_w * in_data['alt']
        Ta_datum = in_data['T_A1'] + gamma_w * in_data['alt']
        
        del gamma_w

# =============================================================================
#         # Estimate cloudiness factor
#         [Rdirvis,
#          Rdifvis,
#          Rdirnir,
#          Rdifnir] = rad.calc_potential_irradiance_weiss(in_data['SZA'],
#                                                         press=in_data['p'])
#         
#         S_dn_0 = Rdirvis + Rdifvis + Rdirnir + Rdifnir
#         
#         
#         f_cd = METRIC.calc_cloudiness(in_data['S_dn'], S_dn_0)
# 
#         del Rdirvis, Rdifvis, Rdirnir, Rdifnir, S_dn_0
#         
# =============================================================================
        out_data['ET0_datum'] = METRIC.pet_asce(Ta_datum,
                                          in_data['u'],
                                          in_data['ea'],
                                          in_data['p'],
                                          in_data['S_dn'],
                                          in_data['z_u'],
                                          in_data['z_T'],
                                          f_cd=1,
                                          reference=self.tall_reference)

        out_data['ET0'] = METRIC.pet_asce(in_data['T_A1'],
                                          in_data['u'],
                                          in_data['ea'],
                                          in_data['p'],
                                          in_data['S_dn'],
                                          in_data['z_u'],
                                          in_data['z_T'],
                                          f_cd=1,
                                          reference=self.tall_reference)
        
        del in_data['S_dn']
        
        # Compute spatial homogeneity metrics
        cv_ndvi, _, _ = endmember_search.moving_cv_filter(in_data['VI'], (10, 10))
        cv_lst, _, std_lst = endmember_search.moving_cv_filter(Tr_datum, (10, 10))
        cv_albedo,_, _ = endmember_search.moving_cv_filter(out_data['albedo'], (10, 10))
        
        # Find hot/cold endmembers
        if self.endmember_search == 0:
            [in_data['cold_pixel'],
             in_data['hot_pixel']] = endmember_search.cimec(in_data['VI'][aoi],
                                                            Tr_datum[aoi],
                                                            out_data['albedo'][aoi],
                                                            in_data['SZA'][aoi],
                                                            cv_ndvi[aoi],
                                                            cv_lst[aoi],
                                                            adjust_rainfall = False)

        elif self.endmember_search == 1:
            [in_data['cold_pixel'], 
             in_data['hot_pixel']] = endmember_search.esa(in_data['VI'][aoi],
                                                          Tr_datum[aoi],
                                                          cv_ndvi[aoi],
                                                          std_lst[aoi],
                                                          cv_albedo[aoi])
        
        else:
            [in_data['cold_pixel'],
             in_data['hot_pixel']] = endmember_search.cimec(in_data['VI'][aoi],
                                                            Tr_datum[aoi],
                                                            out_data['albedo'][aoi],
                                                            in_data['SZA'][aoi],
                                                            cv_ndvi[aoi],
                                                            cv_lst[aoi],
                                                            adjust_rainfall = False)

        # Reduce potential ET based on vegetation density based on Allen et al. 2013
        out_data['ET_r_f_cold'] = np.ones(in_data['T_R1'].shape) * 1.05
        out_data['ET_r_f_cold'][in_data['VI'] < VI_MAX] = 1.05/VI_MAX * in_data['VI'][in_data['VI'] < VI_MAX] # Eq. 4 [Allen 2013]
        
        out_data['ET_r_f_hot'] = in_data['VI'] * out_data['ET_r_f_cold'] \
                                 + (1.0 - in_data['VI']) * in_data['ET_bare_soil'] # Eq. 5 [Allen 2013]
        
        del in_data['SZA'], in_data['VI'], Tr_datum, cv_ndvi, cv_lst, std_lst, cv_albedo
        
        out_data['T_sd'][:] = float(in_data['T_R1'][aoi][in_data['hot_pixel']])
        out_data['T_vw'][:] = float(in_data['T_R1'][aoi][in_data['cold_pixel']])

        # Model settings
        model_params["calcG_params"] = [self.G_form[0], self.G_form[1][aoi]]
 
        # Other fluxes for vegetation
        self._call_flux_model(in_data, out_data, model_params, aoi)
        
        del in_data, model_params, aoi

        # Calculate the global net radiation
        out_data['R_n1'] = out_data['R_ns1'] + out_data['R_nl1']

        print("Finished processing!")
        return out_data


    def _call_flux_model(self, in_data, out_data, model_params, i):
        ''' Call METRIC model to calculate fluxes for all data points

        Parameters
        ----------
        in_data : dict
            The input data for the model.
        out_data : dict
            Dict containing the output data from the model which will be updated. It also contains
            previusly calculated shortwave radiation and roughness values which are used as input
            data.

        Returns
        -------
        None
        '''
        
        [out_data['flag'][i], out_data['R_nl1'][i], out_data['LE1'][i], out_data['H1'][i],
         out_data['G1'][i], out_data['R_A1'][i], out_data['u_friction'][i], out_data['L'][i],
         out_data['n_iterations'][i]] = \
                 METRIC.METRIC(in_data['T_R1'][i],
                               in_data['T_A1'][i],
                               in_data['u'][i],
                               in_data['ea'][i],
                               in_data['p'][i],
                               out_data['R_ns1'][i],
                               in_data['L_dn'][i],
                               out_data['emiss'][i],
                               out_data['z_0M'][i],
                               out_data['d_0'][i],
                               in_data['z_u'][i],
                               in_data['z_T'][i],
                               in_data['cold_pixel'],
                               in_data['hot_pixel'],
                               out_data['ET_r_f_cold'][i] * out_data['ET0_datum'][i],
                               LE_hot=out_data['ET_r_f_hot'][i] * out_data['ET0_datum'][i],
                               use_METRIC_resistance = self.use_METRIC_resistance,
                               calcG_params=model_params["calcG_params"],
                               UseDEM=in_data['alt'][i])

    def _open_GDAL_image(self, inputString, dims, variable, subset = []):
        '''Open a GDAL image and returns and array with its first band'''

        if inputString == "":
            return False, None
      
        success = True
        array = None
        try:
            array = np.zeros(dims) + float(inputString)
        except:
            try:
                fid = gdal.Open(inputString, gdal.GA_ReadOnly)
                if subset:
                    array = fid.GetRasterBand(1).ReadAsArray(subset[0],
                                                             subset[1],
                                                             subset[2],
                                                             subset[3])
                else:
                    array = fid.GetRasterBand(1).ReadAsArray()
            except:
                print(
                    'ERROR: file read ' +
                    str(inputString) +
                    '\n Please type a valid file name or a numeric value for ' +
                    variable)
                success = False
            finally:
                fid = None
                
        return success, array

    def _write_raster_output(self, outfile, output, geo, prj, fields):
        '''Writes the arrays of an output dictionary which keys match the list 
           in fields to a raster file '''

        # If the output file has .nc extension then save it as netCDF,
        # otherwise assume that the output should be a GeoTIFF
        ext = splitext(outfile)[1]
        if ext.lower() == ".nc":
            driver = "netCDF"
            opt = ["FORMAT=NC2"]
            is_netCDF = True
        else:
            driver = "GTiff"
            opt = []
            is_netCDF = False

        # Save the data using GDAL
        rows, cols = np.shape(output['H1'])
        driver = gdal.GetDriverByName(driver)
        nbands = len(fields)
        ds = driver.Create(outfile, cols, rows, nbands, gdal.GDT_Float32, opt)
        ds.SetGeoTransform(geo)
        ds.SetProjection(prj)
        for i, field in enumerate(fields):
            band = ds.GetRasterBand(i + 1)
            band.SetNoDataValue(np.NaN)
            band.WriteArray(output[field])
            band.FlushCache()
        ds.FlushCache()
        ds = None
        
        # In case of netCDF format use netCDF4 module to assign proper names 
        # to variables (GDAL can't do this). Also it seems that GDAL has
        # problems assigning projection to all the bands so fix that.
        if is_netCDF:
            ds = Dataset(outfile, 'a')
            grid_mapping = ds["Band1"].grid_mapping
            for i, field in enumerate(fields):
                ds.renameVariable("Band"+str(i+1), field)
                ds[field].grid_mapping = grid_mapping
            ds.close()
