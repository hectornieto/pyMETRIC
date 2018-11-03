# This file is part PyTSEB, consisting of of high level pyTSEB scripting
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

from re import match

from pyMETRIC.PyMETRIC import PyMETRIC


class METRICConfigFileInterface():

    def __init__(self):

        self.params = {}
        self.ready = False

        temp_params = {'model': 'METRIC', 'use_METRIC_resistance': 1, 'G_form': 0}
        temp_model = PyMETRIC(temp_params)
        self.input_vars = temp_model._get_input_structure().keys()

    def parse_input_config(self, input_file, is_image=True):
        ''' Parses the information contained in a configuration file into a dictionary'''

        if not is_image:
            print("Point time-series interface is not implemented for ESVEP!")
            return None

        # Read contents of the configuration file
        config_data = dict()
        try:
            with open(input_file, 'r') as fid:
                for line in fid:
                    if match('\s', line):  # skip empty line
                        continue
                    elif match('#', line):  # skip comment line
                        continue
                    elif '=' in line:
                        # Remove comments in case they exist
                        line = line.split('#')[0].rstrip(' \r\n')
                        field, value = line.split('=')
                        config_data[field] = value
        except IOError:
            print('Error reading ' + input_file + ' file')

        return config_data

    def get_data(self, config_data, is_image):
        '''Parses the parameters in a configuration file directly to METRIC variables for running
           METRIC'''

        if not is_image:
            print("Point time-series interface is not implemented for METRIC!")
            return None

        try:
            for var_name in self.input_vars:
                try:
                    self.params[var_name] = str(config_data[var_name]).strip('"')
                except KeyError:
                    pass

            self.params['model'] = config_data['model']

            if 'calc_row' not in config_data or int(config_data['calc_row']) == 0:
                self.params['calc_row'] = [0, 0]
            else:
                self.params['calc_row'] = [
                    1,
                    float(config_data['row_az'])]

            if int(config_data['G_form']) == 0:
                self.params['G_form'] = [[0], float(config_data['G_constant'])]
            elif int(config_data['G_form']) == 1:
                self.params['G_form'] = [[1], float(config_data['G_ratio'])]
            elif int(config_data['G_form']) == 2:
                self.params['G_form'] = [[2,
                                         float(config_data['G_amp']),
                                         float(config_data['G_phase']),
                                         float(config_data['G_shape'])],
                                         12.0]
            elif int(config_data['G_form']) == 4:
                self.params['G_form'] = [[4],
                                         (float(config_data['G_tall']), 
                                          float(config_data['G_short']))]

            self.params['output_file'] = config_data['output_file']

            self.ready = True

        except KeyError as e:
            print('Error: missing parameter '+str(e)+' in the input data.')
        except ValueError as e:
            print('Error: '+str(e))

    def run(self, is_image):

        if not is_image:
            print("Point time-series interface is not implemented for METRIC!")
            return None

        if self.ready:
            if self.params['model'] == "pyMETRIC":
                model = PyMETRIC(self.params)
            else:
                print("Unknown model: " + self.params['model'] + "!")
                return None
            model.process_local_image()
            
        else:
            print("pyMETRIC will not be run due to errors in the input data.")
