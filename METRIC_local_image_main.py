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

import sys

from pyMETRIC.METRICConfigFileInterface import METRICConfigFileInterface

config_file = 'Config_LocalImage.txt'


def run_METRIC_from_config_file(config_file):
    # Create an interface instance
    setup = METRICConfigFileInterface()
    # Get the data from configuration file
    config_data = setup.parse_input_config(config_file, is_image=True)
    setup.get_data(config_data, is_image=True)
    # Run the model
    setup.run(is_image=True)
    return

if __name__ == '__main__':
    args = sys.argv
    if len(args) > 1:
        config_file = args[1]
    print('Run pyMETRIC with configuration file = ' + str(config_file))
    run_METRIC_from_config_file(config_file)
