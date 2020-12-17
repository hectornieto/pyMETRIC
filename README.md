# pyMETRIC

## Synopsis

This project contains *Python* code for *Mapping EvapoTranspiration at high Resolution with Internalized Calibration* model for estimating sensible and latent heat flux (evapotranspiration) based on measurements of radiometric surface temperature. 

The project consists of: 

1. lower-level modules with the basic functions needed in any resistance energy balance model 
2. higher-level script for easily running METRIC with satellite/airborne imagery.

## Installation

Download the project to your local system, enter the download directory and then type

`python setup.py install` 

if you want to install pyMETRIC and its low-level modules in your Python distribution. 

The following Python libraries will be required:

- pyTSEB
- pyTVDI
- Numpy
- Pandas
- GDAL

## Code Example
### High-level example
You can run METRIC with the scripts METRIC_local_image_main.py, which will read an input configuration file (default is Config_LocalImage.txt). You can edit this configuration file or make a copy to fit your data and site characteristics and either run it in a Python GUI or in a terminal shell:

`python TSEB_local_image_main.py <configuration file>`
> where \<configuration file> points to a customized configuration file... leave it blank if you want to use the default file Config_LocalImage.txt


### Low-level example
You can also run METRIC or any related process in python by importing the module *METRIC* from the *pyMETRIC* package. 
It will also import the ancillary modules (*pyTSEB.resitances.py* as `res`, *pyTSEB.net_radiation* as `rad`,
*pyTSEB.MO_similarity.py* as `MO`, and *pyTSEB.meteo_utils* as `met`)

```python
import pyMETRIC.METRIC as METRIC 
output=METRIC.METRIC(Tr_K, T_A_K, u, ea, p, Sn, L_dn, emis, z_0M, d_0, z_u, z_T, cold_pixel, hot_pixel, LE_cold)
```

You can type
`help(METRIC.METRIC)`
to understand better the inputs needed and the outputs returned

The direct and difuse shortwave radiation (`Sn`) and the downwelling longwave radiation (`L_dn`) can be estimated by

```python
emisAtm = TSEB.rad.calc_emiss_atm(ea,Ta_K_1) # Estimate atmospheric emissivity from vapour pressure (mb) and air Temperature (K)
L_dn = emisAtm * TSEB.met.calc_stephan_boltzmann(Ta_K_1) # in W m-2
Sn = Sdn * (1.0 - albedo)
```
   
## Basic Contents
### Low-level modules
The low-level modules in this project are aimed at providing customisation and more flexibility in running METRIC. 
The following modules are included

- *.pyMETRIC/METRIC.py*
> core functions for running METRIC. 

- *.pyMETRIC/endmember_search.py*
> functions automatically finding the cold and hot pixels in an image

## Main Scientific References
- Allen, R. G., Tasumi, M., and Trezza, R. (2007). Satellite-based energy balance for mapping evapotranspiration with internalized calibration (METRIC)—Model. Journal of irrigation and drainage engineering, 133(4), 380-394.
- Allen, Richard G., Boyd Burnett, William Kramber, Justin Huntington, Jeppe Kjaersgaard, Ayse Kilic, Carlos Kelly, and Ricardo Trezza, 2013. Automated Calibration of the METRIC Landsat Evapotranspiration Process. Journal of the American Water Resources Association (JAWRA) .49(3):563–576 https://doi.org/10.1111/jawr.12056
- Nishan Bhattarai, Lindi J. Quackenbush, Jungho Im, tephen B. Shaw, 2017. A new optimized algorithm for automating endmember pixel selection in the SEBAL and METRIC models. Remote Sensing of Environment, Volume 196, Pages 178-192, https://doi.org/10.1016/j.rse.2017.05.009.

## Contributors
- **Hector Nieto** <hector.nieto@irta.cat> <hector.nieto.solana@gmail.com> main developer
- **Radoslaw Guzinski** main developer, tester

## License
pyMETRIC: Python code for Mapping EvapoTranspiration at high Resolution with Internalized Calibration Model

Copyright 2018 Hector Nieto and contributors.
    
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
