# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 18:56:06 2018

@author: hector
"""

import pyTVDI
import numpy as np

NDVI_SOIL = 0.1
NDVI_FULL = 0.75

def find_metric_endmembers(ndvi_array, lst_array, albedo_array, criterion=[0, {}]):
    '''Image based search for hot and cold pixels
    
    Parameters
    ----------
    ndvi_array : ndvi image numpy array
    lst_array : ndvi image numpy array
    criterion : search criteria dictionary
        [flag] = 0 : max/min temperature
        [flag] = 1 : mean temperature for the %perc highest/lowest NDVI pixels
            [perc] : two tail percentile to apply
        [flag] = 2 : triangle
            [hot_ndvi] : ndvi value at which the dry edge will be extrapolated
                        to get the hot pixel
            [cold_flag] : flag that set the criterion to get the cold edge
                if flag is numeric the maxvar_VI method is used with flag value 
                set as the maximum NDVI
    
    Returns
    -------
    hot_pixel : tuple with the pixel coordinates of the hot pixel
    cold_pixel : tuple with the pixel coordinates of the cold pixel'''
    
    # Default method is the absolute max/min temperature
    
    if criterion[0] == 0: #ManMin Temperature Method
        cold_pixel, hot_pixel = maxmin_temperature(ndvi_array,lst_array)
        
    elif criterion[0] == 1:#Percentile Method
        if 'perc' not in criterion[1].keys:
            criterion[1]['perc'] = 5 #Default is 5% percentile
        cold_pixel, hot_pixel = percentile_temperature(ndvi_array, lst_array, criterion['perc'])

    elif criterion[0] == 2:#TVDI Method
        if criterion[1]['cold_edge_method'] == 0: #Wet edge (cold LST) used the mean
            ts_min_method = pyTVDI.cMEAN
            ts_min_params = [20, 1]
        else:  #Wet edge (cold LST) used the variable maximum NDVI
            ts_min_method = pyTVDI.cVAR_MAX_NDVI
            ts_min_params = [float(criterion[1]['cold_edge_param']), 1]
        
        # Calculate the triangl
        cold_pixel, hot_pixel = TVDI_temperature(ndvi_array,
                                                lst_array,
                                                ts_min_method,
                                                ts_min_params,
                                                ndvi_lower_limit = NDVI_SOIL)
    
    return cold_pixel, hot_pixel
       
def maxmin_temperature(ndvi_array, lst_array, ndvi_lower_limit = NDVI_SOIL):
    #cold pixel
    lst = np.amin(lst_array[ndvi_array >= ndvi_lower_limit])
    cold_pixel = tuple(np.argwhere(lst_array == lst)[0])
    print('Cold  pixel found with %s K and %s VI'%(float(lst_array[cold_pixel]),
                                                   float(vi_array[cold_pixel])))
    #hot pixel
    lst = np.amax(lst_array[ndvi_array >= ndvi_lower_limit])
    hot_pixel = tuple(np.argwhere(lst_array == lst)[0])
    print('Hot  pixel found with %s K and %s VI'%(float(lst_array[hot_pixel]),
                                                   float(vi_array[hot_pixel])))
   
    return cold_pixel, hot_pixel

def TVDI_temperature(ndvi_array,
                    lst_array,
                    ts_min_method,
                    ts_min_params,
                    ndvi_lower_limit = NDVI_SOIL):

    lin_fit, ts_min, fit_stats=pyTVDI.calc_triangle_tang(ndvi_array,
                                                         lst_array,
                                                         0.01,
                                                         ndvi_lower_limit,
                                                         ts_min_method, 
                                                         ts_min_params)
    
    cold_pixel = tuple(np.argwhere(lst_array == ts_min)[0])
    lst = lin_fit[0] + lin_fit[1] * ndvi_lower_limit
    hot_pixel = tuple(np.argwhere(lst_array == lst)[0])

    return hot_pixel, cold_pixel

def cimec(ndvi_array,
          lst_array,
          albedo_array,
          sza_array,
          cv_ndvi,
          cv_lst,
          ETrF_bare = 0,
          adjust_rainfall = False):
    ''' Finds the hot and cold pixel using the 
    Calibration using Inverse Modelling at Extreme Conditios
    
    Parameters
    ----------
    ndvi_array : numpy array
        NDVI array (-)
    lst_array : numpy array
        Land Surface Temperature array (Kelvin)
    albedo : numpy array
        surface albedo
    sza : numpy array
        solar zenith angle (degrees)
    cv_lst : numpy array
        Coefficient of variation of LST as homogeneity measurement
        form neighboring pixels
    ETrF_bare : float
        fraction of ETr for bare soil
    adjust_rainfall : None or tuple
        if tuple (rainfall_60, ETr_60) indicates that hot temperature
        will be adjusted based on previous 60 day cummulated rainfall and ET
    Returns
    -------
    cold_pixel : int or tuple
    
    hot_pixel : int or tuple

    ETrF_cold : float    
    
    ETrF_hot : float
    
    References
    ----------
    .. [Allen2017] Allen, Richard G., Boyd Burnett, William Kramber, Justin Huntington, 
        Jeppe Kjaersgaard, Ayse Kilic, Carlos Kelly, and Ricardo Trezza, 2013. 
        Automated Calibration of the METRIC Landsat Evapotranspiration Process. 
        Journal of the American Water Resources Association (JAWRA) .49(3):563–576
        https://doi.org/10.1111/jawr.12056
    '''
#==============================================================================
#     # Cold pixel
#==============================================================================
    # Step 1. Find the 5% top NDVI pixels
    ndvi_top = np.percentile(ndvi_array, 95)
    ndvi_index = ndvi_array >= ndvi_top

    # Step 2. Identify the coldest 20% LST pixels from ndvi_index and compute their LST and NDVI mean value
    lst_low = np.percentile(lst_array[ndvi_index], 20)
    lst_index = lst_array <= lst_low
    lst_cold = np.mean(lst_array[lst_index])
    ndvi_1 = np.mean(ndvi_array[lst_index])
    
    # Step 3. Cold pixel candidates are within 0.2K from lst_cold 
    #and albedo within 0.02% of albedo_thres
    beta = (90.0 - sza_array) # Solar elevation angle
    albedo_thres = 0.001343 * beta + 0.3281 * np.exp(-0.0188 * beta) # Eq. 7 in [Allen2017]_
    cold_pixel = np.logical_and.reduce((lst_index, 
                                       np.abs(lst_array - lst_cold) <= 0.2, 
                                       np.abs(albedo_array - albedo_thres) <= 0.02))
    
    # Step 5. From step 3 select the most homogeneous pixel based on its temperature
    cold_pixel = np.logical_and(cold_pixel,
                                cv_lst == np.amin(cv_lst[cold_pixel]))
    
    cold_pixel = tuple(np.argwhere(cold_pixel)[0])
    print('Cold  pixel found with %s K and %s VI'%(float(lst_array[cold_pixel]),
                                                   float(vi_array[cold_pixel])))

    
    # Step 6. ETrF for the cold candidate
    if ndvi_1 < NDVI_FULL:
        ETrF_cold = 1.54 * ndvi_1 - 0.1 # Eq. 4a in [Allen2017]..
    else:
        ETrF_cold = 1.05 # Eq. 4b in [Allen2017]..
    
#==============================================================================
#     # Cold pixel
#==============================================================================
    # Step 1. Find the 10% lowest NDVI    
    ndvi_low = np.percentile(ndvi_array, 10)
    ndvi_index = ndvi_array <= ndvi_low
    
    # Step 2. Identify the hotest 20% LST pixels from ndvi_index and compute their LST and NDVI mean value
    lst_high = np.percentile(lst_array[ndvi_index], 80)
    lst_index = lst_array >= lst_high
    lst_hot = np.mean(lst_array[lst_index])
    
    if not isinstance(adjust_rainfall, bool):
        # Step 3. Adjust the average temperature based on 60 day rainfall and ETr
        lst_hot -= 2.6 - 13.0 * adjust_rainfall[0]/adjust_rainfall[1] # Eq. 8 in [Allen2017]..
    
    # Step 4. Hot pixel candidates are within 0.2K from lst_hot and has homogeneous NDVI
    hot_pixel = np.logical_and(lst_index, 
                               np.abs(lst_array - lst_hot) <= 0.2)
    
    hot_pixel = np.logical_and(hot_pixel,
                               cv_ndvi == np.amin(cv_ndvi[hot_pixel]))

    hot_pixel = tuple(np.argwhere(hot_pixel)[0])
    print('Hot  pixel found with %s K and %s VI'%(float(lst_array[hot_pixel]),
                                                   float(vi_array[hot_pixel])))
    
    # Step 5. ETrF for the hot candidate
    f_c = (ndvi_array[hot_pixel] - NDVI_SOIL) / (NDVI_FULL - NDVI_SOIL)
    ETrF_hot = f_c * ETrF_cold + (1.0 - f_c) * ETrF_bare
    
    return cold_pixel, hot_pixel, ETrF_cold, ETrF_hot


def esa(vi_array,
          lst_array,
          cv_vi,
          std_lst,
          cv_albedo):
    ''' Finds the hot and cold pixel using the 
    Exhaustive Search Algorithm
    
    Parameters
    ----------
    vi_array : numpy array
        Vegetation Index array (-)
    lst_array : numpy array
        Land Surface Temperature array (Kelvin)
    cv_ndvi : numpy array
        Coefficient of variation of Vegetation Index as homogeneity measurement
        from neighboring pixels
    std_lst : numpy array
        Standard deviation of LST as homogeneity measurement
        from neighboring pixels
    cv_albedo : numpy array
        Coefficient of variation of albdeo as homogeneity measurement
        from neighboring pixels

    Returns
    -------
    cold_pixel : int or tuple
    
    hot_pixel : int or tuple

    ETrF_cold : float    
    
    ETrF_hot : float
    
    References
    ----------
    .. [Bhattarai2017] Nishan Bhattarai, Lindi J. Quackenbush, Jungho Im, 
        Stephen B. Shaw, 2017.
        A new optimized algorithm for automating endmember pixel selection 
        in the SEBAL and METRIC models.
        Remote Sensing of Environment, Volume 196, Pages 178-192,
        https://doi.org/10.1016/j.rse.2017.05.009.
    '''

    # Step 1. Find homogeneous pixels
    print('Filtering pixels by homgeneity')
    homogeneous = np.logical_and.reduce((cv_vi <= 0.25,
                                         cv_albedo <= 0.25,
                                         std_lst < 1.5))
    
    print('Found %s homogeneous pixels'%np.sum(homogeneous))
    # Step 2 Filter outliers by Building ndvi and lst histograms
    lst_min, lst_max, vi_min, vi_max = histogram_fiter(vi_array, lst_array)    

    print('Removing outliers by histogram')
    mask = np.logical_and.reduce((homogeneous,
                                      lst_array >= lst_min,
                                      lst_array <= lst_max,
                                      vi_array >= vi_min,
                                      vi_array <= vi_max))
    
    print('Keep %s pixels after outlier removal'%np.sum(mask))

    # Step 3. Interative search of cold pixel
    print('Iterative search of candidate cold pixels')
    cold_pixels = incremental_search(vi_array, lst_array, mask, is_cold = True)
    print('Found %s candidate cold pixels'%np.sum(cold_pixels))


    print('Iterative search of candidate hot pixels')
    hot_pixels = incremental_search(vi_array, lst_array, mask, is_cold = False)            
    print('Found %s candidate hot pixels'%np.sum(hot_pixels))


    # Step 4. Rank the pixel candidates
    print('Ranking candidate anchor pixels')
    lst_rank = rank_array(lst_array)
    vi_rank = rank_array(vi_array)
    rank = vi_rank - lst_rank
    cold_pixel = np.logical_and(cold_pixels, rank == np.max(rank[cold_pixels]))

    cold_pixel = tuple(np.argwhere(cold_pixel)[0])
    print('Cold  pixel found with %s K and %s VI'%(float(lst_array[cold_pixel]), 
                                                   float(vi_array[cold_pixel])))
    
    
    rank = lst_rank - vi_rank
    hot_pixel = np.logical_and(hot_pixels, rank == np.max(rank[hot_pixels]))

    hot_pixel = tuple(np.argwhere(hot_pixel)[0])
    print('Hot  pixel found with %s K and %s VI'%(float(lst_array[hot_pixel]), 
                                                   float(vi_array[hot_pixel])))
    
    return cold_pixel, hot_pixel


def histogram_fiter(vi_array, lst_array):
    cold_bin_pixels = 0
    hot_bin_pixels = 0
    bare_bin_pixels = 0
    full_bin_pixels = 0
    
    while (cold_bin_pixels < 50 
            or hot_bin_pixels <50 
            or bare_bin_pixels < 50
            or full_bin_pixels < 50):

        max_lst = np.amax(lst_array)
        min_lst = np.amin(lst_array)
        max_vi = np.amax(vi_array)
        min_vi = np.amin(vi_array)
        
        print('Setting LST boundaries %s - %s'%(min_lst, max_lst))
        n_bins = int(np.ceil((max_lst - min_lst) / 0.25))
        lst_hist, lst_edges = np.histogram(lst_array, n_bins)
        
        print('Setting VI boundaries %s - %s'%(min_vi, max_vi))
        n_bins = int(np.ceil((max_vi - min_vi) / 0.01))
        vi_hist, vi_edges = np.histogram(vi_array, n_bins)

        # Get number of elements in the minimum and maximum bin
        cold_bin_pixels = lst_hist[0]
        hot_bin_pixels = lst_hist[-1]
        bare_bin_pixels = vi_hist[0]
        full_bin_pixels = vi_hist[-1]

        # Remove possible outliers
        if cold_bin_pixels < 50:
            lst_array = lst_array[lst_array >= lst_edges[1]]     

        if hot_bin_pixels < 50:
            lst_array = lst_array[lst_array <= lst_edges[-2]]     

        if bare_bin_pixels < 50:
            vi_array = vi_array[vi_array >= vi_edges[1]]     

        if full_bin_pixels < 50:
            vi_array = vi_array[vi_array <= vi_edges[-2]]     

    return lst_edges[0], lst_edges[-1], vi_edges[0], vi_edges[-1]

def rank_array(array):
    
    temp = array.argsort(axis = None)    
    ranks = np.arange(np.size(array))[temp.argsort()].reshape(array.shape)
    
    return ranks
    
def incremental_search(vi_array, lst_array, mask, is_cold = True):
    step = 0
    if is_cold:
        while True:
            
            for n_lst in range(1, 11 + step):
                for n_vi in range(1, 11 + step):
                    print('Searching cold pixels from the %s %% minimum LST and %s %% maximum VI'%(n_lst, n_vi))
                    vi_high = np.percentile(vi_array[mask], 100 - n_vi)
                    lst_cold = np.percentile(lst_array[mask], n_lst)
                    cold_index = np.logical_and.reduce((mask,
                                                         vi_array >= vi_high,
                                                         lst_array <= lst_cold))
                     
                    if np.sum(cold_index) >= 10:
                        return cold_index
            
            # If we reach here is because not enought pixels were found
            # Incresa the range of percentiles
            step += 5
    else:
        while True:
            for n_lst in range(1,11):
                for n_vi in range(1,11):
                    print('Searching hot pixels the %s %% maximum LST and %s %% minimum VI'%(n_lst, n_vi))
                    vi_low = np.percentile(vi_array[mask], n_vi)
                    lst_hot = np.percentile(lst_array[mask], 100 - n_lst)
                    hot_index = np.logical_and.reduce((mask,
                                                        vi_array <= vi_low,
                                                        lst_array >= lst_hot))
                     
                    if np.sum(hot_index) >= 10:
                        return hot_index
            # If we reach here is because not enought pixels were found
            # Incresa the range of percentiles
            step += 5

size = (1000, 1000)
lst_array = np.ones(size) + 273 + 5* np.random.rand(size[0],size[1])
lst_array[100,100] = 1e5
vi_array = np.random.rand(size[0],size[1])
albedo_array = np.ones(size) *0.23 + 2*np.random.normal(0, 0.1, size = size)


cold_pixel, hot_pixel = esa(vi_array,
                              lst_array,
                              np.zeros(vi_array.shape),
                              np.zeros(vi_array.shape),
                              np.zeros(vi_array.shape))

cold_pixel, hot_pixel, ETrF_cold, ETrF_hot = cimec(vi_array,
                                                  lst_array,
                                                  albedo_array,
                                                  np.ones(size) + 57,
                                                  np.zeros(size),
                                                  np.zeros(size),
                                                  ETrF_bare = 0,
                                                  adjust_rainfall = False)

cold_pixel, hot_pixel = maxmin_temperature(vi_array, lst_array)