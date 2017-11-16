#!/usr/bin/python27

import sys
import numpy as np
import csv
import code
import glob
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import factorial
from inspect import currentframe

SMOOTH_WINDOW_SIZE = int( sys.argv[3] )
MEASURES_PER_SECOND = 20
WINDOW_DURATION = 4 # in seconds
WINDOW_SIZE = WINDOW_DURATION*MEASURES_PER_SECOND
SENSOR_COUNT = 8
ACTIVE_SENSORS = 5

def savitzky_golay( y, window_size, order, deriv=0, rate=1 ):

    try:
        window_size = np.abs( np.int( window_size ) )
        order = np.abs( np.int( order ) )
    except ValueError, msg:
        raise ValueError( "window_size and order have to be of type int" )

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError( "window_size size must be a positive odd number" )

    if window_size < order + 2:
        raise TypeError( "window_size is too small for the polynomials order" )

    order_range = range( order + 1 )
    half_window = (window_size - 1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def getData( pattern ):

    with open( glob.glob( pattern )[0], 'r' ) as fp:
        data = [[int(x) for x in row] for row in list( csv.reader( fp, delimiter=',' ) )]

    return data


''' getInactiveSensors will return the indexes of <count> sensors found in data having the lowest average magnitude. ''' 
def getInactiveSensors( data, count ):

    # Cannot return more than what's given
    if ( count > len( data[0] ) ):
        raise ValueError( '[-] Requested more sensors than those passed' )

    np_data = np.array( data )

    # avg_sensor_data will hold the average measured EMG magnitude for each sensor
    avg_sensor_data = [np.mean(np_data[:,x]) for x in range( np_data.shape[1] )]

    avgs = sorted( zip( avg_sensor_data, range( len( avg_sensor_data ) ) ), reverse=False )[0:count]
    idx_avgs = [tup[1] for tup in avgs]

    return idx_avgs
    

def getModelTrim( model, size ):

    # Cannot proceed if the sliding window is smaller than the given model
    if ( len(model) < size ):
        raise ValueError( '[-] Model length shorter than sliding window' )

    np_model = np.array( model )

    # variance_list will hold the variance of a subset of points defined by the input param, size, for each point in the model series
    variance_list = [np.var( np_model[x:x+size] ) for x in range( len(model)-size )]
    max_var = max( variance_list )
    max_pts = [i for i in range( len(variance_list) ) if ( variance_list[i] == max_var )]

    window = np_model[max_pts[0]:max_pts[0]+size,:]

    print "Model trimmed to {}:{} of original".format( max_pts[0], max_pts[0]+size )
    
    return window


def createGaussianKernel( buckets ):

    x = np.linspace( -3, 3, buckets )
    kernel = mlab.normpdf( x, 0, 1 )
    return kernel


def combine( data ):
    '''Combines the set of data to a single value using a particular function'''
    
    d = np.array( data )

    # Take the average of each column
    #result = [np.mean( d[:,i] ) for i in range( d.shape[1] ) ]

    # Sum after multiplying across a normal distirbution
    kernel = createGaussianKernel( len( data ) )

    #result = [ sum( [kernel[i]*data[i][j] for i in range( len( kernel ) )] ) for j in range( len( data[0] ) )]

    result = sum( [ kernel[i]*data[i] for i in range( len( kernel ) )] )
    return result


def getPairwiseCombinations( data ):

    pair_combinations = np.empty_like( data )

    for i in range( data.shape[1]-1 ):
        for j in range( i+1, data.shape[1] ):
            col = np.transpose( np.array( [np.abs( data[:,i] - data[:,j] )] ) )

            pair_combinations = np.append( pair_combinations, col, 1 )

    # Trim combination matrix to only include new pair columns
    pair_combinations = np.delete( pair_combinations, range( data.shape[1] ), 1 )

    return pair_combinations
    

'''getWindowDissimilarity computes the dissimilarity between the provided arguments. Arguments are epected to have the same dimensions'''
def getWindowDissimilarity( model, series_window ):

    if ( model.shape != series_window.shape ):
        raise ValueError( 'Model and Window of Series should have the same dimensions' )

    diffs = list()

    # Compute differences between each point in the model and each corresponding point in the current window
    for i in range( model.shape[0] ):
        slice_diffs = np.array( [abs(x) for x in (model[i,:]-series_window[i,:])][:ACTIVE_SENSORS] )
        diffs.append( np.average( slice_diffs ) )

    window_dissimilarity = combine( diffs )

    return window_dissimilarity


''' getSeriesDissimilarity will return a series of dissimilarities for each point along the series. 
Dissimilarities for each point in time will be measured as the difference between a trimmed version ofhe model, and a sliding window. '''
def getSeriesDissimilarity( model, series ):

    # Store indeces of only the sensors which give us valuable measurements. Discard the rest
    inactive_sensors_idx = getInactiveSensors( model, SENSOR_COUNT-ACTIVE_SENSORS )

    model = getModelTrim( model, WINDOW_SIZE )

    # for i in range( model.shape[1] ):
    #     plt.figure(0)
    #     plt.plot( range( len( model ) ), model[:,i], 'k-' )
    
    # plt.show()

    active_model = np.delete( np.array( model ), inactive_sensors_idx, 1 )
    active_series = np.delete( np.array( series ), inactive_sensors_idx, 1 )

    # Pre-process the model and series to have additional columns holding all permutations of pairwise sensor differences 
    # For each column, compare it to every other column for all possible pairwise combinations of columns
    model_pair_combinations = np.empty_like( active_model )
    series_pair_combinations = np.empty_like( active_series )

    for i in range( active_model.shape[1]-1 ):
        for j in range( i+1, active_model.shape[1] ):
            model_col = np.transpose( np.array( [np.abs( active_model[:,i] - active_model[:,j] )] ) )
            series_col = np.transpose( np.array( [np.abs( active_series[:,i] - active_series[:,j] )] ) )

            model_pair_combinations = np.append( model_pair_combinations, model_col, 1 )
            series_pair_combinations = np.append( series_pair_combinations, series_col, 1 )
            #active_model = np.append( active_model, model_col, 1 )
            #active_series = np.append( active_series, series_col, 1 )

    # Trim combination matrix to only include new pair columns
    model_pair_combinations = np.delete( model_pair_combinations, range( active_model.shape[1] ), 1 )
    series_pair_combinations = np.delete( series_pair_combinations, range( active_model.shape[1] ), 1 )

    # Append new columns to model and data
    active_model = np.append( active_model, getPairwiseCombinations( active_model ), 1 )
    active_series = np.append( active_series, getPairwiseCombinations( active_series ), 1 )

    # Normalize each sensor's series
    #norm_model = [[float(elem)/np.sum(active_model[:,i]) for elem in active_model[:,i]] for i in range( active_model.shape[1] )]
    #norm_series = [[float(elem)/np.sum(active_series[:,i]) for elem in active_series[:,i]] for i in range( active_series.shape[1] )]

    norm_model = active_model
    norm_series = active_series
    norm_model = np.transpose( np.array( [savitzky_golay( np.array(norm_model[:,i]), SMOOTH_WINDOW_SIZE, 3 ) for i in range(norm_model.shape[1])] ) )
    norm_series = np.transpose( np.array( [savitzky_golay( np.array(norm_series[:,i]), SMOOTH_WINDOW_SIZE, 3 ) for i in range(norm_series.shape[1])] ) )

    dissimilarities = list()

    for i in range( len(series)-WINDOW_SIZE ):
        window_dissim = getWindowDissimilarity( norm_model, norm_series[i:i+WINDOW_SIZE,:] )
        dissimilarities.append( window_dissim )
        
    count = len( dissimilarities )
    time_space = range( count )

    # for i in range( len( norm_series[0] ) ):
    #     plt.plot( time_space, np.array(norm_series)[:count,i].tolist(), 'b-' )

    # plt.twinx()

    #for i in range( len( dissimilarities[0] ) ):
    #    plt.plot( time_space, np.array(dissimilarities)[:,i], 'gx' )

    # for i in range( 5 ):
    #     plt.plot( time_space, np.array( norm_series )[:count,i].tolist(), 'b-' )
    # for i in range( 5, norm_series.shape[1] ):
    #     plt.plot( time_space, np.array( norm_series )[:count,i].tolist(), 'b.' )

    # plt.twinx()

    # for i in range( 5 ):
    #     plt.plot( time_space, np.array( dissimilarities )[:,i], 'r-' )
    # for i in range( 5, len( dissimilarities[0] ) ):
    #     plt.plot( time_space, np.array( dissimilarities )[:,i], 'rx' )    

    [plt.plot( time_space, np.array( norm_series )[:count,i].tolist(), 'k-' ) for i in range( norm_series.shape[1] )]
    plt.twinx()
    plt.plot( time_space, dissimilarities, 'r.' )
    

    plt.show()

    #code.interact( local=locals(), banner="Final shell" )

def main( args ):

    model = getData( args[1] )
    series = getData( args[2] )

    getSeriesDissimilarity( model, series )


if __name__ == "__main__":
    main( sys.argv )
