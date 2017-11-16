import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import pdb

WINDOW_SIZE = int(sys.argv[3])
WINDOW_START = 0

def main( args ):

    file1 = args[1]
    file2 = args[2]
    emg_data_model = list()
    emg_data_series = list()
    
    # Get Data from the two files 
    with open( file1, 'r' ) as fp1:
        with open( file2, 'r' ) as fp2:
            content1 = csv.reader( fp1, delimiter=',' );
            content2 = csv.reader( fp2, delimiter=',' );
            for row in content1:
                emg_data_model.append( [abs(int(x)) for x in row] )
            for row in content2:
                emg_data_series.append( [abs(int(x)) for x in row] )


    # Aggregate all sensor values to a single series by summing
    emg_data_model = map( sum, emg_data_model )
    emg_data_series = map( sum, emg_data_series )

    # Get a window from the model
    window_model = emg_data_model[WINDOW_START:WINDOW_START+WINDOW_SIZE]
    
    # Run the the model window over the series
    c = list()
    num_rows = len( emg_data_series )
    for row_num in range( num_rows ):
        # end should be no farther than the end of the list
        end = (WINDOW_SIZE+row_num) if (WINDOW_SIZE+row_num) <= num_rows else num_rows
        window_series = emg_data_series[row_num:end]
        # Fill in 0s when data comes close to boundary
        dummy_chunk_len = WINDOW_SIZE - (end-row_num)
        if ( dummy_chunk_len > 0 ):
            dummy_chunk = [0] * dummy_chunk_len
            window_series += dummy_chunk
        try:
            cor = np.corrcoef([window_model, window_series])[0, 1]
        except Exception as e:
            pdb.set_trace()
        c.append( cor )

    fix, ax1 = plt.subplots()
    ax1.plot(emg_data_model, 'go-', label='model', linewidth=2)
    ax1.plot(emg_data_series, 'ro-', label='series', linewidth=2)
    ax2 = ax1.twinx()
    ax2.plot(c, 'bx-', label='correlation')

    plt.show()

    pass
    # print '\n'.join( str(emg_data) )


if __name__ == "__main__":
    main( sys.argv )
