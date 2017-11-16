import matplotlib.pyplot as plt
import numpy
import csv
import sys
import pdb

def main( args ):

    tgt_file = args[1]
    emg_data = list()
    with open( tgt_file, 'r' ) as fp:
        content = csv.reader( fp, delimiter=',' );
        for row in content:
            print row
            emg_data.append( [int(x) for x in row] )
            


    for col_num in range( len( emg_data[0] ) ):
        plt.plot(map( abs, [x[col_num] for x in emg_data]), 'go-', label='line {}'.format(col_num), linewidth=2)
    plt.ylabel('some numbers')
    plt.show()

    pass
    # print '\n'.join( str(emg_data) )


if __name__ == "__main__":
    main( sys.argv )
