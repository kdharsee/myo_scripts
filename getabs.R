#!/usr/bin/Rscript

args <- commandArgs( );
print( paste( "Reading from: ",  args[1] ) )

filelist = list.files( args[1], pattern="[[:print:]]*.csv", recursive=TRUE, full.names=TRUE )

for ( currentfile in filelist ) {

    print( paste( "Current File: ", currentfile ) )

    emgdata <- read.csv( file=currentfile, head=FALSE, sep="," )
    emgdata <- abs( emgdata )

    fname = strsplit( currentfile, ".csv" )[1];
    
    absfile = paste( fname, "_abs.csv", sep="" )

    print( paste( "Writing absolute values to:", absfile ) )

    write.table( emgdata, sep=",", file=absfile, col.names=FALSE, row.names=FALSE )

    readline( "Press ENTER to continue.." )
}

