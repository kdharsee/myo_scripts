#!/usr/bin/Rscript

args <- commandArgs( );
#print( args )
print( paste( "Reading from: ",  args[1] ) )

filelist = list.files( args[1], pattern="[[:print:]]*.csv", recursive=TRUE, full.names=TRUE )

for ( currentfile in filelist ) {

    print( paste( "Current File: ", currentfile ) )

    emgdata <- read.csv( file=currentfile, head=FALSE, sep="," )
    emgdata <- abs( emgdata )

    #fname = strsplit( currentfile, "[.]" );
    #absfile = paste( fname, "csv", sep="." )
    #write.csv( emgdata, file=absfile )
    #print( paste( "Wrote absolute files to:", absfile ) )
    # Calculate some aggregations
    emgdata$mean <-rowMeans( emgdata, dims=1 )
    emgdata$sum <- rowSums( emgdata, dims=1 )
    emgdata$max <- apply( emgdata[,1:8], 1, max )

    smoothdata <- list()

    plot( 1:dim( emgdata )[1], apply( emgdata[,1:8], 1, max ), type="n" )

    title( main=currentfile )
    
    for ( i in 1:8 ) {
        #    smooth = loess( emgdata[,i] ~ (1:(dim(emgdata)[1])) )
        xdata = 1:dim( emgdata )[1]
        ydata = emgdata[,i]
        smoothed = smooth.spline( xdata, ydata, df=50)
        lines( smoothed, col=rgb(r=(i/10),g=(i/10),b=0.2+(i/10) ) )

    }

    readline( "Press ENTER to continue.." )
}

