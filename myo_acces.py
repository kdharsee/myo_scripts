import sys

api = enum( REGISTER:'register' 
            CHECK_USER:'check_user'
            STORE_MODEL:'store_model'
            GET_MODEL:'get_model' )

def register( email, name, model_file ):
    pass
def check_user( email, name=None ):
    pass
def store_model( model_file ):
    pass
def get_model( email, name=None ):
    pass



def printUsage():
    print( "python {} <access_cmd> [<access_param1>, [...] ]".format( __file__ ) )

def main( argv ):

    if ( len( argv ) < 2 ):
        print( "Too Few Agruments." )
        printUsage()
        exit(1);

    cmd = argv[1]
    params = list()
    if ( len( argv ) > 2 ):
        params = argv[2:]

    switch ( cmd ):
        case:


    pass

if __name__ == "__main__":
    main( sys.argv )
