import pdb
import sqlite3
import sys
import slidingwindow
from enum import Enum
from subprocess import check_output

SMOOTH_SCALE = 51
DB_FILE = 'myo_access.db'
TMP_FILE = 'myo_access.tmp'

#############################################
# BEGIN -- DB MANIPULATIONS
#############################################

CREATE_TABLE_USERS_QUERY = '''
CREATE TABLE users (
id integer PRIMARY KEY AUTOINCREMENT,
email text NOT NULL UNIQUE,
name text NOT NULL,
model text
);'''

def register( email, name, model=None ):

    db_conn = sqlite3.connect( DB_FILE )
    db = db_conn.cursor()

    query = '''
    INSERT INTO users (email, name, model)
    VALUES ( :email, :name, :model );
    '''
    param = { 'email': email,
              'name': name,
              'model': model }

    db.execute( query, param )
    db.connection.commit()
    print db.fetchall()

    db.close()
    db_conn.close()


    
def check_user( email, name=None ):

    db_conn = sqlite3.connect( DB_FILE )
    db = db_conn.cursor()

    if ( name ):
        query = '''
        SELECT * FROM users
        WHERE email = :email AND name = :name
        '''
        param = { 'email': email,
                  'name': name }
    else:
        query = '''
        SELECT * FROM users
        WHERE email = :email
        '''
        param = { 'email': email }
                
    r = db.execute( query, param ).fetchall()

    db.close()
    db_conn.close()

    print r
    return r
    

def store_model( email, model ):

    db_conn = sqlite3.connect( DB_FILE )
    db = db_conn.cursor()

    query = '''
    UPDATE users
    SET model = :model
    WHERE email = :email
    '''
    param = { 'model': model,
              'email': email }

    db.execute( query, param )
    db.connection.commit()

    db.close()
    db_conn.close()

    
def get_model( email, name=None ):

    db_conn = sqlite3.connect( DB_FILE )
    db = db_conn.cursor()

    if ( name ):
        query = '''
        SELECT model FROM users
        WHERE email = :email AND name = :name
        '''
        param = { 'email': email,
                  'name': name }
    else:
        query = '''
        SELECT model FROM users
        WHERE email = :email
        '''
        param = { 'email': email }

    r = db.execute( query, param ).fetchall()

    db.close()
    db_conn.close()
    if ( r[0] ):
        return r[0][0].replace( '\r\n', '\n' )
    else:
        return list()



def verify_users_table():

    db_conn = sqlite3.connect( DB_FILE )
    db = db_conn.cursor()

    query = '''
    SELECT * 
    FROM sqlite_master
    WHERE type='table' AND name='users'
    LIMIT 1;
    '''
    r = db.execute( query ).fetchall()

    if ( not r ):
        db.execute( CREATE_TABLE_USERS_QUERY )
        db.connection.commit()

    db.close()
    db_conn.close()

#############################################
# END -- DB MANIPULATIONS
#############################################

#############################################
# BEGIN -- EMG RECORDING/TRANSFORMS
#############################################

def record_emg( seconds, email=None ):
    # Call EMG Sampler, and capture output
    emg_series = check_output( ['Apps/x64/Release/emg-data-sample.exe', str(seconds)] )

    # Get rid of dumb Windows carriage returns
    emg_series = emg_series.replace( '\r\n', '\n' )
    pdb.set_trace()
    # If an email is specified, store it in the database with the specified user
    if ( email ):
        store_model( email, emg_series )
    # If no email is specified, store it in a temporary file 
    else:
        with open( TMP_FILE, 'w' ) as fp:
            fp.write( emg_series )


def compare_emg( email ):
    # Fetch the model from the database
    model = get_model( email )

    # Fetch series from temporary file
    series = None
    with open( TMP_FILE, 'r' ) as fp:
        series = fp.read()

    # Perform comparison between model and series
    model = [[int(x) for x in row.split(',')] for row in model.split('\n' ) if row]
    series = [[int(x) for x in row.split(',')] for row in series.split('\n' ) if row]
    ds = slidingwindow.getSeriesDissimilarity( model, series, SMOOTH_SCALE )
    pdb.set_trace()
    # Return True/False according to matching analysis
    return True
    
#############################################
# END -- EMG RECORDING/TRANSFORMS
#############################################

def myo_hook( argv ):
    if ( len( argv ) < 2 ):
        print( "Too Few Agruments." )
        printUsage()
        exit(1);

    # Verify that the users table is set up
    verify_users_table()
    
    # Check that requested command is valid
    try:
        cmd = api(argv[1])
    except ValueError as e:
        cmd = None
        
    # Collect parameters from command line
    params = list()
    if ( len( argv ) > 2 ):
        params = argv[2:]
    param_len = len( params )

    if ( cmd == api.REGISTER ):
        if ( (param_len < 2) or (param_len > 3) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.REGISTER]( *params )
        pass
    elif ( cmd == api.CHECK_USER ):
        if ( (param_len < 1) or ( param_len > 2 ) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.CHECK_USER]( *params )
        pass
    elif ( cmd == api.STORE_MODEL ):
        if ( param_len != 2 ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.STORE_MODEL]( *params )
        pass
    elif ( cmd == api.GET_MODEL ):
        if ( (param_len < 1) or ( param_len > 2 ) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.GET_MODEL]( *params )
        pass
    elif ( cmd == api.RECORD ):
        if ( (param_len < 1) or ( param_len > 3 ) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.RECORD]( *params )
        pass
    elif ( cmd == api.COMPARE ):
        if ( (param_len != 1) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.COMPARE]( *params )
        pass
    else:
        printUsage()
        exit(1)
        
    
def main( argv ):
    myo_hook( argv )


def printUsage( which=None ):
    register_usage =	"\t\t\tregister\t<email> <name> [<model_file>]"
    check_user_usage =	"\t\t\tcheck_user\t<email> [<name>]"
    store_model_usage =	"\t\t\tstore_model\t<email> <model_file>"
    get_model_usage =	"\t\t\tget_model\t<email> [<name>]"
    record_usage =	"\t\t\trecord\t<interval in seconds> [<email>]"
    compare_usage =	"\t\t\tcompare\t<email>"

    usage_dict = {
        api.REGISTER	: register_usage,
        api.CHECK_USER	: check_user_usage,
        api.STORE_MODEL	: store_model_usage,
        api.GET_MODEL	: get_model_usage,
        api.RECORD	: record_usage,
        api.COMPARE	: compare_usage
    }
    
    print( "python {} <access_cmd> [<access_param1>[, ...] ]\n".format( __file__ ) )
    if ( not which ):
        for k in usage_dict:
            print( usage_dict[k] )
    else:
        print ( usage_dict[which] )



class api(Enum):
    REGISTER='register' 
    CHECK_USER='check_user'
    STORE_MODEL='store_model'
    GET_MODEL='get_model'
    RECORD='record'
    COMPARE='compare'
    

api_funcs = {
    api.REGISTER:register,
    api.CHECK_USER:check_user,
    api.STORE_MODEL:store_model,
    api.GET_MODEL:get_model,
    api.RECORD:record_emg,
    api.COMPARE:compare_emg
}

    
if __name__ == "__main__":
    main( sys.argv )
