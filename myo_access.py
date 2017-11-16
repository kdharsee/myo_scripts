import pdb
import sqlite3
import sys
from enum import Enum

CREATE_TABLE_USERS_QUERY = '''
CREATE TABLE users (
id integer PRIMARY KEY AUTOINCREMENT,
email text NOT NULL UNIQUE,
name text NOT NULL,
model BLOB
);'''


def register( db, email, name, model=None ):

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

    
def check_user( db, email, name=None ):

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

    print r
    return r
    

def store_model( db, email, model ):

    query = '''
    UPDATE users
    SET model = :model
    WEHERE email = :email
    '''
    param = { 'model': model,
              'email': email }

    db.execute( query, param )
    db.connection.commit()

    print db.fetchall()

    
def get_model( db, email, name=None ):

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
    print r
    return r


def verify_users_table( db ):

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



def printUsage( which=None ):
    register_usage =	"\t\t\tregister\t<email> <name> [<model_file>]"
    check_user_usage =	"\t\t\tcheck_user\t<email> [<name>]"
    store_model_usage =	"\t\t\tstore_model\t<email> <model_file>"
    get_model_usage =	"\t\t\tget_model\t<email> [<name>]"

    usage_dict = {
        api.REGISTER : register_usage,
        api.CHECK_USER : check_user_usage,
        api.STORE_MODEL : store_model_usage,
        api.GET_MODEL : get_model_usage
    }
    
    print( "python {} <access_cmd> [<access_param1>, [...] ]\n".format( __file__ ) )
    if ( not which ):
        for k in usage_dict:
            print( usage_dict[k] )
    else:
        print ( usage_dict[which] )

        
    
def main( argv ):

    if ( len( argv ) < 2 ):
        print( "Too Few Agruments." )
        printUsage()
        exit(1);

    db_file = 'myo_acces.db'
        
    db_conn = sqlite3.connect( db_file )
    db = db_conn.cursor()

    verify_users_table( db )
    
    try:
        cmd = api(argv[1])
    except ValueError as e:
        cmd = None
        
    params = list()
    if ( len( argv ) > 2 ):
        params = argv[2:]

    param_len = len( params )

    if ( cmd == api.REGISTER ):
        if ( (param_len < 2) or (param_len > 3) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.REGISTER]( db, *params )
        pass
    elif ( cmd == api.CHECK_USER ):
        if ( (param_len < 1) or ( param_len > 2 ) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.CHECK_USER]( db, *params )
        pass
    elif ( cmd == api.STORE_MODEL ):
        if ( param_len != 2 ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.STORE_MODEL]( db, *params )
        pass
    elif ( cmd == api.GET_MODEL ):
        if ( (param_len < 1) or ( param_len > 2 ) ):
            printUsage( cmd )
            exit(1)
        r = api_funcs[api.GET_MODEL]( db, *params )
        pass
    else:
        printUsage()
        exit(1)


class api(Enum):
    REGISTER='register' 
    CHECK_USER='check_user'
    STORE_MODEL='store_model'
    GET_MODEL='get_model'

api_funcs = {
    api.REGISTER:register,
    api.CHECK_USER:check_user,
    api.STORE_MODEL:store_model,
    api.GET_MODEL:get_model
}

    
if __name__ == "__main__":
    main( sys.argv )
