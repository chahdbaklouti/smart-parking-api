#import psycopg2

#def get_connection():
#   return psycopg2.connect(
#       host="localhost",
#       database="smart_parking",
#       user="postgres",
#       password="0000",
#       port="5432"
#   )

import psycopg2

def get_connection():
    return psycopg2.connect(
        "postgresql://neondb_owner:npg_sJcpBG1hHmu7@ep-orange-paper-aqfyrrie.c-8.us-east-1.aws.neon.tech/neondb?sslmode=require"
    )