import configparser

env = configparser.ConfigParser()
env.read('env.ini')

with open(env['data']['drop_wards'], 'r') as fp:
    drop_wards = fp.read().split('\n') 

drop_ward_codes = [ward.split(' - ')[0] for ward in drop_wards]