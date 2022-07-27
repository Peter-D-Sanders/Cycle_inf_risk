"""
SCRIPT NAME:
    MF2P.py

DESCRIPTION:
    Models a matrix of values given two variables.

FUNCTIONS:

UPDATE RECORD:
Date          Version     Author         Description
24/07/2022    v1.0        Pete Sanders   Created
        
RECOMMENDED FUTURE IMPROVEMENTS:
        
Bugs:
    
"""
# Import libraries
import pandas as pd
import numpy as np
import json
import ast

#%% Import data and combine matricies into 3D array
def load_data():
    global data
    
    data = pd.read_csv('Bikes_data.csv')
    data = json.loads(data)
    
    # convert to pandas df object
    data = pd.read_json('SurveyResults_200414.json')
    data = pd.DataFrame(data)
    
    # Save as csv (not sure tis step is nessessary but i want to look at the data.)
    data.to_csv('Bikes_data.csv')

#load_data()

#%% Flatten data
def flatten_profile():
    global data
    
    # Flatten the 'profile' column in the data df, 1 of x.
    profile_columns = list(ast.literal_eval(data.loc[0,"profile"]).keys())
    profile_columns = ['profile_' + sub for sub in profile_columns]
    profile = pd.DataFrame(data=None, index=None, columns = profile_columns)
    
    for i in data.index:
        profile_columns = list(ast.literal_eval(data.loc[i,"profile"]).keys())
        profile_columns = ['profile_' + sub for sub in profile_columns]
        profile_rows = pd.DataFrame([np.array(list(ast.literal_eval(data.loc[i,"profile"]).values()))], columns = profile_columns)
        profile = pd.concat([profile, profile_rows], join="outer", ignore_index=True)
        profile.reset_index(inplace = True, drop = True)
    
    profile.to_csv('profile.csv')
    profile = pd.read_csv('profile.csv')
    
    # Flatten the 'profile' column in the data df, 2 of x (flatten the 'berlinTraffic' column).
    berlinTraffic_columns = list(ast.literal_eval(profile.loc[0,"profile_berlinTraffic"]).keys())
    berlinTraffic_columns = ['profile_berlinTraffic_' + sub for sub in berlinTraffic_columns]
    berlinTraffic = pd.DataFrame(data=None, index=None, columns = berlinTraffic_columns)
    
    for i in profile.index:
        if str(profile["profile_berlinTraffic"][i]) == 'nan':
            berlinTraffic_columns = ["profile_berlinTraffic_noise"]
            berlinTraffic_rows = pd.DataFrame(data = ['nan'], index = None, columns = berlinTraffic_columns)
            berlinTraffic = pd.concat([berlinTraffic, berlinTraffic_rows], join="outer", ignore_index=True)
            berlinTraffic.reset_index(inplace = True, drop = True)
        else:
            berlinTraffic_columns = list(ast.literal_eval(profile.loc[i,"profile_berlinTraffic"]).keys())
            berlinTraffic_columns = ['profile_berlinTraffic_' + sub for sub in berlinTraffic_columns]
            berlinTraffic_rows = pd.DataFrame([np.array(list(ast.literal_eval(profile.loc[i,"profile_berlinTraffic"]).values()))], columns = berlinTraffic_columns)
            berlinTraffic = pd.concat([berlinTraffic, berlinTraffic_rows], join="outer", ignore_index=True)
            berlinTraffic.reset_index(inplace = True, drop = True)
    
    berlinTraffic.to_csv('berlinTraffic.csv')
    
    # Flatten the 'profile' column in the data df, 3 of x (flatten the 'transportRatings' column).
    transportRatings_columns = list(ast.literal_eval(profile.loc[0,"profile_transportRatings"]).keys())
    transportRatings_columns = ['profile_transportRatings_' + sub for sub in transportRatings_columns]
    transportRatings = pd.DataFrame(data=None, index=None, columns = transportRatings_columns)
    
    for i in profile.index:
        transportRatings_columns = list(ast.literal_eval(profile.loc[i,"profile_transportRatings"]).keys())
        transportRatings_columns = ['profile_transportRatings_' + sub for sub in transportRatings_columns]
        transportRatings_rows = pd.DataFrame([np.array(list(ast.literal_eval(profile.loc[i,"profile_transportRatings"]).values()))], columns = transportRatings_columns)
        transportRatings = pd.concat([transportRatings, transportRatings_rows], join="outer", ignore_index=True)
        transportRatings.reset_index(inplace = True, drop = True)
    
    transportRatings.to_csv('transportRatings.csv')
    
    # Flatten the 'profile' column in the data df, 4 of x (flatten the 'motivationalFactors' column).
    motivationalFactors_columns = list(ast.literal_eval(profile.loc[0,"profile_motivationalFactors"]).keys())
    motivationalFactors_columns = ['profile_transportRatings_' + sub for sub in motivationalFactors_columns]
    motivationalFactors = pd.DataFrame(data=None, index=None, columns = motivationalFactors_columns)
    
    for i in profile.index:
        motivationalFactors_columns = list(ast.literal_eval(profile.loc[i,"profile_motivationalFactors"]).keys())
        motivationalFactors_columns = ['profile_transportRatings_' + sub for sub in motivationalFactors_columns]
        motivationalFactors_rows = pd.DataFrame([np.array(list(ast.literal_eval(profile.loc[i,"profile_motivationalFactors"]).values()))], columns = motivationalFactors_columns)
        motivationalFactors = pd.concat([motivationalFactors, motivationalFactors_rows], join="outer", ignore_index=True)
        motivationalFactors.reset_index(inplace = True, drop = True)
    
    motivationalFactors.to_csv('motivationalFactors.csv')

#flatten_profile()

#%% Flatten the 'ratings' column in the data df.
def flatten_ratings():
    global data
    global ratings
    
    num_ratings = len(list(ast.literal_eval(data.loc[0,"ratings"])))
    
    ratings_columns = list(ast.literal_eval(data.loc[0,"ratings"])[0].keys())
    ratings_columns = ['ratings_' + sub for sub in ratings_columns]
    ratings_columns.insert(0,'session_id')
    ratings = pd.DataFrame(data=None, index=None, columns = ratings_columns)
    
    # need a double loop here
    for i in data.index[20768:]:
        rating = 0
        num_ratings = len(list(ast.literal_eval(data.loc[i,"ratings"])))
        
        while rating < num_ratings:
            # build a dataframe that containst the session id from the data df.
            ratings_columns = list(ast.literal_eval(data.loc[i,"ratings"])[rating].keys())
            ratings_columns = ['ratings_' + sub for sub in ratings_columns]
            ratings_columns.insert(0,'session_id')

            # and each row contains the information for a single rating.
            ratings_data = list(ast.literal_eval(data.loc[i,"ratings"])[rating].values())
            ratings_data.insert(0,data.loc[i,'session_id'])
            ratings_rows = pd.DataFrame([np.array(ratings_data)], columns = ratings_columns)

            # joint this to the ratings df.
            ratings = pd.concat([ratings, ratings_rows], join="outer", ignore_index=True)
            ratings.reset_index(inplace = True, drop = True)
            
            rating = rating + 1
        
    ratings.to_csv('ratings.csv')
    
#flatten_ratings()

# the ratings df can then be joined to the flattened_data df using an outside join,
# that should mean that each row is a uniquie rating, but the resultant df would be enormous. 

#%% Combine the 'data', 'profile', 'berlinTraffic', and 'transportRatings' dfs
def combine_data():
    profile = pd.read_csv('profile.csv')
    berlinTraffic = pd.read_csv('berlinTraffic.csv')
    transportRatings = pd.read_csv('transportRatings.csv')
    motivationalFactors = pd.read_csv('motivationalFactors.csv')
    ratings = pd.read_csv('ratings.csv')
    
    flattened_data = pd.merge(data, profile)
    flattened_data = pd.merge(flattened_data, berlinTraffic)
    flattened_data = pd.merge(flattened_data, transportRatings)
    flattened_data = pd.merge(flattened_data, motivationalFactors)
    flattened_data = flattened_data.merge(ratings, how = 'outer', on = 'session_id')
    
    flattened_data.reset_index(inplace = True, drop = True)
    
    flattened_data.drop(['Unnamed: 0', 'profile', 'profile_berlinTraffic', 'profile_transportRatings','profile_motivationalFactors','ratings'], axis = 1, inplace = True)
    
    flattened_data.to_csv('flattened_data.csv')
    
#combine_data()

#%% Split data sets
flattened_data = pd.read_csv('flattened_data.csv')

#%% Train model(s) and export

#%% Make predictions and assess model accuracy

