"""
SCRIPT NAME:
    Cycle_inf_risk.py

DESCRIPTION:
    Predicts the percieved risk of cycle infrastructure based on user profile and scenario.

FUNCTIONS:
    Functions used to turn on/off sections of code:
        load_data()
        flatten_profile()
        flatten_ratings()
        combine_data()
        clean_data()

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

#%% Clean data
def clean_data():
    global clean_data
    
    flattened_data = pd.read_csv('flattened_data.csv')
    
    # Filter for columns of interest
    clean_data = flattened_data[['profile_gender', 'profile_zipcode', 'profile_ageGroup',
                                'profile_bicycleUse', 'profile_hasChildren', 'profile_perspective',
                                'profile_transportRatings_car', 'profile_transportRatings_public',
                                'profile_transportRatings_bicycle', 'profile_transportRatings_motorbike',
                                'profile_transportRatings_pedestrian', 'profile_transportRatings_safe',
                                'profile_transportRatings_faster', 'profile_transportRatings_bikefun',
                                'profile_transportRatings_weather', 'ratings_scene_id', 'ratings_rating']]
    
    # Convert non-numerical values to numerical
    # 'profile_gender': m = 1, w = 2, d = 3, nan = -9999
    print(clean_data.profile_gender.unique())
    clean_data.loc[clean_data['profile_gender'] == 'm', 'profile_gender'] = 1
    clean_data.loc[clean_data['profile_gender'] == 'w', 'profile_gender'] = 2
    clean_data.loc[clean_data['profile_gender'] == 'd', 'profile_gender'] = 3
    clean_data.loc[clean_data['profile_gender'].isnull(), 'profile_gender'] = -9999
    
    # 'profile_zipcode': nan = -9999
    print(clean_data.profile_zipcode.unique())
    clean_data.loc[clean_data['profile_zipcode'].isnull(), 'profile_zipcode'] = -9999
    
    # 'profile_ageGroup': nan = -9999
    print(clean_data.profile_ageGroup.unique())
    clean_data.loc[clean_data['profile_ageGroup'].isnull(), 'profile_ageGroup'] = -9999
    
    # 'profile_bicycleUse': nan = -9999
    print(clean_data.profile_bicycleUse.unique())
    clean_data.loc[clean_data['profile_bicycleUse'].isnull(), 'profile_bicycleUse'] = -9999
    
    # 'profile_hasChildren': True = 1, False = 0, nan = -9999
    print(clean_data.profile_hasChildren.unique())
    clean_data.loc[clean_data['profile_hasChildren'] == True, 'profile_hasChildren'] = 1
    clean_data.loc[clean_data['profile_hasChildren'] == False, 'profile_hasChildren'] = 0
    clean_data.loc[clean_data['profile_hasChildren'].isnull(), 'profile_hasChildren'] = -9999
    
    # 'profile_perspective': C = 1, A = 2, P = 3
    print(clean_data.profile_perspective.unique())
    clean_data.loc[clean_data['profile_perspective'] == 'C', 'profile_perspective'] = 1
    clean_data.loc[clean_data['profile_perspective'] == 'A', 'profile_perspective'] = 2
    clean_data.loc[clean_data['profile_perspective'] == 'P', 'profile_perspective'] = 3

    # 'profile_transportRatings_car':
    print(clean_data.profile_transportRatings_car.unique())
    
    # 'profile_transportRatings_public':
    print(clean_data.profile_transportRatings_public.unique())  
    
    # 'profile_transportRatings_bicycle':
    print(clean_data.profile_transportRatings_bicycle.unique())  
    
    # 'profile_transportRatings_motorbike':
    print(clean_data.profile_transportRatings_motorbike.unique())  
    
    # 'profile_transportRatings_pedestrian':
    print(clean_data.profile_transportRatings_pedestrian.unique())  
   
    # 'profile_transportRatings_safe': nan = -9999
    print(clean_data.profile_transportRatings_safe.unique())
    clean_data.loc[clean_data['profile_transportRatings_safe'].isnull(), 'profile_transportRatings_safe'] = -9999
    
    # 'profile_transportRatings_faster': nan = -9999
    print(clean_data.profile_transportRatings_faster.unique())
    clean_data.loc[clean_data['profile_transportRatings_faster'].isnull(), 'profile_transportRatings_faster'] = -9999
    
    # 'profile_transportRatings_bikefun': nan = -9999
    print(clean_data.profile_transportRatings_bikefun.unique()) 
    clean_data.loc[clean_data['profile_transportRatings_bikefun'].isnull(), 'profile_transportRatings_bikefun'] = -9999

    # 'profile_transportRatings_weather': nan = -9999
    print(clean_data.profile_transportRatings_weather.unique())
    clean_data.loc[clean_data['profile_transportRatings_weather'].isnull(), 'profile_transportRatings_weather'] = -9999   
    
    # 'ratings_scene_id': nan = -9999
    print(clean_data.ratings_scene_id.unique())
    scene_ids = list(clean_data.ratings_scene_id.unique())
    
    a = 1
    for i in scene_ids:
        clean_data.loc[clean_data['ratings_scene_id'] == i, 'ratings_scene_id'] = a
        a = a + 1

    clean_data.loc[clean_data['ratings_scene_id'].isnull(), 'ratings_scene_id'] = -9999   
    
    clean_data.to_csv('clean_data.csv')
    
#clean_data()

#%% Split data sets
clean_data = pd.read_csv('clean_data.csv')
clean_data.drop(['Unnamed: 0'], axis = 1, inplace = True)


#%% Train model(s) and export

#%% Make predictions and assess model accuracy

