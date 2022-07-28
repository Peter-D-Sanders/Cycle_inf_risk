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
        split_data()
        train_models()
        make_predictions()

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
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
# from sklearn import tree
import joblib

from sklearn.inspection import permutation_importance

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
    # 'profile_gender': m = 1, w = 2, d = 3, nan = -99
    print(clean_data.profile_gender.unique())
    clean_data.loc[clean_data['profile_gender'] == 'm', 'profile_gender'] = 1
    clean_data.loc[clean_data['profile_gender'] == 'w', 'profile_gender'] = 2
    clean_data.loc[clean_data['profile_gender'] == 'd', 'profile_gender'] = 3
    clean_data.loc[clean_data['profile_gender'].isnull(), 'profile_gender'] = -99
    
    # 'profile_zipcode': nan = -99
    print(clean_data.profile_zipcode.unique())
    clean_data.loc[clean_data['profile_zipcode'].isnull(), 'profile_zipcode'] = -99
    
    # 'profile_ageGroup': nan = -99
    print(clean_data.profile_ageGroup.unique())
    clean_data.loc[clean_data['profile_ageGroup'].isnull(), 'profile_ageGroup'] = -99
    
    # 'profile_bicycleUse': nan = -99
    print(clean_data.profile_bicycleUse.unique())
    clean_data.loc[clean_data['profile_bicycleUse'].isnull(), 'profile_bicycleUse'] = -99
    
    # 'profile_hasChildren': True = 1, False = 0, nan = -99
    print(clean_data.profile_hasChildren.unique())
    clean_data.loc[clean_data['profile_hasChildren'] == True, 'profile_hasChildren'] = 1
    clean_data.loc[clean_data['profile_hasChildren'] == False, 'profile_hasChildren'] = 0
    clean_data.loc[clean_data['profile_hasChildren'].isnull(), 'profile_hasChildren'] = -99
    
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
   
    # 'profile_transportRatings_safe': nan = -99
    print(clean_data.profile_transportRatings_safe.unique())
    clean_data.loc[clean_data['profile_transportRatings_safe'].isnull(),
                              'profile_transportRatings_safe'] = -99
    
    # 'profile_transportRatings_faster': nan = -99
    print(clean_data.profile_transportRatings_faster.unique())
    clean_data.loc[clean_data['profile_transportRatings_faster'].isnull(),
                              'profile_transportRatings_faster'] = -99
    
    # 'profile_transportRatings_bikefun': nan = -99
    print(clean_data.profile_transportRatings_bikefun.unique()) 
    clean_data.loc[clean_data['profile_transportRatings_bikefun'].isnull(),
                              'profile_transportRatings_bikefun'] = -99

    # 'profile_transportRatings_weather': nan = -99
    print(clean_data.profile_transportRatings_weather.unique())
    clean_data.loc[clean_data['profile_transportRatings_weather'].isnull(),
                              'profile_transportRatings_weather'] = -99 
    
    # 'ratings_scene_id': nan = -99
    print(clean_data.ratings_scene_id.unique())
    scene_ids = list(clean_data.ratings_scene_id.unique())
    
    a = 1
    for i in scene_ids:
        clean_data.loc[clean_data['ratings_scene_id'] == i, 'ratings_scene_id'] = a
        a = a + 1

    clean_data.loc[clean_data['ratings_scene_id'].isnull(), 'ratings_scene_id'] = -99 
    
    
    # 'ratings_rating': nan = drop
    print(clean_data.ratings_rating.unique())
    clean_data.dropna(inplace=True)
    
    clean_data.to_csv('clean_data.csv')
    
#clean_data()

#%% Split data sets
def split_data():
    global X_train
    global X_test
    global y_train
    global y_test
    
    clean_data = pd.read_csv('clean_data.csv')
    clean_data.drop(['Unnamed: 0'], axis = 1, inplace = True)
    
    X = clean_data.drop(columns = ['ratings_rating'])
    y = clean_data['ratings_rating']
    X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2)

split_data()

#%% Train model(s) and export
def train_models():
    global DTC
    global RFC
    global MLP
    global SVM

    # Decision tree
    DTC = DecisionTreeClassifier()
    DTC.fit(X_train, y_train)
    joblib.dump(DTC, 'Cycle_inf_risk-dtc.joblib')
    
    # Random forrest
    RFC = RandomForestClassifier(random_state=0)
    RFC.fit(X_train, y_train)
    joblib.dump(RFC, 'Cycle_inf_risk-rfc.joblib')
    
    # Multi layer perception
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    MLP.fit(X_train, y_train)
    joblib.dump(MLP, 'Cycle_inf_risk-mlp.joblib')
    
    # Support vector machines
    SVM = svm.SVC()
    SVM.fit(X_train, y_train)
    joblib.dump(SVM, 'Cycle_inf_risk-svm.joblib')
    
#train_models()
    
#%% Load trained models, make predictions, and score
def make_predictions():
    global X_test
    global y_test
    global DTC_predicts
    global RFC_predicts
    global individual_prediction
    global importances
    global RFC
    
    # Random forest
    RFC = joblib.load('Cycle_inf_risk-rfc.joblib')
    
    RFC_predictions = RFC.predict(X_test)
    RFC_predicts = pd.DataFrame(RFC_predictions)
    X_test = X_test.reset_index(drop=True)
    RFC_predicts = pd.concat([X_test, RFC_predicts], axis=1)
    
    RFC_score = accuracy_score(y_test, RFC_predictions)
    print('RFC Score: ' + str(RFC_score))
    
    X_individual_data = {'profile_gender':[1], 'profile_ageGroup':[5], 'profile_bicycleUse':[3] ,
                         'profile_hasChildren':[1],'profile_perspective':[1], 'profile_transportRatings_car':[1],
                         'profile_transportRatings_public':[2], 'profile_profile_transportRatings_bicycle':[5],
                         'profile_profile_transportRatings_motorbike':[0], 'profile_profile_transportRatings_pedestrian':[4],
                         'profile_profile_transportRatings_safe':[0], 'profile_profile_transportRatings_faster':[4],
                         'profile_profile_transportRatings_bikefun':[4], 'profile_profile_transportRatings_weather':[0],
                         'ratings_scene_id':[111]}
    X_individual = pd.DataFrame(data = X_individual_data)
    individual_prediction = RFC.predict(X_individual)

    # Decision tree
    #DTC = joblib.load('Cycle_inf_risk-dtc.joblib')
    
    #DTC_predictions = DTC.predict(X_test)
    #DTC_predicts = pd.DataFrame(DTC_predictions)
    #X_test = X_test.reset_index(drop=True)
    #DTC_predicts = pd.concat([X_test, DTC_predicts], axis=1)
    
    # export tree and score
    #tree.export_graphviz(DTC, out_file = 'Cycle_inf_risk-dtc.dot',
    #                     feature_names = ['age', 'gender'],
    #                     class_names = sorted(y_train.unique()),
    #                     label = 'all',
    #                     rounded = True,
    #                     filled = True)
    
    #DTC_score = accuracy_score(y_test, DTC_predictions)
    #print('DTC Score: ' + str(DTC_score))

    
    # Multi Layer Perception
    # load trained model and make predictions
    #MLP = joblib.load('Cycle_inf_risk-mlp.joblib')
    
    #MLP_predictions = MLP.predict(X_test)
    #MLP_predicts = pd.DataFrame(MLP_predictions)
    #X_test = X_test.reset_index(drop=True)
    #MLP_predicts = pd.concat([X_test, MLP_predicts], axis=1)
    
    #MLP_score = accuracy_score(y_test, MLP_predictions)
    #print('MLP Score: ' + str(MLP_score))
    
    
    # Support vector machines
    # load trained model and make predictions
    #SVM = joblib.load('Cycle_inf_risk-svm.joblib')
    
    #SVM_predictions = SVM.predict(X_test)
    #SVM_predicts = pd.DataFrame(SVM_predictions)
    #X_test = X_test.reset_index(drop=True)
    #SVM_predicts = pd.concat([X_test, SVM_predicts], axis=1)
    
    #SVM_score = accuracy_score(y_test, SVM_predictions)
    #print('SVM Score: ' + str(SVM_score))

#make_predictions()

#%% Assess model
RFC = joblib.load('Cycle_inf_risk-rfc.joblib')

# Mean decrease in impurity
importances = RFC.feature_importances_
std = np.std([tree.feature_importances_ for tree in RFC.estimators_], axis=0)

feature_names = ['gender', 'ageGroup', 'bicycleUse', 'hasChildren','perspective',
                 'transRating_car', 'transRating_public', 'transRating_bicycle',
                 'transRating_m.bike', 'transRating_ped.', 'ttransRating_safe',
                 'transRating_faster', 'transRating_bikefun', 'transRating_weather',
                 'scene_id']
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Feature permutation
result = permutation_importance(RFC, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
