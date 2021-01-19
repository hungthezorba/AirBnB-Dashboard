import os
import pickle
import pandas as pd

from flask import Flask, jsonify, request
import numpy as np
import ast

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from datetime import timedelta
from datetime import datetime

MODEL_PATH_1 = "model/kmeans_props.pkl"
with open(MODEL_PATH_1, "rb") as rf:
    kmeans_props = pickle.load(rf)

MODEL_PATH_2 = "model/kmeans_am.pkl"
with open(MODEL_PATH_2, "rb") as rf:
    kmeans_am = pickle.load(rf)

MODEL_PATH_3 = "model/kmeans_host.pkl"
with open(MODEL_PATH_3, "rb") as rf:
    kmeans_host = pickle.load(rf)


# Init the app
app = Flask(__name__)

# App health check
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    msg = (
        "This is a sentence to check if the server is running"
    )
    return jsonify({"message": msg})


def handle_missing_value(df, columns=[]):
    pass


# Function to scale and standardized numerical and continous variable (not boolean or categorical)
def prepare_and_std_props(df):
    if "id" in list(df):
        df_new = df.drop(['id'], axis=1) # Drop identity column
    else:
        df_new = df
    df_new_columns = df_new.columns.to_list()
    
    # Scale features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_new)
    
    # Initialize dataframe from scaled array
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df_new_columns
    df_scaled.head()
    return df_scaled

## Function to cluster amenities features
def cluster_function_am(sample, clst):
    # Retrieve dataframe from json
    df = pd.DataFrame.from_dict(sample, orient='index')
    
    #### Feature transformation for amenities cluster ####

    totalAmenities = []
    for index,row in df.iterrows():
        charList = list(row["amenities"]) #c
        for i in charList:
            if i ==",":
                pass
            elif i == " ":
                pass
            elif not i.isalpha():
                charList.remove(i)
        cleanedString = "".join(charList).lower() #d 

        #get this row amenities
        thisRowAmenities = cleanedString.split(", ")
        totalAmenities.append(thisRowAmenities)
        
        totalAmenitiesClean = []
    for i in totalAmenities:
        thisListClean = []
        for element in i:
            try:
                element = element.replace("]","")
            except:
                pass 
            try:
                element =  element.replace("\"","")
            except:
                pass
            thisListClean.append(element)
        totalAmenitiesClean.append(thisListClean)
    df["amenities_clean"] = pd.Series(totalAmenitiesClean, index=df.index)
    df["number_amenities"] = df["amenities_clean"].apply(lambda x:len(x))
    
    # Room-type-rank Handler
    room_type_unique = df["room_type"].unique().tolist()
    room_type_rank = {
        'Shared room':1,
        'Private room':2,
        'Hotel room':3,
        'Entire home/apt':4,
    }
    encode_room_type = []
    for index,row in df.iterrows():
        InAgroup = False
        for key in list(room_type_rank):
            if key == row["room_type"]:
                InAgroup = True
                break
        if not InAgroup:
            encode_room_type.append(0)
        else:
            encode_room_type.append(room_type_rank[key])
    df["room_type_rank"] = pd.Series(encode_room_type, index=df.index)
    
    # Bathroom features
        # this cell encode the bathrooms_text to number of bathrooms, if the row is nan 
    # we will check the roomtypes encode because if the room_types encode is less than 2, there are high chance 
    # that the traveler will need to share a bathroom with the host
    bathrooms_Count = []
    bathrooms_share = []
    for index,row in df.iterrows():
        try:
            count = round(float(row["bathrooms_text"].split()[0]))
            if "shared" in row["bathrooms_text"].lower():
                bathrooms_share.append(1)
            else:
                bathrooms_share.append(0)
        except:
    #         print(index)
            count = 1
            if row["room_type_rank"] < 3:
                bathrooms_share.append(1)
            else:
                bathrooms_share.append(0)
        bathrooms_Count.append(count)
    
    df["bathrooms_Count"] = pd.Series(bathrooms_Count, index=df.index)
    df["bathrooms_share"] = pd.Series(bathrooms_share, index=df.index)
    
    # Fill NaN for beds column if (has)
    df["beds"].fillna(1, inplace=True)
    df["bedrooms"].fillna(1, inplace=True)
    
    # Aggregate fit-in features
    df_predict = df[['number_amenities', 'bathrooms_Count', 'bathrooms_share', 'beds']]
    df_predict_scaled = prepare_and_std_props(df_predict)
    
    print(df.isna().sum())
    # predict
    y_cluster = clst.predict(df_predict_scaled)
    return y_cluster
    
    

## Cluster function for properties
def cluster_function_props(sample, clst):
    
    df = pd.DataFrame.from_dict(sample, orient='index')
    

    
    ##### if you have any step of data transformation, include it here ####
    df["property_type"] = df["property_type"].str.replace("tipi","tent")
    df["property_type"] = df["property_type"].str.replace("bed and breakfast","casa particular")
    df["property_type"] = df["property_type"].str.replace("Private room in condominium","Entire apartment")
    property_count = df["property_type"].value_counts()
    
    home_types = [] 
    for i in list(property_count.index):
        if "in" in i:
            home_types.append(i.split(" in ")[-1].lower())
        elif len(i.split()) == 1:
            home_types.append(i.lower())
        else:
            home_types.append(" ".join(i.split()[1:]).lower())
    
    home_types = pd.Series(home_types).unique().tolist()    
    home_types.remove("entire condominium")
    
    needDoubleCheck = ["house","hotel","apartment","room"]
    
    encodePrivacy = []
    StringPropType = [] # used for visualize 
    count = 0 
    for index,row in df.iterrows():
        guess = []
        for key in home_types:
            if key in row["property_type"].lower():
                guess.append(key)
                inDoubleCheck = False
                if key in needDoubleCheck:
                    inDoubleCheck = True
                if not inDoubleCheck :
                    count +=1 
                    break
        try:
            StringPropType.append(guess[-1])
        except:
            print(f"{index} ",end=" ")
            print(row["property_type"])
        if "shared" in row["property_type"].lower():
            encodePrivacy.append(0)
        elif "entire" in row["property_type"].lower(): 
            encodePrivacy.append(2)
        else:
            encodePrivacy.append(1)
            
    df["encodePrivacy"] = pd.Series(encodePrivacy, index=df.index)
    df["StringPropType"] = pd.Series(StringPropType, index=df.index)
    
    with open("model/DictGroupProp","rb") as f:
        DictGroupProp = pickle.load(f)
        f.close()
    
    with open("model/DictLabelProp","rb") as f:
        DictLabel = pickle.load(f)
        f.close()
    
    PropTypes =  pd.Series(StringPropType).to_numpy().tolist()
    
    EncodedPropType = []
    for prop_type in PropTypes:
        InAgroup = False
        for key in DictGroupProp:
            if prop_type in DictGroupProp[key]:
                InAgroup = True
                EncodedPropType.append(DictLabel[key])
                break 
        if not InAgroup:
            EncodedPropType.append(DictLabel["room"])
    df["encoded_prop_type"] = pd.Series(EncodedPropType, index=df.index)
    
    ## Room-Type-Rank handler ##
    room_type_unique = df["room_type"].unique().tolist()
    room_type_rank = {
        'Shared room':1,
        'Private room':2,
        'Hotel room':3,
        'Entire home/apt':4,
    }
    encode_room_type = []
    for index,row in df.iterrows():
        InAgroup = False
        for key in list(room_type_rank):
            if key == row["room_type"]:
                InAgroup = True
                break
        if not InAgroup:
            encode_room_type.append(0)
        else:
            encode_room_type.append(room_type_rank[key])
    df["room_type_rank"] = pd.Series(encode_room_type, index=df.index)
    
    ## Handle formatting Price ##
    price = []
    count = 0
    for index,row in df.iterrows():
        try:
            if row["price"][0] =="$":
                price.append(float(row["price"][1:]))
        except:
            listOfNum = row["price"][1:].split(",")
            if len(listOfNum) !=2:
                print(f"{index}",end=" ")
                print(row["price"])
            a = int(listOfNum[0])*1000
            b = float(listOfNum[1])
            thisPrice = float(a)+b
            price.append(thisPrice)

    df["price"] = pd.Series(price, index=df.index)
    #### #####
    df_predict = df[['price', 'encoded_prop_type', 'encodePrivacy', 'room_type_rank']]
    df_predict_scaled = prepare_and_std_props(df_predict)
    
    for i in ['price', 'encoded_prop_type', 'encodePrivacy', 'room_type_rank']:
        if(df_predict[i].isna().sum() >= 1):
            print(i, "has missing values", df_predict[i].isna().sum())

    # predict
    y_cluster = clst.predict(df_predict_scaled)
    return y_cluster

def cluster_function_host(sample, clst):
    # Retrieve dataframe from request body
    df = pd.DataFrame.from_dict(sample, orient='index')
    
    #### Impute missing data ####
    df['host_since'].fillna(df['host_since'].mode()[0], inplace=True)
    
    #### Feature transformation for host cluster ####
    # Number of verfication methods per host
    df["num_of_verf"] = df['host_verifications'].apply(lambda x: len(ast.literal_eval(x)) if x != 'None' else 0)
    
    # Host since (days) feature
    date_host_ds = df['host_since']
    date_host_ds = pd.to_datetime(date_host_ds, infer_datetime_format=True)
    # Retrieve today's date and time
    today = datetime.today()
    host_since_days = date_host_ds.apply(lambda x: (today - x).days)
    host_since_days_df = host_since_days.to_frame().rename(columns={'host_since': 'host_since_days'}) # Convert series to dataframe
    
    
    # Fill NaN value with median
    df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    
    # Initialize scaler
    scale_score = MinMaxScaler()
    df['review_scores_rating_nml'] = scale_score.fit_transform(df[['review_scores_rating']])
    
    # Groupby average host rating for all their houses in their listings
    df_host = df.groupby('host_id')['review_scores_rating_nml'].agg(['mean'])
    (df_host.shape, df.shape)
    df_host.reset_index(inplace=True)
    df_host.rename(columns={'mean': 'average_review_score'}, inplace=True)
    
    # Groupby average host rating for all their houses in their listings
    df_host_count = df.groupby('host_id')['host_id'].agg(['count'])
    (df_host_count.shape, df.shape)
    df_host_count.reset_index(inplace=True)
    df_host_count.rename(columns={'count': 'num_of_host_props'}, inplace=True)
    
    # Define list of superhost based on host's id
    df_host_superhost = df[['host_id', 'host_is_superhost']]
    df_host_superhost.drop_duplicates(subset='host_id', keep='first', inplace=True)
    df_host_superhost.reset_index(drop=True, inplace=True)
    
    # Encode superhost feature
    df_host_superhost['host_is_superhost'] = df_host_superhost['host_is_superhost'].apply(lambda x: "1" if x == "t" else "0")
    df_host_superhost['host_is_superhost'] = df_host_superhost['host_is_superhost'].astype('int32')
    # Host experience by year feature
    df = pd.concat([df, host_since_days_df], axis=1)
    df['host_year_exp'] = df['host_since_days'].apply(lambda x: x/365.25)
    
    # Host-experience feature
    df_host_experience= df.groupby("host_id")["host_year_exp"].agg(["mean"])
    df_host_experience.reset_index(inplace=True)
    df_host_experience.rename(columns={'mean': 'host_experiece'}, inplace=True)
    
    # Recognition features
    list_of_recog = ['host_has_profile_pic', 'host_identity_verified']
    # Impute with most frequent value
    for col in list_of_recog:
        df[col] = df[col].apply(lambda x: x if pd.notnull(x) else df[col].mode()[0])
    # Encode recog features

    
    # Define list of superhost based on host's id
    df_host_recog = df[['host_id', 'host_has_profile_pic', 'host_identity_verified']]
    df_host_recog.drop_duplicates(subset='host_id', keep='first', inplace=True)
    df_host_recog.reset_index(drop=True, inplace=True)
    for col in list_of_recog:
        df_host_recog[col] = df_host_recog[col].apply(lambda x: "1" if x == "t" else "0")
        df_host_recog[col] = df_host_recog[col].astype('int32')
    
    # Define dataframe for number of verf
    df_host_verf_num = df.groupby("host_id")["num_of_verf"].agg(["max"])
    df_host_verf_num.reset_index(inplace=True)
    df_host_verf_num.rename(columns={'max': 'num_of_verf'}, inplace=True)
    
    # Merge all host features
    df_host = df_host.merge(df_host_count, on='host_id')
    df_host = df_host.merge(df_host_superhost, on='host_id')
    df_host = df_host.merge(df_host_recog, on='host_id')
    df_host = df_host.merge(df_host_experience, on='host_id')
    df_host = df_host.merge(df_host_verf_num, on='host_id')
    
    df_host.drop(['host_id'], axis=1, inplace=True)
    # Scale host features
    df_behave_scaled = prepare_and_std_props(df_host)
    print(df_host.isna().sum())
    
    # Clustering
    y_cluster = kmeans_host.predict(df_behave_scaled)
    
    return y_cluster

# Cluster function api for properties
@app.route("/cluster/props", methods=["POST"])
def cluster_props():
    sample = request.get_json()
#     print(len(sample))
    predictions = cluster_function_props(sample, kmeans_props)
    pred = predictions.tolist()
    result = {
        'prediction': pred
    }

    return jsonify(result)

# Cluster function api for amenities
@app.route("/cluster/am", methods=["POST"])
def cluster_am():
    sample = request.get_json()
#     print(len(sample))
    predictions = cluster_function_am(sample, kmeans_am)
    pred = predictions.tolist()
    result = {
        'prediction': pred
    }
    return jsonify(result)

# Cluster function api for host
@app.route("/cluster/host", methods=["POST"])
def cluster_host():
    sample = request.get_json()
#     print(len(sample))
    predictions = cluster_function_host(sample, kmeans_host)
    pred = predictions.tolist()
    result = {
        'prediction': pred
    }
    return jsonify(result)


# main
if __name__ == '__main__':
    app.run(debug=True)