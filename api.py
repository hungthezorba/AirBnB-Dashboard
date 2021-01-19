import os
import pickle
import pandas as pd

from flask import Flask, jsonify, request
import numpy as np

from sklearn.preprocessing import StandardScaler

MODEL_PATH_1 = "model/kmeans_props.pkl"
with open(MODEL_PATH_1, "rb") as rf:
    kmeans_props = pickle.load(rf)

MODEL_PATH_2 = "model/kmeans_am.pkl"
with open(MODEL_PATH_2, "rb") as rf:
    kmeans_am = pickle.load(rf)


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


# Cluster function api
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

# Cluster function api
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

# main
if __name__ == '__main__':
    app.run(debug=True)