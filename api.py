import os
import pickle
import pandas as pd

from flask import Flask, jsonify, request
import numpy as np

from sklearn.preprocessing import StandardScaler

MODEL_PATH_1 = "model/kmeans_props.pkl"
with open(MODEL_PATH_1, "rb") as rf:
    kmeans_props = pickle.load(rf)

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

# predict function
def predict_function_props(sample, clst):
    
    df = pd.DataFrame.from_dict(sample, orient='index')
    
    # IMPORTANT: USE THE SUITABLE ORIENT

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


# Predict function api
@app.route("/predict/props", methods=["POST"])
def predict():
    sample = request.get_json()
#     print(len(sample))
    predictions = predict_function_props(sample, kmeans_props)
    pred = predictions.tolist()
    result = {
        'prediction': pred
    }

    return jsonify(result)


# evaluate function
def evaluate_function(sample, clf):
    
    # IMPORTANT: USE THE SUITABLE ORIENT
    test = pd.DataFrame.from_dict(sample, orient='index')

    # if you have any step of data transformation, include it here
    converter = {'True' : 1, 'False' : 0}
    nominal_columns = list(test.columns[1:-1])
    print(nominal_columns)
    for col in nominal_columns:
        test[col] = test[col].astype(int)        
 
    # separate features / label column here:
    X_test = test.iloc[:, 1:-1]
    y_test = test['type']
    
    # label encoder
    from sklearn import preprocessing
    enc = preprocessing.LabelEncoder()
    encoder_dict = np.load('model/label_encoder.npy', allow_pickle=True).tolist()
    for nom in encoder_dict:
        for col in X_test.columns:
            if nom == col:
                enc.classes_ = encoder_dict[nom]
                X_test[[col]] = enc.transform(X_test[[col]])
                
    # predict
    y_pred = clf.predict(X_test)
    
    # evaluate
    from sklearn.metrics import accuracy_score, precision_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    return accuracy, precision


# evaluate function api
@app.route("/evaluate", methods=["POST"])
def evaluate():
    sample = request.get_json()
    accuracy = evaluate_function(sample, clf)

    result = {
        'accuracy': accuracy,
        'precision': precision
    }
    
    return jsonify(result)

# main
if __name__ == '__main__':
    app.run(debug=True)