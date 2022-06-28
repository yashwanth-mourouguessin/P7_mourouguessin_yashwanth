import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import sklearn

app = Flask(__name__)
transformer = pickle.load(open('transformer.pkl', 'rb'))

from zipfile import ZipFile
#with ZipFile('rdf_classifier.pkl.zip', 'r') as zip:    
    #zip.printdir() 
    #zip.extractall()
    
model = pickle.load(open('xgb_classifier_2.pkl', 'rb'))
threshold = 0.15

selected_col_1 = ['CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'REGION_POPULATION_RELATIVE', 
                'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 
                'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'REGION_RATING_CLIENT', 
                'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 
                'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2','EXT_SOURCE_3', 
                'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE','DAYS_LAST_PHONE_CHANGE', 
                'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK', 
                'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
                'Annuity/Income','Flag_Income_Greater_Credit', 'nbr_doc']

selected_col_2 = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_CONTRACT_TYPE',
                  'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE',
                  'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'WEEKDAY_APPR_PROCESS_START']
 

selected_col = selected_col_1 + selected_col_2

@app.route('/',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    df = pd.json_normalize(data)
    vec = transformer.transform(df[selected_col])
    proba = model.predict_proba(vec)[:,1]
    output = (proba >= threshold).astype('int')

    return {'proba': float(proba[0]), 'output': int(output[0])}

if __name__ == "__main__":
    app.run(debug=True)
