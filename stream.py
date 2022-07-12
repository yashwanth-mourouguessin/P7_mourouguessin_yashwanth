import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from streamlit_shap import st_shap

import shap
import plotly.graph_objects as go

transformer = pickle.load(open('transformers.pkl', 'rb'))

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

model = pickle.load(open('xgb_classifier_final.pkl', 'rb'))

BASE = "http://127.0.0.1:5000/"

st.title('Scoring Clients')

df = pd.read_csv('test_df.csv')
#st.dataframe(df)


vec = pd.DataFrame()
for col in selected_col:
    vec[col] = transformer[col].transform([[elt] for elt in df[col]]).reshape(1, df.shape[0])[0]
vec['SK_ID_CURR'] = df.SK_ID_CURR

# FEATURE IMPORTANCE
st.subheader('FEATURE IMPORTANCE')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(vec[selected_col])
st_shap(shap.summary_plot(shap_values, features=vec[selected_col].to_numpy(), feature_names=vec[selected_col].columns))

#
st.subheader("Client's informations")
option = st.selectbox('id_client', tuple(df.SK_ID_CURR.tolist()))
df_per_personne = df.loc[df.SK_ID_CURR == option]
st.dataframe(df_per_personne)

#st.subheader("Transformed informations")
vec_per_personne = vec.loc[vec.SK_ID_CURR == option]
#st.dataframe(vec_per_personne)

res = requests.post(BASE, json=json.loads(vec_per_personne[selected_col].to_json(orient="records")))
#st.json(res.json())


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = res.json()['proba'],
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = {'axis': {'range': [0, 1]},
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.35}}))

#with st.expander("OUTPUT"):
#    st.header("Target")
#    st.subheader(res.json()['output'])


#    st.header("Score")
#    st.plotly_chart(fig)
    
    # IMPORTANCE DES FEATURES D'UNE PERSONNE
#    st.header("Client's features importance")
#    st_shap(shap.force_plot(explainer.expected_value,
#                            explainer.shap_values(vec_per_personne[selected_col]),
#                            features=vec_per_personne[selected_col],
#                            feature_names=vec_per_personne[selected_col].columns))

st.header("Target")
st.subheader(res.json()['output'])


st.header("Score")
st.plotly_chart(fig)
    
# IMPORTANCE DES FEATURES D'UNE PERSONNE
st.header("Client's features importance")
st_shap(shap.force_plot(explainer.expected_value,
                            explainer.shap_values(vec_per_personne[selected_col]),
                            features=vec_per_personne[selected_col],
                            feature_names=vec_per_personne[selected_col].columns))


