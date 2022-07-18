import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
from streamlit_shap import st_shap
import matplotlib.pyplot as plt
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

BASE = "https://scoring-client.herokuapp.com"

st.title('Scoring Clients')

df = pd.read_csv('test_df.csv')

vec = pd.DataFrame()
for col in selected_col:
    vec[col] = transformer[col].transform([[elt] for elt in df[col]]).reshape(1, df.shape[0])[0]
vec['SK_ID_CURR'] = df.SK_ID_CURR

# FEATURE IMPORTANCE
st.subheader('FEATURE IMPORTANCE')
explainer_1 = shap.Explainer(model, vec[selected_col])
shap_values_1 = explainer_1(vec[selected_col])
st_shap(shap.plots.bar(shap_values_1, max_display=46))

st.subheader("Client's informations")
option = st.selectbox('id_client', tuple(df.SK_ID_CURR.tolist()))
df_per_personne = df.loc[df.SK_ID_CURR == option]
st.dataframe(df_per_personne)

#st.subheader("Transformed informations")
vec_per_personne = vec.loc[vec.SK_ID_CURR == option]
df_per_personne = df.loc[df.SK_ID_CURR == option]
idx = vec.loc[vec.SK_ID_CURR == option].index[0]

res = requests.post(BASE, json=json.loads(vec_per_personne[selected_col].to_json(orient="records")))


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = res.json()['proba'],
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = {'axis': {'range': [0, 1]},
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.35}}))

with st.expander("OUTPUT"):
    st.header("Target")
    st.subheader(res.json()['output'])


    st.header("Score")
    st.plotly_chart(fig)
    
    # IMPORTANCE DES FEATURES D'UNE PERSONNE
    st.header("Client's features importance")
    st_shap(shap.plots.bar(shap_values_1[idx]))

    
    # Distribution de Features
    st.header("Features")
    options = st.multiselect('SELECT TWO DIFFERENT FEATURES', options=selected_col, default=['CNT_CHILDREN', 'CODE_GENDER'])
    option_features_1, option_features_2 = options[0], options[1]

    df_f_1 = pd.DataFrame(df[option_features_1].value_counts())
    index = list(df_f_1.index)
    values = list(df_f_1[option_features_1])
    fig = plt.figure(figsize = (30, 5))
    v_1 = df_per_personne[option_features_1].loc[0]
    clrs = ['red' if x==v_1 else 'grey' for x in index]
    plt.bar(index, values, color=clrs)
    plt.xlabel(option_features_1)
    plt.ylabel("COUNT")
    plt.title("REPARTION des données selon la variable "+option_features_1)
    st.pyplot(fig)

    df_f_2 = pd.DataFrame(df[option_features_2].value_counts())
    index = list(df_f_2.index)
    values = list(df_f_2[option_features_2])
    fig = plt.figure(figsize = (30, 5))
    v_2 = df_per_personne[option_features_2].loc[0]
    clrs = ['red' if x==v_2 else 'grey' for x in index]
    plt.bar(index, values, color=clrs)
    plt.xlabel(option_features_2)
    plt.ylabel("COUNT")
    plt.title("REPARTION des données selon la variable "+option_features_2)
    st.pyplot(fig)
