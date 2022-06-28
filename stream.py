import streamlit as st
import pandas as pd
import numpy as np

import requests
import json

import plotly.graph_objects as go


BASE = "http://127.0.0.1:5000/"

#liste_datas = json.load(open('test_df.json', 'rb'))

#for data in liste_datas[:50]:
#    res = requests.post(BASE, json=data)
#    print(res.json())

st.title('Scoring Clients')


df = pd.read_csv('test_df.csv')
#st.dataframe(df)

option = st.selectbox('id_client', tuple(df.SK_ID_CURR.tolist()))
df_per_personne = df.loc[df.SK_ID_CURR == option]
st.dataframe(df_per_personne)

res = requests.post(BASE, json=json.loads(df_per_personne.to_json(orient="records")))
#st.json(res.json())

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = res.json()['proba'],
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Score"},
    gauge = {'axis': {'range': [0, 1]},
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 0.15}}))

with st.expander("OUTPUT"):
    st.header("Target")
    st.subheader(res.json()['output'])

    st.header("Score")
    st.plotly_chart(fig)



