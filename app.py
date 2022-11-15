import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
df1 = df.drop(columns='Churn', axis=1)

st.title("Customer Churn Predictor")

contract = st.selectbox('Contract Type', df1['Contract'].unique())

monthly_charges = st.number_input('Monthly Charges')

tenure_year = st.number_input('Years of tenure')

total_charges = st.number_input('Total Charges')

internet_service = st.selectbox('Internet Service', df1['InternetService'].unique())

if st.button('Calculate'):
    query = np.array([int(tenure_year),float(monthly_charges),float(total_charges),contract,internet_service])
    query_df = pd.DataFrame(columns=['tenure_year','MonthlyCharges','TotalCharges','Contract','InternetService'],data=query.reshape(1,-1))
    query_df['tenure_year'] = query_df['tenure_year'].astype('int64')
    query_df['MonthlyCharges'] = query_df['MonthlyCharges'].astype('float32')
    query_df['TotalCharges'] = query_df['TotalCharges'].astype('float32')
    print(query_df.info())
    query_joined = pd.concat([df1,query_df], ignore_index=True)
    print(df1.info())
    query_dummi = pd.get_dummies(query_joined)
    prediction = model.predict(query_dummi.tail(1))

    if prediction[0] == 0:
        st.title('Customer is likely to continue with us')
    else:
        st.title('Customer is likely to be churned')
