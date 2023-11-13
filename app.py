import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import csv

gss_data = pd.read_csv(r'gss2016.csv', delimiter=",")

st.dataframe(gss_data)

gss_data_filtered = gss_data[['sex','race','age','degree','wrkstat','income','happy']]

st.dataframe(gss_data_filtered)