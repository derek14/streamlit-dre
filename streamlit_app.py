import streamlit as st
from utils import ApartmentDatabase

st.title("Dubai Inventory Analysis")
building_option = st.selectbox(
    'Pick a building?',
    ('Act One', 'Creek Vista Grande', 'Crest Grande Tower C', 'Design Quarters', 'The Residence, District One', 'Dunya Tower'))

if building_option == 'Creek Vista Grande':
    building = 1
elif building_option == 'Crest Grande Tower C':
    building = 2
elif building_option == 'Design Quarters':
    building = 3
elif building_option == 'The Residence, District One':
    building = 4
elif building_option == 'Dunya Tower':
    building = 5
else:
    building = 0

st.header(building_option)
apartments_database = ApartmentDatabase(
  url=st.secrets["SUPA_URL"],
  key=st.secrets["SUPA_KEY"],
  email=st.secrets["email"],
  password=st.secrets["password"],
  with_webdriver=False,
)

st.subheader("A.I. Commentary")
st.caption("Commentary written by GPT-4.")
commentary_df = apartments_database.retrieve_commentary_dataframe(building=building)
st.write(commentary_df["commentary"].values[0].replace("$","\$"))

st.subheader("Ensembled Pricing Model")
st.caption("Regresss and recalibrate - We used AutoML techniques to first identify the best models to extract relations in the data, from there we picked the best 5 models, retrain them with the entire dataset. To properly adjust for the lagged effect in a trending market, we do another recalibration as a final step.")
prediction_df = apartments_database.retrieve_prediction_dataframe(building=building)
st.dataframe(apartments_database.format_prediction_df(prediction_df), hide_index=True, use_container_width=True)

st.subheader("Recent Transactions")
st.caption("Last 20 transactions.")
building_df = apartments_database.retrieve_building_dataframe(building=building)
recent_data = apartments_database.recent_df(building_df, cutoff=20)
st.dataframe(apartments_database.format_building_df(recent_data), hide_index=True, use_container_width=True)

st.subheader("Transaction Volume")
st.caption("Transaction volumn aggregated by week.")
weekly_counts = apartments_database.format_bar_df(building_df)
st.bar_chart(weekly_counts)