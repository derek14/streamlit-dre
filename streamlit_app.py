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
commentary_df = apartments_database.retrieve_commentary_dataframe(building=building)
st.write(commentary_df["commentary"].values[0].replace("$","\$"))

st.subheader("Ensembled Pricing Model")
prediction_df = apartments_database.retrieve_prediction_dataframe(building=building)
st.dataframe(apartments_database.format_prediction_df(prediction_df), hide_index=True, use_container_width=True)

st.subheader("Recent Transactions")
act_one_df = apartments_database.retrieve_building_dataframe(building=building)
recent_data = apartments_database.recent_df(act_one_df, cutoff=20)
st.dataframe(apartments_database.format_building_df(recent_data), hide_index=True, use_container_width=True)