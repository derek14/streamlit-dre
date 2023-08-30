import streamlit as st
from utils import ApartmentDatabase

st.title("Dubai Inventory Analysis")
option = st.selectbox(
    'Pick a building?',
    ('Act One', 'Creek Vista Grande', 'Crest Grande Tower C', 'Design Quarters', 'The Residence, District One', 'Dunya Tower'))

match option:
    case 'Creek Vista Grande':
        building=1
    case 'Crest Grande Tower C':
        building=2
    case 'Design Quarters':
        building=3
    case 'The Residence, District One':
        building=4
    case 'Dunya Tower':
        building=5
    case _:
        building=0

st.header(option)
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