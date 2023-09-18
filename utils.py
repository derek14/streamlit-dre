import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import hashlib
from supabase import create_client, Client
import openai
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
import streamlit as st
import seaborn as sns

@st.cache_data
def get_sha256_hex(string):
    hash_object = hashlib.sha256()
    string_bytes = string.encode()
    hash_object.update(string_bytes)
    hex_digest = hash_object.hexdigest()
    return hex_digest

def create_generate_chat_chain(system_prompt:str, human_prompt:str, openai_key:str, temperature:int=0, model:str="gpt-3.5-turbo-0613"):
    llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_key, model=model)
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return LLMChain(llm=llm, prompt=chat_prompt)

class ApartmentDatabase():
  def __init__(self, url, key, email, password, with_webdriver:bool=True):
    self.supabase: Client = create_client(url, key)
    user = self.supabase.auth.sign_in_with_password({ "email": email, "password": password })
    self.supabase.postgrest.auth(user.session.access_token)
    self.table_name = 'apartments_transactions'
    self.table = self.supabase.table(self.table_name)
    # if with_webdriver:
    #   chrome_options = webdriver.ChromeOptions()
    #   chrome_options.add_argument('--headless')
    #   chrome_options.add_argument('--no-sandbox')
    #   chrome_options.add_argument('--disable-dev-shm-usage')
    #   self.chrome_options = chrome_options

  # def scrape_to_dataframe(self, target_url, xpath:str="//tr[@aria-label='Listing']", page:int=1):
  #   ext = "" if page ==1 else f"?page={page}"
  #   wd = webdriver.Chrome(options=self.chrome_options)
  #   wd.get(target_url+ext)
  #   delay = 3

  #   try:
  #     myElem = WebDriverWait(wd, delay).until(EC.presence_of_element_located((By.XPATH, xpath)))
  #   except TimeoutException:
  #     print("Loading took too much time!")

  #   df = pd.read_html(wd.page_source)[0]
  #   return df

  def tower_filter_a(self, df, tower:str="C"):
    filtered_df = df[df['Unit Number'].str.startswith(tower)]
    filtered_df['Unit Number'] = filtered_df['Unit Number'].str.replace(tower, '')
    return filtered_df

  def tower_filter_b(self, df, tower:str="B"):
    filtered_df = df[df['Location'].str.endswith(tower)]
    return filtered_df

  def clean_up_table_for_saving(self, df, building:int=0, ptype:int=0):
    df['Sale Price'] = df['Sale Price'].str.replace('[^\d.]', '', regex=True)
    df['Sale Price'] = pd.to_numeric(df['Sale Price']).astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['Built-up Area'] = df['Built-up Area'].str.split(' ').str[0].str.replace(',', '').astype(int)
    df['unit'] = df['Unit Number'].astype(str).str[-2:]
    df['floor'] = df['Unit Number'].astype(str).str[:-2]
    df['floor'] = df['floor'].replace("G", "0")
    df['Beds'] = df['Beds'].replace("Studio", "0")

    df["building"] = building
    # df["ptype"] = ptype
    df.columns = df.columns.str.replace('Beds', 'beds')
    df.columns = df.columns.str.replace('Built-up Area', 'bua')
    df.columns = df.columns.str.replace('Sale Price', 'price')

    df = df[["building", "floor", "unit", "year", "month", "day", "beds", "bua", "price"]]
    concat_row = lambda row: get_sha256_hex(' '.join([str(val) for val in row]))[:10]
    df['key'] = df.apply(concat_row, axis=1)
    return df

  def save_df_to_database(self, df):
    rows = df.to_dict(orient='records')
    response = self.table.upsert(rows, on_conflict='key').execute()
    print("Saved!")

  def retrieve_dataframe(self):
    data = self.table.select("*").execute()
    df = pd.DataFrame(data.data)
    return df

  @st.cache_data
  def retrieve_building_dataframe(_self, building:int=0):
    data = _self.table.select("*").filter('building', 'eq', building).execute()
    df = pd.DataFrame(data.data)
    return df
  
  @st.cache_data
  def retrieve_prediction_dataframe(_self, building:int=0):
    data = _self.supabase.table("dubai_predictions").select("*").execute()
    df = pd.DataFrame(data.data)
    return df[df["building"]==building]
  
  @st.cache_data
  def retrieve_commentary_dataframe(_self, building:int=0):
    data = _self.supabase.table("building_commentaries").select("*").execute()
    df = pd.DataFrame(data.data)
    return df[df["building"]==building]
  
  def format_violin_df(self, df):
    df["per_sq_ft"] = df["price"] / df["bua"]
    df["beds"] = df["beds"].astype(str)
    df.rename(columns={'per_sq_ft': 'AED per sq ft'}, inplace=True)
    df.rename(columns={'beds': 'Beds'}, inplace=True)
    return df.head(30)

  def format_bar_df(self, df):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df.set_index('date', inplace=True)
    weekly_counts = df.resample('W').count()['key']
    return weekly_counts

  def format_prediction_df(self, df):
    df = df[["unit", "min", "max"]]
    df.set_axis(['Unit', 'Lower Bound', 'Upper Bound'], axis='columns', inplace=True)
    return df
  
  def format_building_df(self, df):
    df = df[['transaction_date', "floor", "unit", "beds", "bua", "price"]]
    df.set_axis(['Transaction Date', 'Floor', 'Unit', 'Beds', "Area (sq ft)", 'Price (AED)'], axis='columns', inplace=True)
    return df

  def clear_database(self):
    result = self.table.select("id").execute()
    ids = [row["id"] for row in result.data]
    result = self.table.delete().in_("id", ids).execute()

  def show_head_markdown(self, df):
    markdown = df.head().to_markdown()
    print(markdown)

  def show_full_markdown(self, df):
    df.columns = df.columns.str.replace('bua', 'area')
    markdown = df[["floor", "unit", "year", "month", "day", "beds", "area", "price"]].to_markdown()
    print(markdown)

  def filter_df_for_gpt(self, df, target_unit, target_bua):
    gpt_df = df[["floor", "unit", "year", "month", "day", "beds", "bua", "price"]]
    gpt_df['transaction_date'] = pd.to_datetime(gpt_df[['year', 'month', 'day']])
    gpt_df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    filtered_df = gpt_df[(gpt_df["unit"]==target_unit)&(gpt_df['bua'] >= target_bua*.9) & (gpt_df['bua'] <= target_bua*1.1)].assign(transaction_date=gpt_df["transaction_date"].dt.date)
    filtered_df.sort_values('transaction_date', ascending=False, inplace=True)
    return filtered_df

  def filter_df_for_similar_units(self, df, target_unit, target_bua, cutoff:int=3, relaxed=False):
    gpt_df = df[["floor", "unit", "year", "month", "day", "beds", "bua", "price"]]
    gpt_df['transaction_date'] = pd.to_datetime(gpt_df[['year', 'month', 'day']])
    gpt_df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    if relaxed:
      filtered_df = gpt_df[(gpt_df['bua'] >= target_bua*.9) & (gpt_df['bua'] <= target_bua*1.1)].assign(transaction_date=gpt_df["transaction_date"].dt.date)
    else:
      filtered_df = gpt_df[(gpt_df["unit"]==target_unit)&(gpt_df['bua'] >= target_bua*.9) & (gpt_df['bua'] <= target_bua*1.1)].assign(transaction_date=gpt_df["transaction_date"].dt.date)
    filtered_df.sort_values('transaction_date', ascending=False, inplace=True)
    filtered_df = filtered_df.drop_duplicates(subset='floor')
    return filtered_df.head(cutoff)

  @st.cache_data
  def recent_df(_self, df, cutoff=20):
    gpt_df = df[["floor", "unit", "year", "month", "day", "beds", "bua", "price"]]
    gpt_df['transaction_date'] = pd.to_datetime(gpt_df[['year', 'month', 'day']])
    gpt_df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    filtered_df = gpt_df.sort_values(by='transaction_date', ascending=False)[:cutoff].assign(transaction_date=gpt_df["transaction_date"].dt.date)
    return filtered_df

  def recent_df_for_gpt(self, df, cutoff=20):
    gpt_df = df[["floor", "unit", "year", "month", "day", "beds", "bua", "price"]]
    gpt_df['transaction_date'] = pd.to_datetime(gpt_df[['year', 'month', 'day']])
    gpt_df.drop(['year', 'month', 'day'], axis=1, inplace=True)
    filtered_df = gpt_df.sort_values(by='transaction_date', ascending=False)[:cutoff].assign(transaction_date=gpt_df["transaction_date"].dt.date)
    return filtered_df.to_markdown(index=False)

  def recent_df_for_gpt_after_filter(self, df, cutoff=20):
    gpt_df = df[["floor", "unit",'transaction_date', "beds", "bua", "price"]]
    filtered_df = gpt_df[:cutoff]
    return filtered_df.to_markdown(index=False)

  def df_only_continuous_variables(self, df):
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['num_days_ago'] = (df['date'].max() - df['date']).dt.days
    df['num_days_ago_sq'] = df['num_days_ago'] **2
    beds_encoded = pd.get_dummies(df['beds'], prefix='beds')
    df = pd.concat([df, beds_encoded], axis=1)
    df["floor"] = df['floor'].astype(int)
    df["unit"] = df['unit'].astype(int)
    df = df.drop(['year', 'month', 'day', 'date', 'beds', 'key', 'building'], axis=1)
    df = df.sample(frac=1)
    return df

  # def do_machine_learning(self, df, test_size=0.1):
  #   df = apartments_database.df_only_continuous_variables(df)
  #   y = df['price']
  #   X = df.drop(columns=['price'], axis=1)
  #   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size)
  #   clf = LazyRegressor(ignore_warnings=False, custom_metric=None)
  #   models, predictions = clf.fit(X_train, X_test, y_train, y_test)
  #   return clf, models, X_train

  def input_df_from_data(self, data, training_df):
    df = pd.DataFrame(data)
    df.columns = df.columns.str.replace('area', 'bua')
    beds_encoded = pd.get_dummies(df['beds'], prefix='beds')
    df = df[['floor','unit','bua','num_days_ago', 'num_days_ago_sq']]
    df = pd.concat([df, beds_encoded], axis=1)
    missing_columns = list(set(training_df.columns) - set(df.columns))
    df = df.assign(**{col: 0 for col in missing_columns})
    return df
  
  def input_df_from_df(self, df, training_df):
    df['floor'] = df['floor'].replace("G", "0")
    df['beds'] = df['beds'].replace("Studio", "0")
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['year'] = df['transaction_date'].dt.year
    df['month'] = df['transaction_date'].dt.month
    df['day'] = df['transaction_date'].dt.day
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['num_days_ago'] = (df['date'].max() - df['date']).dt.days
    df['num_days_ago_sq'] = df['num_days_ago'] **2
    beds_encoded = pd.get_dummies(df['beds'], prefix='beds')
    df = pd.concat([df, beds_encoded], axis=1)
    df["floor"] = df['floor'].astype(int)
    df["unit"] = df['unit'].astype(int)
    df = df.drop(['year', 'month', 'day', 'date', 'beds','transaction_date'], axis=1)
    missing_columns = list(set(training_df.columns) - set(df.columns))
    df = df.assign(**{col: 0 for col in missing_columns})
    return df

  def generate_prediction_df(self, models, clf, df, columns, cutoff:int=10):
    data = []
    models.sort_values("R-Squared", inplace=True, ascending=False)
    for model in models.index.tolist()[:cutoff]:
      obj = clf.models.get(model).get_params()['regressor']
      obj.loss="quantile"
      obj.alpha=0.1
      data.append({model: clf.models.get(model).set_params(regressor=obj).predict(df)})
    return pd.DataFrame.from_dict({list(data[i].keys())[0]: list(data[i].values())[0] for i in range(len(data))}, orient='index', columns=columns)