import pandas as pd
import streamlit as st
import numpy as np
import folium
from streamlit_folium import folium_static
import geopandas
from folium.plugins import MarkerCluster
import plotly.express as px
from datetime import date, datetime
from PIL import Image

st.set_page_config(layout='wide')

@st.cache( allow_output_mutation=True)

def get_data(path):
    data=pd.read_csv(path)

    return data

@st.cache(allow_output_mutation=True)
def get_geofile(url):
    geofile = geopandas.read_file(url)

    return geofile

def set_feature(data):
    #add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def load_image(logo):
    image = Image.open(logo)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(' ')

    with c2:
        st.image(image)

    with c3:
        st.write(' ')

def overview_data(data):


    st.sidebar.title('Data Overview')

    f_attributes = st.sidebar.multiselect('Enter columns', data.columns)
    f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].unique())

    st.title('Data Overview')

    if (f_zipcode != []) & (f_attributes != []):
        data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
    elif (f_zipcode != []) & (f_attributes == []):
        data = data.loc[data['zipcode'].isin(f_zipcode), :]
    elif (f_zipcode == []) & (f_attributes != []):
        data = data.loc[:, f_attributes]
    else:
        data = data.copy()

    st.dataframe(data)

    c1, c2 = st.columns((1, 1))
    # Average metrics

    df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    df4 = data[['price_m2', 'zipcode']].groupby('zipcode').mean().reset_index()

    # merge
    m1 = pd.merge(df1, df2, on='zipcode', how='inner')
    m2 = pd.merge(m1, df3, on='zipcode', how='inner')
    df = pd.merge(m2, df4, on='zipcode', how='inner')

    df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING', 'PRICE/M2']

    c1.header('Average Values')
    c1.dataframe(df, height=600)

    num_attributes = data.select_dtypes(include=['int64', 'float64'])
    mean_ = pd.DataFrame(num_attributes.apply(np.mean))
    median_ = pd.DataFrame(num_attributes.apply(np.median))
    std_ = pd.DataFrame(num_attributes.apply(np.std))

    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    df1 = pd.concat([max_, min_, mean_, median_, std_], axis=1).reset_index()
    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std']

    c2.header('Descriptive Analysis')
    c2.dataframe(df1, height=600)
    return None

def portifolio_density(data, geofile):
    st.title('Region Overview')

    c1, c2 = st.columns([2.2,2])
    c1.header('Portfolio Density')

    df = data.sample(1000)

    # Base Map - Folium

    density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                             default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup='Price R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format(
                          row['price'],
                          row['date'],
                          row['sqft_living'],
                          row['bedrooms'],
                          row['bathrooms'],
                          row['yr_built']
                      )).add_to(marker_cluster)

    with c1:
        folium_static(density_map)

    # Region price Map

    c2.header('Price Density')

    df = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']



    # df = df.sample(10)

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].tolist())]

    region_price_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()],
                                  default_zoom_start=15)

    folium.Choropleth(data=df,
                                geo_data=geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity=0.7,
                                line_opacity=0.2,
                                legend_name='AVG PRICE').add_to(region_price_map)
    with c2:
        folium_static(region_price_map)
    return None

def commercial_distribution(data):

    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

    min_year_built = int(data['yr_built'].min())
    max_year_built = int(data['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_year_built = st.sidebar.slider('Year Built', min_year_built, max_year_built, min_year_built)

    st.header('Average Price per Year built')

    # data selection
    df = data.loc[data['yr_built'] < f_year_built]
    df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line(df, x='yr_built', y='price')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Average Price per day')
    st.sidebar.subheader('Select Max Date')

    # Filters
    min_date = datetime.strptime(data['date'].min(), '%Y-%m-%d')
    max_date = datetime.strptime(data['date'].max(), '%Y-%m-%d')

    f_date = st.sidebar.slider('Date', min_date, max_date, min_date)

    # data filtering

    data['date'] = pd.to_datetime(data['date'])

    df = data.loc[data['date'] < f_date]

    df = df[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line(df, x='date', y='price')
    st.plotly_chart(fig, use_container_width=True)

    # Histograma
    st.header('Price Distribution')
    st.sidebar.subheader('Select Max Price')

    # Filter

    min_price = int(data['price'].min())
    max_price = int(data['price'].max())
    avg_price = int(data['price'].mean())

    # Data filtering
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price)
    df = data.loc[data['price'] < f_price]

    # data plot
    fig = px.histogram(df, x='price', nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    return None

def attributes_distribution(data):
    st.sidebar.title('Attributes Options')
    st.title('House Attributes')

    # filters
    f_bedrooms = st.sidebar.selectbox('Max number of bedrooms', sorted(set(data['bedrooms'].unique())))
    f_bathrooms = st.sidebar.selectbox('Max number of bathrooms', sorted(set(data['bathrooms'].unique())))

    c1, c2 = st.columns(2)

    # House per bedrooms
    c1.header('Houses per bedrooms')
    df = data[data['bedrooms'] < f_bedrooms]
    fig = px.histogram(df, x='bedrooms', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House pr bathrooms
    c2.header('Houses per bathrooms')
    df = data[data['bathrooms'] < f_bathrooms]
    fig = px.histogram(df, x='bathrooms', nbins=19)
    c2.plotly_chart(fig, use_container_width=True)

    # filters

    f_floors = st.sidebar.selectbox('Max number of floor', sorted(set(data['floors'].unique())))
    f_waterview = st.sidebar.checkbox('Only Houses with Water View')

    c1, c2 = st.columns(2)

    # House per floors
    c1.header('Houses per floor')
    df = data[data['floors'] < f_floors]

    # plot

    fig = px.histogram(df, x='floors', nbins=19)
    c1.plotly_chart(fig, use_container_width=True)

    # House per water view

    c2.header('Houses overlooking water')

    if f_waterview:
        df = data[data['waterfront'] == 1]
    else:
        df = data.copy()
    fig = px.histogram(df, x='waterfront', nbins=10)
    c2.plotly_chart(fig, use_container_width=True)

    return None

def buy_properties(data):



    st.sidebar.title('Properties to buy and selling price')
    st.sidebar.header('Indications for buying and selling')

    f_buy_properties = st.sidebar.checkbox ('Properties that must be purchased')

    df_median = data[['zipcode', 'price']].groupby('zipcode').median().reset_index()
    df_median.columns = ['zipcode', 'price_median']
    df_gropy = pd.merge(data, df_median, on='zipcode', how='inner')

    df_gropy['buy'] = 'no'
    df_gropy['buy'] = df_gropy[['buy', 'price', 'price_median', 'condition']].apply(
        lambda x: 'yes' if (x['price'] < x['price_median']) & (x['condition'] <= 3) else 'no', axis=1)

    df_gropy['month'] = pd.to_datetime(df_gropy['date']).dt.strftime('%m').astype('int64')
    df_gropy['day'] = pd.to_datetime(df_gropy['date']).dt.strftime('%d').astype('int64')

    df_gropy_sell = df_gropy.copy()

    def season(day, month):
        if month in (1, 2):
            return 'SUMMER'
        elif month == 3:
            if day < 21:
                return 'SUMMER'
            else:
                return 'AUTUMN'
        elif month in (4, 5):
            return 'AUTUMN'
        elif month == 6:
            if day < 21:
                return 'AUTUMN'
            else:
                return 'WINTER'
        elif month in (7, 8):
            return 'WINTER'
        elif month == 9:
            if day < 21:
                return 'WINTER'
            else:
                return 'SPRING'
        elif month in (10, 11):
            return 'SPRING'
        elif month == 12:
            if day < 21:
                return 'SPRING'
            else:
                return 'SUMMER'

    df_gropy_sell['season_median'] = list(map(season, df_gropy_sell['day'], df_gropy_sell['month']))

    df_median_2 = df_gropy_sell[['zipcode', 'season_median', 'price']].groupby(
        ['zipcode', 'season_median']).median().reset_index()

    df_gropy_sell = pd.merge(df_gropy_sell, df_median_2, on='zipcode', how='inner')
    df_gropy_sell = df_gropy_sell.drop('season_median_x', axis=1)
    df_gropy_sell = df_gropy_sell.rename(
        columns={'price_y': 'price_season', 'price_x': 'price_buy', 'season_median_y': 'season_median'})

    df_gropy_sell['price_sell'] = df_gropy_sell[['price_buy', 'price_season']].apply(
        lambda x: x['price_buy'] * 1.30 if x['price_buy'] < x['price_season'] else x['price_buy'] * 1.10, axis=1)

    if f_buy_properties:
        df_buy = df_gropy[df_gropy['buy']=='yes']
        df_sell = df_gropy_sell[df_gropy_sell['buy'] == 'yes']
    else:
        df_buy = df_gropy.copy()
        df_sell = df_gropy_sell.copy()

    df_buy = df_buy.rename(columns={'price':'price_buy'})

    st.title('Properties to buy and selling price')

    c1,c2 = st.columns(2)
    c1.header('Properties for buying')
    c2.header('Price for selling')

    c1.dataframe(df_buy[['id', 'price_buy', 'zipcode', 'condition', 'price_median', 'buy']])


    c2.dataframe(df_sell[['id', 'price_buy', 'season_median','price_season', 'price_sell', 'condition', 'zipcode']])

    return None


if __name__ == '__main__':
    #Data extraction
    path = 'kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    logo = 'house_hocket_logo.png'

    data = get_data(path)
    geofile = get_geofile(url)

    #transformation
    data = set_feature(data)

    load_image(logo)
    overview_data(data)
    portifolio_density(data, geofile)
    commercial_distribution(data)
    attributes_distribution(data)
    buy_properties(data)









