import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.graph_objs as go
import altair as alt








# Load data
df = pd.read_csv("FAO.csv", encoding="ISO-8859-1")
df_pop = pd.read_csv("FAOSTAT_data_6-13-2019.csv")
df_area = pd.read_csv("countries_area_2013.csv")

# Function to prepare the data for a specific area
def prepare_data_for_area(area_name):
    # Step 2: Prepare the data
    d3 = df.loc[:, 'Y1993':'Y2013']  # Mengambil hanya 20 tahun terakhir
    data1 = new_data.join(d3)  # Menggabungkan data new_data dengan d3

    # Step 3: Mendapatkan data hanya untuk 'Food'
    d4 = data1.loc[data1['Element'] == 'Food']  # Memilih baris yang 'Element' nya adalah 'Food'
    d5 = d4.drop('Element', axis=1)  # Menghapus kolom 'Element'
    d5 = d5.fillna(0)  # Menggantikan nilai yang hilang dengan 0

    # Membuat daftar tahun
    year_list = list(d3.columns)

    # Mendapatkan data untuk area yang dipilih
    selected_area = d4[d4['Area'] == area_name]
    selected_area_total = selected_area.groupby('Item')[year_list].sum()  # Menjumlahkan semua tahun untuk setiap item
    selected_area_total['Total'] = selected_area_total.sum(axis=1)  # Menambahkan kolom 'Total' yang merupakan jumlah dari semua tahun
    selected_area_total = selected_area_total.reset_index()

    return selected_area_total

# Data consistency adjustments
df['Area'] = df['Area'].replace(['Swaziland'], 'Eswatini')
df['Area'] = df['Area'].replace(['The former Yugoslav Republic of Macedonia'], 'North Macedonia')

# Load new data
df_pop = pd.read_csv("FAOSTAT_data_6-13-2019.csv")
df_area = pd.read_csv("countries_area_2013.csv")
df_pop = pd.DataFrame({'Area': df_pop['Area'], 'Population': df_pop['Value']})
df_area = pd.DataFrame({'Area': df_area['Area'], 'Surface': df_area['Value']})

# Add missing line using pd.concat
missing_line = pd.DataFrame({'Area': ['Sudan'], 'Surface': [1886]})
df_area = pd.concat([df_area, missing_line], ignore_index=True)

# Merge tables
d1 = pd.DataFrame(df.loc[:, ['Area', 'Item', 'Element']])
data = pd.merge(d1, df_pop, on='Area', how='left')
new_data = pd.merge(data, df_area, on='Area', how='left')

d2 = df.loc[:, 'Y1961':'Y2013']
data = new_data.join(d2)

d3 = df.loc[:, 'Y1993':'Y2013']  # take only last 20 years
data1 = new_data.join(d3)  # recap: new_data does not contains years data

d4 = data1.loc[data1['Element'] == 'Food']  # get just food
d5 = d4.drop('Element', axis=1)
d5 = d5.fillna(0).infer_objects(copy=False)  # substitute missing values with 0 and infer types

year_list = list(d3.iloc[:, :].columns)
d6 = d5.pivot_table(values=year_list, index=['Area'], aggfunc='sum')

italy = d4[d4['Area'] == 'Italy']
italy = italy.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
italy = pd.DataFrame(italy.to_records())

item = d5.pivot_table(values=year_list, index=['Item'], aggfunc='sum')
item = pd.DataFrame(item.to_records())

d5 = d5.pivot_table(values=year_list, index=['Area', 'Population', 'Surface'], aggfunc='sum')
area = pd.DataFrame(d5.to_records())
d6.loc[:, 'Total'] = d6.sum(axis=1)
d6 = pd.DataFrame(d6.to_records())
d = pd.DataFrame({'Area': d6['Area'], 'Total': d6['Total'], 'Population': area['Population'], 'Surface': area['Surface']})









# RANKING

# Process data
year_list = list(df.iloc[:,10:].columns)
df_new = df.pivot_table(values=year_list, columns='Element', index=['Area'], aggfunc='sum') #for each country sum over years separatly Food&Feed
df_fao = df_new.T

# Producer of just Food
df_food = df_fao.xs('Food', level=1, axis=0)
df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()

# Producer of just Feed
df_feed = df_fao.xs('Feed', level=1, axis=0)
df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()

# Rank of most Produced Items
df_item = df.pivot_table(values=year_list, columns='Element', index=['Item'], aggfunc='sum')
df_item = df_item.T

# FOOD
df_food_item = df_item.xs('Food', level=1, axis=0)
df_food_item_tot = df_food_item.sum(axis=1).sort_values(ascending=False).head()  # sum across rows

# FEED
df_feed_item = df_item.xs('Feed', level=1, axis=0)
df_feed_item_tot = df_feed_item.sum(axis=1).sort_values(ascending=False).head()  # sum across rows

# Streamlit application
st.title('🥙 Food Data Analysis & Clustering')

st.write('### Top 5 Food & Feed Producer')
df_fao_tot = df_new.T.sum(axis=0).sort_values(ascending=False).head()
st.bar_chart(df_fao_tot)

st.write('### Top 5 Food Producer')
df_food_tot = df_food.sum(axis=0).sort_values(ascending=False).head()
st.bar_chart(df_food_tot)

st.write('### Top 5 Feed Producer')
df_feed_tot = df_feed.sum(axis=0).sort_values(ascending=False).head()
st.bar_chart(df_feed_tot)

st.write('### Top 5 Food Produced Item')
df_food_item_tot = df_food_item.sum(axis=0).sort_values(ascending=False).head()
st.bar_chart(df_food_item_tot)

st.write('### Top 5 Feed Produced Item')
df_feed_item_tot = df_feed_item.sum(axis=0).sort_values(ascending=False).head()
st.bar_chart(df_feed_item_tot)








# CLUSTERING 1 - DBSCAN

# Step 2: Prepare the data for clustering
X = pd.DataFrame({'Area': d['Area'], 'Total': d['Total'], 'Surface': d['Surface'], 'Population': d['Population']})

# Input parameters for DBSCAN clustering from user using Streamlit sidebar
eps = st.sidebar.number_input("Enter eps:", min_value=0.1, max_value=1.0, step=0.1, value=0.5)
min_samples = st.sidebar.number_input("Enter min_samples:", min_value=1, max_value=10, step=1, value=2)

# Preprocessing: Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['Total', 'Surface', 'Population']])

# Function to perform DBSCAN clustering
def DBSCAN_Clustering(X_scaled, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    return clusters

# Function to plot 3D clustering using Plotly for interactivity
def Plot3dClustering(X, clusters):
    fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                        labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                        title='3D Scatter plot of DBSCAN Clustering',
                        color_continuous_scale='Plasma')
    fig.update_layout(legend_title="Clusters")
    st.plotly_chart(fig)

# Perform DBSCAN clustering
clusters = DBSCAN_Clustering(X_scaled, eps, min_samples)

# Plot 3D clustering using Plotly in Streamlit
st.title('3D Scatter Plot of DBSCAN Clustering')
Plot3dClustering(X, clusters)

# Display clusters details
st.subheader('Cluster Details')
X['Cluster'] = clusters
for cluster_label in np.unique(clusters):
    if cluster_label == -1:
        cluster_name = 'Noise'
    else:
        cluster_name = f'Cluster {cluster_label}'
    st.write(f"{cluster_name}:")
    cluster_data = X[X['Cluster'] == cluster_label][['Area', 'Total', 'Population', 'Surface']]
    st.dataframe(cluster_data)
    st.write("\n")








# CLUSTERING 2 - KMEANS

# Input number of clusters from user using Streamlit sidebar
num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, max_value=10, step=1)

# Function to perform K-Means clustering
def K_Means(X, n):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=7, random_state=0)
    model.fit(X_scaled)
    clust_labels = model.predict(X_scaled)
    cent = model.cluster_centers_
    return clust_labels, cent

# Function to plot 3D clustering using Plotly for interactivity
def Plot3dClusteringKMeans(X, clusters):
    fig = px.scatter_3d(X, x='Total', y='Population', z='Surface', color=clusters,
                        labels={'Total': 'Total Production', 'Population': 'Population', 'Surface': 'Surface Area'},
                        title=f'3D Clustering with {num_clusters} clusters (K-Means)',
                        color_continuous_scale='Plasma')
    fig.update_layout(legend_title="Clusters")
    st.plotly_chart(fig)

# Preprocessing: Ensure only numeric data is used
X_numeric = X[['Total', 'Surface', 'Population']].copy()

# Elbow Method to determine optimal number of clusters
wcss = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=7, random_state=0)
    kmeans.fit(X_numeric)
    wcss.append(kmeans.inertia_)

# Perform K-Means clustering
clust_labels_kmeans, cent = K_Means(X_numeric, num_clusters)

# Adding K-Means cluster labels to the DataFrame
X['KMeans_Cluster'] = clust_labels_kmeans

# Plotting K-Means clustering in 3D using Plotly in Streamlit
st.title(f'3D Scatter Plot of K-Means Clustering with {num_clusters} clusters')
Plot3dClusteringKMeans(X, 'KMeans_Cluster')

# Display K-Means cluster details
st.subheader('Cluster Details (K-Means)')
for i in range(num_clusters):
    st.write(f"Cluster {i}:\n")
    cluster_kmeans = X[X['KMeans_Cluster'] == i][['Area', 'Total', 'Population', 'Surface']]
    st.dataframe(cluster_kmeans)
    st.write("\n")

# Display the Elbow Method graph using Streamlit
st.write('### Elbow Method for Optimal K in K-Means')
st.line_chart(wcss)








# CLUSTERING 2 - DBSCAN

# Streamlit app
st.title('Clustering Production of Food Items with DBScan')

# User input for the area name
area_name = st.sidebar.text_input("Enter the area name for clustering:")

if area_name:
    # Prepare the data for the selected area
    area_total = prepare_data_for_area(area_name)

    # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
    st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
    st.write(area_total[['Item', 'Total']])

    # Step 2: Prepare the data
    Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

    # Step 3: Preprocessing
    label_encoder = LabelEncoder()
    Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

    Y_scaled = Y[['Total']]

    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_scaled)

    # Step 4: Perform clustering using DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=2)  # Adjust parameters as needed
    clusters = dbscan.fit_predict(Y_scaled)

    # Step 5: Adding the cluster labels to the dataframe
    Y['Cluster'] = clusters

    # Output items and their clusters
    clustered_items = {}
    for cluster in np.unique(clusters):
        items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
        if cluster == -1:
            cluster_label = 'Noise'
        else:
            cluster_label = f'Cluster {cluster + 1}'
        clustered_items[cluster_label] = items_in_cluster

    st.write("Hasil Clustering:")
    for cluster_label, items in clustered_items.items():
        st.write(f"\n{cluster_label}:")
        st.write(items)

    # Find the best item to produce in each cluster
    st.write(f"#### Conclusion for each cluster")
    for cluster_label, items in clustered_items.items():
        best_item = items.loc[items['Total'].idxmax()]
        st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

    # Step 6: Visualize the results in a scatter plot with different colors for each cluster
    unique_clusters = np.unique(clusters)
    colors = px.colors.qualitative.Plotly

    # Plotting the clusters with different colors for each cluster
    fig = go.Figure()
    for i, cluster in enumerate(unique_clusters):
        cluster_data = Y[Y['Cluster'] == cluster]
        color = 'black' if cluster == -1 else colors[i % len(colors)]
        label = 'Noise' if cluster == -1 else f'Cluster {cluster + 1}'
        fig.add_trace(go.Scatter(
            x=cluster_data['Item_encoded'],
            y=cluster_data['Total'],
            mode='markers',
            marker=dict(color=color),
            name=label,
            text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],  # Add hover text
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f'Clustered Production of Food Items in {area_name}',
        xaxis_title='Item',
        yaxis_title='Total Production',
        xaxis=dict(
            tickmode='array',
            tickvals=Y['Item_encoded'],
            ticktext=Y['Item']
        ),
        width=1000,  # Set the width of the figure
        height=500  # Set the height of the figure
    )
    st.plotly_chart(fig)








# CLUSTERING 2 - KMEANS

# Streamlit app
st.title('Clustering Production of Food Items with K-Means')

# User input for the area name and number of clusters
area_name = st.sidebar.text_input("Enter the name of the country:")
num_clusters = st.sidebar.number_input("Enter the number of clusters:", min_value=1, value=3)

if area_name:
    # Prepare the data for the selected area
    area_total = prepare_data_for_area(area_name)

    # Menampilkan total produksi per item dari tahun 1993 hingga 2013 di area yang dipilih
    st.write(f"Total produksi per item dari tahun 1993 hingga 2013 di {area_name}:")
    st.write(area_total[['Item', 'Total']])

    # Step 2: Prepare the data
    Y = pd.DataFrame({'Item': area_total['Item'], 'Total': area_total['Total']})

    # Step 3: Preprocessing
    label_encoder = LabelEncoder()
    Y['Item_encoded'] = label_encoder.fit_transform(Y['Item'])

    Y_scaled = Y[['Total']]

    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_scaled)

    # Step 4: Perform clustering using K-means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    clusters = kmeans.fit_predict(Y_scaled)

    # Step 5: Adding the cluster labels to the dataframe
    Y['Cluster'] = clusters

    # Output items and their clusters
    clustered_items = {}
    for cluster in np.unique(clusters):
        items_in_cluster = Y[Y['Cluster'] == cluster][['Item', 'Total']]
        cluster_label = f'Cluster {cluster + 1}'
        clustered_items[cluster_label] = items_in_cluster

    for cluster_label, items in clustered_items.items():
        st.write(f"\n{cluster_label}:")
        st.write(items)

    # Find the best item to produce in each cluster
    st.write(f"#### Conclusion for each cluster")
    for cluster_label, items in clustered_items.items():
        best_item = items.loc[items['Total'].idxmax()]
        st.write(f"The best item to produce in {cluster_label} is {best_item['Item']} with a total production of {best_item['Total']}.")

    # Step 6: Visualize the results in a 2D scatter plot
    unique_clusters = np.unique(clusters)
    colors = px.colors.qualitative.Plotly

    # Plotting the clusters with different colors for each cluster
    fig = go.Figure()
    for i, cluster in enumerate(unique_clusters):
        cluster_data = Y[Y['Cluster'] == cluster]
        color = 'black' if cluster == -1 else colors[i % len(colors)]
        label = f'Cluster {cluster + 1}'
        fig.add_trace(go.Scatter(
            x=cluster_data['Item_encoded'],
            y=cluster_data['Total'],
            mode='markers',
            marker=dict(color=color),
            name=label,
            text=[f'Item: {item}<br>Total: {total}' for item, total in zip(cluster_data['Item'], cluster_data['Total'])],  # Add hover text
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f'Clustered Production of Food Items in {area_name}',
        xaxis_title='Item',
        yaxis_title='Total Production',
        xaxis=dict(
            tickmode='array',
            tickvals=Y['Item_encoded'],
            ticktext=Y['Item']
        ),
        width=1200,  # Set the width of the figure
        height=600  # Set the height of the figure
    )
    st.plotly_chart(fig)








# CLUSTERING 3 - DBSCAN

# Select necessary columns
production_columns = ['Y' + str(year) for year in range(1961, 2013)]
selected_columns = ['Area', 'Item', 'Element', 'latitude', 'longitude'] + production_columns
production_data = df[selected_columns].copy()
        
# Get unique items for item selection
items = production_data['Item'].unique()
        
# Sidebar for item selection
st.sidebar.header('Choose Item Type')
item_type = st.sidebar.selectbox("Select item type for clustering", items)
        
# Subset data for selected item and 'Food' element
subset_data = production_data[(production_data['Item'] == item_type) & (production_data['Element'] == 'Food')].copy()
        
# Calculate total production from 1961 to 2013
subset_data['total_production_1961_2013'] = subset_data[production_columns].sum(axis=1)
        
# Features for clustering: latitude, longitude, total production
X = subset_data[['latitude', 'longitude', 'total_production_1961_2013']].values
        
# Standardize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
        
# DBSCAN parameters (can be adjusted)
eps = st.sidebar.slider("Enter eps value", min_value=0.1, max_value=2.0, step=0.1, value=0.5)
min_samples = st.sidebar.slider("Enter min_samples value", min_value=1, max_value=10, step=1, value=2)
        
# Perform clustering with DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)
        
# Add cluster labels to the subset data
subset_data['cluster_label'] = clusters
        
# Separate clustered data and noise points
clustered = subset_data[subset_data['cluster_label'] != -1]
noise = subset_data[subset_data['cluster_label'] == -1]
unique_labels = np.unique(clusters)

# Plotting the clustering result in 3D scatter plot with Plotly
fig = px.scatter_3d(clustered, x='longitude', y='latitude', z='total_production_1961_2013',
                    color='cluster_label', opacity=0.8, size_max=15,
                    title=f'3D Scatter Plot with DBSCAN Clustering for {item_type}',
                    color_continuous_scale='Plasma')
# Add noise points to the plot
fig.add_trace(px.scatter_3d(noise, x='longitude', y='latitude', z='total_production_1961_2013',
                                    color_discrete_sequence=['black'], symbol='cluster_label',
                                    opacity=0.8, size_max=15).data[0])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# Show plot in Streamlit
st.plotly_chart(fig)

# Fungsi untuk mencari area terbaik di setiap cluster
def find_best_areas(clustered_data):
    unique_labels = clustered_data['cluster_label'].unique()
    results = []
    
    for label in unique_labels:
        if label != -1:
            cluster_data = clustered_data[clustered_data['cluster_label'] == label]
            max_production = cluster_data['total_production_1961_2013'].max()
            best_area = cluster_data[cluster_data['total_production_1961_2013'] == max_production].iloc[0]
            result = {
                'cluster_label': label,
                'best_area': best_area['Area'],
                'total_production': best_area['total_production_1961_2013']
            }
            results.append(result)
    
    return results

        
# Display data points in noise and each cluster
st.header("Data Points in Noise:")
st.write(noise[['Area', 'latitude', 'longitude', 'total_production_1961_2013']])

for label in unique_labels:
    if label != -1:
        st.header(f"Data Points in Cluster {label}:")
        cluster_data = clustered[clustered['cluster_label'] == label]
        st.write(cluster_data[['Area', 'latitude', 'longitude', 'total_production_1961_2013']])

        results = find_best_areas(cluster_data)

        for result in results:
            st.write(f"The best area to produce in Cluster {result['cluster_label']} is {result['best_area']} and products with a total production of {result['total_production']}.")