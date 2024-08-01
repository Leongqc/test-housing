import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
import streamlit as st
import plotly.express as px
import folium 
from streamlit_folium import folium_static
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor

data_cleaned = pd.read_csv("cleaned_data.csv")



# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔍 About Us", "🏠 Introduction", "📊 EDA", "📈 Prediction"])

# About Us Page
if page == "🔍 About Us":
    st.title("About Us")
    
     # Display images
    #st.image("WhatsApp Image 2024-07-28 at 3.08.06 AM.jpeg", caption="John", use_column_width=True)
    #st.image("path_to_image2.jpg", caption="Team Member 2", use_column_width=True)
    
    st.markdown("""
    

    Welcome to our team behind the Property Price Prediction Tool! We are a dedicated group of data scientists, real estate experts, and technology enthusiasts committed to providing you with the most accurate and insightful property price predictions.

    ### Our Mission

    Our mission is to leverage cutting-edge technology and advanced data science techniques to empower individuals and businesses in the real estate market. We aim to deliver tools that simplify complex data and provide actionable insights for making informed decisions.

    ### Our Team

    
    

    - **John** - Data Scientist: With expertise in machine learning and statistical analysis, John leads the development of our predictive models.
    - **John** - Real Estate Expert: Jane's deep understanding of the real estate market helps us tailor our models to real-world conditions.
    - **Rikhi** - Software Engineer: Alex ensures that our tools are user-friendly and technically robust, providing seamless experiences for our users.
    - **John** - UI/UX Designer: Emily designs our interfaces to be intuitive and engaging, focusing on delivering a great user experience.

    ### Our Vision

    We envision a future where data-driven insights are accessible to everyone, enabling smarter decisions in the real estate market. By continuously improving our models and tools, we strive to be at the forefront of innovation in real estate analytics.

    Thank you for using our Property Price Prediction Tool. We are excited to be part of your journey in the real estate market!

    Feel free to reach out to us at [contact@ourteam.com](mailto:contact@ourteam.com) for any inquiries or feedback.

     ## Frequently Asked Questions (FAQs)

    **Q: How accurate is the property price prediction?**  
    A: Our tool uses advanced machine learning algorithms and comprehensive datasets to provide accurate predictions. However, please note that predictions are based on historical data and market conditions can change.

    **Q: How do you ensure data privacy?**  
    A: We prioritize data privacy and use industry-standard security measures to protect your data. Your information is never shared with third parties without your consent.

    ## Connect With Us

    Stay updated and connect with us on social media:
    - [Twitter](https://twitter.com/yourprofile)
    - [LinkedIn](https://linkedin.com/company/yourprofile)
    - [Facebook](https://facebook.com/yourprofile)

    ## Contact Us

    Have any questions or need assistance? Reach out to us at [contact@ourteam.com](mailto:contact@ourteam.com) or call us at (123) 456-7890.
    """)
    
   
    
    # Feedback Form
    with st.form(key='feedback_form'):
        st.subheader("Submit Your Feedback")

        # Feedback form fields
        name = st.text_input("Name")
        email = st.text_input("Email")
        feedback = st.text_area("Feedback", height=150)

        # Submit button
        submit_button = st.form_submit_button(label='Submit Feedback')

        if submit_button:
            # Optionally, you can add code here to handle the feedback submission,
            # such as saving it to a file or sending it via email.
            st.success("Thank you for your feedback! We appreciate your input.")
            
            # Here you can include code to save the feedback to a file or send an email.
            # Example:
            # with open('feedback.txt', 'a') as f:
            #     f.write(f"Name: {name}\nEmail: {email}\nFeedback: {feedback}\n\n")
            #     f.close()
            
# Introduction Page
if page == "🏠 Introduction":
    st.title("Welcome to Your Ultimate Property Price Prediction Tool!")
    st.markdown("""
    Are you ready to unlock the future of real estate with cutting-edge technology? Whether you're a seasoned investor, a first-time homebuyer, or a real     estate enthusiast, our state-of-the-art prediction model empowers you to make informed decisions with confidence.
    
    ## Why Property Price Prediction?

    In the vast and versatile toolbox of data science, we chose to focus on property price prediction due to its profound impact on individuals, businesses, and economies. Real estate is one of the most significant investments people make in their lifetimes, and accurate price predictions can provide substantial benefits:

    - **Informed Decision Making:** Accurate predictions help buyers and sellers make well-informed decisions, minimizing financial risks.
    - **Market Insights:** Understanding property trends aids investors in identifying lucrative opportunities and avoiding potential pitfalls.
    - **Economic Impact:** Real estate significantly influences the economy; our model provides data-driven insights to support economic growth and stability.

    ## What We Offer

    In an ever-changing real estate market, understanding property values is crucial. Our tool harnesses advanced machine learning algorithms to provide precise price predictions, tailored to your unique needs. By analyzing key factors such as location, property type, size, and more, we offer insights that help you navigate the complexities of the property market.

    ## Why Choose Us?

    - **Accuracy:** Leveraging the power of data science, our predictions are based on comprehensive datasets, ensuring reliability and precision.
    - **User-Friendly Interface:** Our streamlined, intuitive design makes it easy for anyone to use, regardless of technical expertise.
    - **Comprehensive Analysis:** Gain a deeper understanding of property trends with interactive visualizations and detailed reports.

    Explore the future of real estate with us and take the first step towards making smarter, data-driven property decisions. Welcome aboard!
    """)
# Path to your local MP4 file
    video_path = '18522098-hd_1280_720_30fps.mp4'

# Embed the video
    st.video(video_path)

   # List of locations with their coordinates
    locations = {
        'KLCC': [3.1578, 101.7123],
        'Dutamas': [3.1822, 101.6654],
        'Bukit Jalil': [3.0593, 101.6895],
        'Sri Petaling': [3.0671, 101.6901],
        'Mont Kiara': [3.1718, 101.6501],
        'Desa ParkCity': [3.1865, 101.6341],
        'Damansara Heights': [3.1412, 101.6618],
        'Bangsar South': [3.1124, 101.6654],
        'Ampang Hilir': [3.1532, 101.7475],
        'Jalan Klang Lama': [3.0925, 101.6748],
        'KL City': [3.1543, 101.7155],
        'Sungai Besi': [3.0645, 101.7162],
        'Setapak': [3.2061, 101.7271],
        'City Centre': [3.1554, 101.7140],
        'KL Sentral': [3.1328, 101.6851],
        'Taman Desa': [3.1026, 101.6857],
        'Sentul': [3.1890, 101.6896],
        'Segambut': [3.1890, 101.6628],
        'Wangsa Maju': [3.2041, 101.7372],
        'Batu Caves': [3.2360, 101.6841],
        'Bangsar': [3.1283, 101.6793],
        'Klcc': [3.1588, 101.7133],
        'Setiawangsa': [3.1814, 101.7351],
        'Chan Sow Lin': [3.1224, 101.7146],
        'Taman Tun Dr Ismail': [3.1518, 101.6292],
        'Jalan Kuching': [3.1864, 101.6752],
        'Cheras': [3.0958, 101.7405],
        'Ampang': [3.1589, 101.7615],
        'KL Eco City': [3.1173, 101.6757],
        'Bukit Bintang': [3.1466, 101.7093],
        'Kuchai Lama': [3.0922, 101.6862],
        'Kepong': [3.2126, 101.6370],
        'Bandar Menjalara': [3.2015, 101.6295],
        'Seputeh': [3.1188, 101.6832],
        'Jalan Ipoh': [3.1934, 101.6761],
        'Bukit Tunku': [3.1618, 101.6835],
        'Desa Pandan': [3.1468, 101.7375],
        'Desa Petaling': [3.0853, 101.7047],
        'Taman Melawati': [3.2117, 101.7512],
        'Pantai': [3.1136, 101.6633],
        'Sri Hartamas': [3.1665, 101.6575],
        'Sunway SPK': [3.1905, 101.6378],
        'Brickfields': [3.1291, 101.6838],
        'OUG': [3.0786, 101.6783],
        'Salak Selatan': [3.0905, 101.7044],
        'Titiwangsa': [3.1752, 101.7038],
        'Pandan Perdana': [3.1251, 101.7425],
        'Jalan Sultan Ismail': [3.1536, 101.7055],
        'Mid Valley City': [3.1180, 101.6744],
        'Keramat': [3.1620, 101.7378],
        'Jinjang': [3.2174, 101.6526],
        'Bandar Tasik Selatan': [3.0658, 101.7174],
        'Pandan Jaya': [3.1285, 101.7426],
        'Country Heights Damansara': [3.1801, 101.6301],
        'Bandar Damai Perdana': [3.0602, 101.7381],
        'Bukit Ledang': [3.1552, 101.6786],
        'Pandan Indah': [3.1266, 101.7494],
        'Puchong': [3.0350, 101.6199],
        'Damansara': [3.1368, 101.6168],
        'Sungai Penchala': [3.1591, 101.6211],
        'Taman Duta': [3.1593, 101.6752],
}

# Create a Streamlit app
    st.title('Interactive Map of Kuala Lumpur Locations')

# Define the map's starting location and zoom level
    map_location = [3.1390, 101.6869]  # Kuala Lumpur
    zoom_start = 12

# Create a Folium map
    m = folium.Map(location=map_location, zoom_start=zoom_start)

# Add markers for each location
    for location, coords in locations.items():
        folium.Marker(
            location=coords,
            popup=location,
            icon=folium.Icon(icon='info-sign')
        ).add_to(m)

# Display the map in Streamlit
    folium_static(m)
# EDA Page
if page == "📊 EDA":
    st.title("Exploratory Data Analysis (EDA)")

    if data_cleaned.empty:
        st.write("No data available for analysis.")
    else:
        st.title ("Average Property Price Heatmap")
         # Compute the average price for each location
        average_price_df = data_cleaned.groupby(['Latitude', 'Longitude', 'Location']).agg({'Price': 'mean'}).reset_index()

        # Format prices to two decimal places and convert to strings
        average_price_df['Price_str'] = average_price_df['Price'].apply(lambda x: f'{x:.2f}')

        # Create a scatter map with unique colors for each average price
        fig = px.scatter_mapbox(average_price_df, lat='Latitude', lon='Longitude', color='Price_str', size='Price',
                                mapbox_style="open-street-map", zoom=10, center={'lat': 3.1390, 'lon': 101.6869},
                                hover_name='Location', hover_data={'Price_str': False, 'Price': ':.2f', 'Latitude': False, 'Longitude': False},
                                title="Average Property Price Heatmap")

        # Customize layout to hide labels and other non-essential elements
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            title_font_size=24,
            title_font_family="Arial",
            margin=dict(l=0, r=0, t=0, b=0),
            geo=dict(
                showland=True,
                landcolor="white",
                showocean=True,
                oceancolor="lightblue",
                showcountries=True,
                countrycolor="black"
            )
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

        # Story-like description
        st.markdown("""

## Navigating Property Prices in Kuala Lumpur: Heatmap Insights

- **Heatmap Overview**: Visualizes average property prices across various locations.
- **Points on the Map**: Represent neighborhoods.
- **Color Coding**:
  - **Lighter Hues**: Indicate more affordable properties.
  - **Darker Shades**: Reflect higher property prices.
- **Point Size**: Corresponds to the price itself, providing a sense of scale.
- **Interactive Hover**:
  - **Discover Insights**: View detailed information about specific locations and their average property prices.

""")

        st.title("Scatter Plot of Top 5 Locations")
        # Calculate the top 5 locations by count
        top_5_locations = data_cleaned['Location'].value_counts().head(5).index

        # Filter the DataFrame to include only the top 5 locations
        filtered_df = data_cleaned[data_cleaned['Location'].isin(top_5_locations)]

        # Create the scatter plot
        fig = px.scatter(
            filtered_df, 
            x='Size', 
            y='Price', 
            color='Location', 
            size='Rooms',
            hover_data=['Location', 'Price', 'Size', 'Rooms'],
            title='Scatter Plot of Top 5 Locations'
        )

        # Update the layout to make the plot larger
        fig.update_layout(
            width=1200,  # Set width in pixels
            height=800,  # Set height in pixels
            title='Scatter Plot of Top 5 Locations'
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Enhanced story-like description with additional guidance
        st.markdown("""
### Unveiling the Top Property Markets

Next, we dive into the top 5 property markets. This scatter plot reveals how property size relates to price across these key locations. Each dot represents a property listing, with the color indicating the specific location and the size of the dot reflecting the number of rooms.

#### How to Read This Plot:
- **Location Insights:** Each color represents a different location. You can compare these colors to see how different areas stack up against each other.
- **Size vs. Price:** The x-axis shows the property size, while the y-axis indicates the price. By examining the position of each dot, you can determine how property size influences its price.
- **Dot Size:** Larger dots represent properties with more rooms, giving you a sense of the relationship between the number of rooms and the overall price.
- **Hover Details:** Hover over a dot to see detailed information about that property, including its exact price, size, and the number of rooms.

Explore the plot to uncover where the most sought-after properties are located and how their sizes and prices compare. Use this information to make more informed decisions about property investments or purchases.
""")
        st.title ("Price Distribution by Broad Property Type and Location")
        
        # Find the top 5 locations by count
        top_locations = data_cleaned['Location'].value_counts().nlargest(5).index

        # Filter the DataFrame to include only the top 5 locations
        df_top_locations = data_cleaned[data_cleaned['Location'].isin(top_locations)]

        # For each combination of location and broad property type, select the row with the highest price
        df_highest_price = df_top_locations.loc[df_top_locations.groupby(['Location', 'Broad Property Type'])['Price'].idxmax()]

        # Create the violin plot
        fig = px.violin(
            df_highest_price, 
            y='Price', 
            x='Broad Property Type', 
            color='Location', 
            box=True, 
            points='all',
            title='Price Distribution by Broad Property Type and Location'
        )

        # Update layout for better appearance
        fig.update_layout(
            width=1200,  # Set width in pixels
            height=800,  # Set height in pixels
            title='Price Distribution by Broad Property Type and Location',
            xaxis_title='Broad Property Type',
            yaxis_title='Price'
        )

        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

# Enhanced story-like description with a guide
        st.markdown("""
### Discovering Price Trends by Property Type

Our journey continues with a deeper look into price distributions by property type across top locations. This violin plot provides insights into how property prices vary for different types—whether it's an apartment, condo, or serviced residence. 

#### How to View This Plot:
- **Property Types:** The x-axis categorizes the properties into types such as Apartment, Condo, and Serviced Residence. Each category is represented by a distinct section on the plot.
- **Price Distribution:** The y-axis shows the price range. The shape of each violin indicates the distribution of prices within each property type. A wider section of the violin indicates a higher concentration of properties at that price range.
- **Color Coding:** Different colors represent different locations. This helps you compare how the price distributions for each property type vary across various locations.
- **Box Plot and Points:** The plot includes a box plot within each violin, providing a summary of the central tendency and spread of prices. Individual data points are also shown, allowing you to see the exact price of each property.
- **Insights:** By examining the shape and spread of the violins, you can identify which property types offer the best value and how they compare across different areas. Look for patterns or clusters that indicate more affordable or premium options.

Explore this plot to understand the price dynamics of various property types and how they differ by location. This will help you identify which property types are more cost-effective and where you might find the best deals.
""")
        st.title ("Average price for top 3 broad property types in top 10 locations")

         # Define the broad property types of interest
        property_types_of_interest = ['Apartment', 'Condo', 'Serviced Residence']

    # Create a dropdown for selecting property type
        selected_property_type = st.selectbox("Select Property Type", property_types_of_interest)
        
        # Filter data for the selected property type
        df_prop_type = data_cleaned[data_cleaned['Broad Property Type'] == selected_property_type]
        
        # Find the top 10 locations by count for the selected property type
        top_locations = df_prop_type['Location'].value_counts().nlargest(10).index
        
        # Filter the DataFrame to include only the top 10 locations
        df_top_locations = df_prop_type[df_prop_type['Location'].isin(top_locations)]
        
        # Calculate the average price for each combination of Location and Broad Property Type
        average_prices = df_top_locations.groupby(['Location'])['Price'].mean().reset_index()
        
        # Sort locations to make the line plot coherent
        average_prices = average_prices.sort_values(by='Location')
        
        # Create a line plot for the selected property type
        fig = px.line(average_prices, x='Location', y='Price', 
                      title=f"Average Price for {selected_property_type}",
                      labels={'Price': 'Average Price', 'Location': 'Location'},
                      markers=True)
        
        # Update the layout to make the figure larger
        fig.update_layout(width=1000, height=600)
        
        # Display the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)

# Story-like description
        st.markdown(f"""
### Tracking Average Prices for {selected_property_type}

Finally, we focus on {selected_property_type}s. This line plot illustrates how average prices vary across the top 10 locations for the selected property type. By following the trend line, you can identify where {selected_property_type}s are most and least expensive, helping you make informed decisions whether you're buying or investing. Select a property type from the dropdown to see detailed pricing trends and discover the best deals.
""")


# Prediction Page
if page == "📈 Prediction":
    st.title("Property Price Prediction")

   # Define unique values for dropdowns
    broad_property_types = data_cleaned['Broad Property Type'].unique()
    locations = data_cleaned['Location'].unique()
    furnishings = data_cleaned['Furnishing'].unique()

    # Dropdowns for user input
    broad_property_type = st.selectbox('Select Broad Property Type:', broad_property_types)
    location = st.selectbox('Select Location:', locations)
    furnishing = st.selectbox('Select Furnishing:', furnishings)

    # Sliders for numeric inputs based on dataset
    min_size, max_size = data_cleaned['Size'].min(), data_cleaned['Size'].max()
    size = st.slider('Size (in square feet):', min_value=int(min_size), max_value=int(max_size), value=int((min_size + max_size) / 2))

    min_rooms, max_rooms = data_cleaned['Rooms'].min(), data_cleaned['Rooms'].max()
    rooms = st.slider('Number of Rooms:', min_value=int(min_rooms), max_value=int(max_rooms), value=int((min_rooms + max_rooms) / 2))

    min_bathrooms, max_bathrooms = data_cleaned['Bathrooms'].min(), data_cleaned['Bathrooms'].max()
    bathrooms = st.slider('Number of Bathrooms:', min_value=int(min_bathrooms), max_value=int(max_bathrooms), value=int((min_bathrooms + max_bathrooms) / 2))

    min_car_parks, max_car_parks = data_cleaned['Car Parks'].min(), data_cleaned['Car Parks'].max()
    car_parks = st.slider('Number of Car Parks:', min_value=int(min_car_parks), max_value=int(max_car_parks), value=int((min_car_parks + max_car_parks) / 2))

    # Define features and target
    features = ['Size', 'Rooms', 'Bathrooms', 'Car Parks', 'Broad Property Type', 'Location', 'Furnishing']
    X = data_cleaned[features]
    y = data_cleaned['Price'] / data_cleaned['Size']  # Price per square foot

    # Convert features to numeric
    X = X.copy()
    X['Size'] = pd.to_numeric(X['Size'], errors='coerce')
    X['Rooms'] = pd.to_numeric(X['Rooms'], errors='coerce')
    X['Bathrooms'] = pd.to_numeric(X['Bathrooms'], errors='coerce')
    X['Car Parks'] = pd.to_numeric(X['Car Parks'], errors='coerce')

    # Handle categorical features
    categorical_features = ['Broad Property Type', 'Location', 'Furnishing']
    numeric_features = ['Size', 'Rooms', 'Bathrooms', 'Car Parks']

    # One-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create a pipeline with preprocessing and model training
    pipeline_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # Drop rows with missing values
    X = X.dropna()
    y = y[X.index]  # Ensure target and features have the same index

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline_lr.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_lr = pipeline_lr.predict(X_test)

    # Evaluate the model
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    #st.write(f"Linear Regression - Mean Squared Error: {mse_lr:.2f}")
    #st.write(f"Linear Regression - R² Score: {r2_lr:.2f}")

    # Button to trigger prediction and map generation
    if st.button("Press to Predict"):
        # Create a DataFrame for user input
        user_input = pd.DataFrame({
            'Size': [size],
            'Rooms': [rooms],
            'Bathrooms': [bathrooms],
            'Car Parks': [car_parks],
            'Broad Property Type': [broad_property_type],
            'Location': [location],
            'Furnishing': [furnishing]
        })

        user_input['Size'] = pd.to_numeric(user_input['Size'], errors='coerce')
        user_input['Rooms'] = pd.to_numeric(user_input['Rooms'], errors='coerce')
        user_input['Bathrooms'] = pd.to_numeric(user_input['Bathrooms'], errors='coerce')
        user_input['Car Parks'] = pd.to_numeric(user_input['Car Parks'], errors='coerce')

        # Predict price per square foot for the user input
        user_pred_price_per_sqft = pipeline_lr.predict(user_input)[0]
        user_pred_price = user_pred_price_per_sqft * size

        # Display predictions
        st.write(f"Predicted Price per Square Foot (RM): {user_pred_price_per_sqft:.2f}")
        st.write(f"Predicted Price (RM): {user_pred_price:.2f}")

        # Collect all unique locations
        unique_locations = data_cleaned['Location'].unique()

        # Initialize lists to collect prediction results
        location_data = []

        # Iterate through each unique location
        for loc in unique_locations:
            # Create a DataFrame for the current location
            loc_data = pd.DataFrame({
                'Size': [size],
                'Rooms': [rooms],
                'Bathrooms': [bathrooms],
                'Car Parks': [car_parks],
                'Broad Property Type': [broad_property_type],
                'Location': [loc],
                'Furnishing': [furnishing]
            })

            # Display loc_data for debugging
            loc_data['Size'] = pd.to_numeric(loc_data['Size'], errors='coerce')
            loc_data['Rooms'] = pd.to_numeric(loc_data['Rooms'], errors='coerce')
            loc_data['Bathrooms'] = pd.to_numeric(loc_data['Bathrooms'], errors='coerce')
            loc_data['Car Parks'] = pd.to_numeric(loc_data['Car Parks'], errors='coerce')

            # Predict price for the current location
            predicted_price = pipeline_lr.predict(loc_data) * size
            predicted_price_per_sqft = pipeline_lr.predict(loc_data)

            # Append result to the list
            location_data.append({
                'Location': loc,
                'Size': size,
                'Rooms': rooms,
                'Bathrooms': bathrooms,
                'Car Parks': car_parks,
                'Broad Property Type': broad_property_type,
                'Furnishing': furnishing,
                'Predicted Price': predicted_price[0],
                'Predicted Price per Sq Ft': predicted_price_per_sqft[0]
            })

        # Convert results to DataFrame
        prediction_df = pd.DataFrame(location_data)

        # Merge with coordinates
        prediction_df = prediction_df.merge(data_cleaned[['Location', 'Latitude', 'Longitude']], on='Location', how='left')

        # Find the latitude and longitude for the selected location
        user_location_coords = data_cleaned[data_cleaned['Location'] == location][['Latitude', 'Longitude']]

        # Check if user_location_coords is empty and handle the fallback
        if not user_location_coords.empty:
            user_lat = user_location_coords['Latitude'].values[0]
            user_lon = user_location_coords['Longitude'].values[0]
        else:
            # Define a fallback location
            fallback_location = data_cleaned['Location'].unique()[0]  # Choose a default location if needed
            fallback_coords = data_cleaned[data_cleaned['Location'] == fallback_location][['Latitude', 'Longitude']]
            user_lat = fallback_coords['Latitude'].values[0]
            user_lon = fallback_coords['Longitude'].values[0]

        # Create a map
        fig = px.scatter_mapbox(prediction_df,
                               lat="Latitude",
                               lon="Longitude",
                               color="Predicted Price",
                               hover_name="Location",
                               hover_data={
                                   "Size": True,
                                   "Rooms": True,
                                   "Bathrooms": True,
                                   "Car Parks": True,
                                   "Broad Property Type": True,
                                   "Furnishing": True,
                                   "Predicted Price": ":,.2f",  # Format as currency
                                   "Predicted Price per Sq Ft": ":,.2f"  # Format as currency
                               },
                               color_continuous_scale=px.colors.cyclical.IceFire,
                               title="Real Estate Price Predictions",
                               center={"lat": user_lat, "lon": user_lon},
                               zoom=10,
                               mapbox_style="open-street-map")

        # Highlight user's location
        if not user_location_coords.empty:
            fig.add_scattermapbox(lat=user_location_coords['Latitude'],
                                  lon=user_location_coords['Longitude'],
                                  mode='markers',
                                  marker=dict(size=10, color='red'),
                                  text='You are here',
                                  hoverinfo='text',
                                  name='Your Location')

        st.plotly_chart(fig)
