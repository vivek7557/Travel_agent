import streamlit as st
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.weather_agent import WeatherAgent
from agents.recommendations_agent import RecommendationsAgent

# Set page config
st.set_page_config(
    page_title="Travel Agent AI",
    page_icon="✈️",
    layout="wide"
)

# Title
st.title("🤖 AI Travel Agent")
st.subheader("Your personal travel planning assistant")

# Input form
with st.form("travel_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.text_input("Origin City", "New York")
        destination = st.text_input("Destination", "Paris")
    
    with col2:
        departure_date = st.date_input("Departure Date")
        return_date = st.date_input("Return Date")
    
    budget = st.slider("Budget ($)", 500, 10000, 2000)
    travelers = st.slider("Number of Travelers", 1, 10, 2)
    
    submitted = st.form_submit_button("Find Best Travel Options")

if submitted:
    with st.spinner("Searching for the best travel options..."):
        # Initialize agents
        flight_agent = FlightAgent()
        hotel_agent = HotelAgent()
        weather_agent = WeatherAgent()
        recommendations_agent = RecommendationsAgent()
        
        # Get flight information
        try:
            flights = flight_agent.search_flights(origin, destination, str(departure_date))
            st.subheader("✈️ Flight Options")
            if flights:
                for flight in flights[:3]:  # Show top 3
                    st.write(f"**Airline:** {flight.get('airline', 'N/A')}")
                    st.write(f"**Price:** ${flight.get('price', 'N/A')}")
                    st.write(f"**Duration:** {flight.get('duration', 'N/A')}")
                    st.write("---")
            else:
                st.write("No flights found.")
        except Exception as e:
            st.error(f"Error fetching flights: {str(e)}")
        
        # Get hotel information
        try:
            hotels = hotel_agent.search_hotels(destination, str(departure_date))
            st.subheader("🏨 Hotel Options")
            if hotels:
                for hotel in hotels[:3]:  # Show top 3
                    st.write(f"**Hotel:** {hotel.get('name', 'N/A')}")
                    st.write(f"**Price/Night:** ${hotel.get('price', 'N/A')}")
                    st.write(f"**Rating:** {hotel.get('rating', 'N/A')}")
                    st.write("---")
            else:
                st.write("No hotels found.")
        except Exception as e:
            st.error(f"Error fetching hotels: {str(e)}")
        
        # Get weather information
        try:
            weather = weather_agent.get_weather(destination)
            st.subheader("🌤️ Weather Forecast")
            if weather:
                st.write(f"Weather in {destination}: {weather}")
            else:
                st.write("Weather information not available.")
        except Exception as e:
            st.error(f"Error fetching weather: {str(e)}")
        
        # Get recommendations
        try:
            recommendations = recommendations_agent.get_recommendations(destination)
            st.subheader("🗺️ Recommendations")
            if recommendations:
                for rec in recommendations[:5]:  # Show top 5
                    st.write(f"- {rec}")
            else:
                st.write("No recommendations available.")
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")