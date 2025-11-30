"""
Main entry point for the Travel Agent application
"""
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.weather_agent import WeatherAgent
from agents.recommendations_agent import RecommendationsAgent
from agents.price_comparator import PriceComparator

def main():
    print("Travel Agent System Initialized")
    
    # Initialize all agents
    flight_agent = FlightAgent()
    hotel_agent = HotelAgent()
    weather_agent = WeatherAgent()
    recommendations_agent = RecommendationsAgent()
    price_comparator = PriceComparator()
    
    print("All agents initialized successfully")
    
    # Example usage
    destination = input("Enter your destination: ")
    travel_date = input("Enter travel date (YYYY-MM-DD): ")
    
    # Get flight information
    flights = flight_agent.search_flights("NYC", destination, travel_date)
    print(f"Found {len(flights)} flights")
    
    # Get hotel information
    hotels = hotel_agent.search_hotels(destination, travel_date)
    print(f"Found {len(hotels)} hotels")
    
    # Get weather information
    weather = weather_agent.get_weather(destination)
    print(f"Weather in {destination}: {weather}")
    
    # Get recommendations
    recommendations = recommendations_agent.get_recommendations(destination)
    print(f"Recommendations for {destination}: {recommendations}")
    
    # Compare prices
    best_option = price_comparator.compare_prices(flights, hotels)
    print(f"Best value option: {best_option}")

if __name__ == "__main__":
    main()