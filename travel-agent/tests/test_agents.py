import unittest
from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.weather_agent import WeatherAgent
from agents.recommendations_agent import RecommendationsAgent
from agents.price_comparator import PriceComparator

class TestFlightAgent(unittest.TestCase):
    def setUp(self):
        self.flight_agent = FlightAgent()
    
    def test_search_flights(self):
        # Test that search_flights returns a list
        flights = self.flight_agent.search_flights("NYC", "LAX", "2023-12-25")
        self.assertIsInstance(flights, list)
        
        # If flights are returned, check they have expected keys
        if flights:
            flight = flights[0]
            expected_keys = ['airline', 'price', 'duration', 'departure', 'arrival', 'stops']
            for key in expected_keys:
                self.assertIn(key, flight)

class TestHotelAgent(unittest.TestCase):
    def setUp(self):
        self.hotel_agent = HotelAgent()
    
    def test_search_hotels(self):
        # Test that search_hotels returns a list
        hotels = self.hotel_agent.search_hotels("Paris", "2023-12-25")
        self.assertIsInstance(hotels, list)
        
        # If hotels are returned, check they have expected keys
        if hotels:
            hotel = hotels[0]
            expected_keys = ['name', 'price', 'rating', 'location', 'amenities']
            for key in expected_keys:
                self.assertIn(key, hotel)

class TestWeatherAgent(unittest.TestCase):
    def setUp(self):
        self.weather_agent = WeatherAgent()
    
    def test_get_weather(self):
        # Test that get_weather returns a dictionary
        weather = self.weather_agent.get_weather("New York")
        self.assertIsInstance(weather, dict)
        
        # Check that it has expected keys
        if weather:
            expected_keys = ['location', 'temperature', 'condition', 'humidity', 'wind_speed']
            for key in expected_keys:
                self.assertIn(key, weather)

class TestRecommendationsAgent(unittest.TestCase):
    def setUp(self):
        self.recommendations_agent = RecommendationsAgent()
    
    def test_get_recommendations(self):
        # Test that get_recommendations returns a list
        recommendations = self.recommendations_agent.get_recommendations("Paris")
        self.assertIsInstance(recommendations, list)
        
        # Check that it returns at least one recommendation
        self.assertGreater(len(recommendations), 0)
        
        # Check that all recommendations are strings
        for rec in recommendations:
            self.assertIsInstance(rec, str)

class TestPriceComparator(unittest.TestCase):
    def setUp(self):
        self.price_comparator = PriceComparator()
    
    def test_compare_prices(self):
        # Test with mock flight and hotel data
        flights = [
            {"price": 500, "stops": 0, "airline": "Test Airline"}
        ]
        hotels = [
            {"price": 100, "rating": 4.5, "name": "Test Hotel"}
        ]
        
        result = self.price_comparator.compare_prices(flights, hotels)
        
        # If there's a result, check its structure
        if result:
            expected_keys = ['flight', 'hotel', 'total_cost', 'value_score']
            for key in expected_keys:
                self.assertIn(key, result)

if __name__ == '__main__':
    unittest.main()