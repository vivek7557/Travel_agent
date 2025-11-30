import requests
import json
from config import TIMEOUT, MAX_RETRIES
from typing import Dict, List, Optional

class APIHandler:
    def __init__(self):
        self.session = requests.Session()
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'TravelAgent/1.0'
        })
    
    def make_request(self, method: str, url: str, **kwargs) -> Optional[Dict]:
        """
        Make an API request with error handling and retries
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    timeout=TIMEOUT,
                    **kwargs
                )
                
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt == MAX_RETRIES - 1:
                    # Last attempt failed
                    return None
        
        return None
    
    def get(self, url: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a GET request
        """
        return self.make_request('GET', url, params=params)
    
    def post(self, url: str, data: Dict = None, json_data: Dict = None) -> Optional[Dict]:
        """
        Make a POST request
        """
        if json_data:
            return self.make_request('POST', url, json=json_data)
        elif data:
            return self.make_request('POST', url, data=data)
        else:
            return self.make_request('POST', url)
    
    def put(self, url: str, data: Dict = None, json_data: Dict = None) -> Optional[Dict]:
        """
        Make a PUT request
        """
        if json_data:
            return self.make_request('PUT', url, json=json_data)
        elif data:
            return self.make_request('PUT', url, data=data)
        else:
            return self.make_request('PUT', url)
    
    def delete(self, url: str) -> Optional[Dict]:
        """
        Make a DELETE request
        """
        return self.make_request('DELETE', url)

# Example implementations for specific services
class FlightAPIHandler(APIHandler):
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
    
    def search_flights(self, origin: str, destination: str, date: str):
        # Implementation would connect to flight APIs
        pass

class HotelAPIHandler(APIHandler):
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
    
    def search_hotels(self, destination: str, check_in: str, check_out: str = None):
        # Implementation would connect to hotel APIs
        pass

class WeatherAPIHandler(APIHandler):
    def __init__(self, api_key: str = None):
        super().__init__()
        if api_key:
            self.session.headers.update({'x-api-key': api_key})
    
    def get_weather(self, location: str):
        # Implementation would connect to weather APIs
        pass