import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any
from datetime import datetime

class DataProcessor:
    def __init__(self):
        pass
    
    def clean_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        Clean and standardize raw data from various sources
        """
        cleaned_data = []
        
        for item in raw_data:
            cleaned_item = {}
            
            # Standardize keys and clean values
            for key, value in item.items():
                # Convert key to lowercase and replace spaces with underscores
                clean_key = key.lower().replace(' ', '_').replace('-', '_')
                
                # Clean the value based on its type
                if isinstance(value, str):
                    # Remove extra whitespace and common artifacts
                    clean_value = value.strip()
                    # Try to convert price strings to numbers
                    if 'price' in clean_key.lower() or 'cost' in clean_key.lower():
                        clean_value = self._extract_price(clean_value)
                else:
                    clean_value = value
                
                cleaned_item[clean_key] = clean_value
            
            cleaned_data.append(cleaned_item)
        
        return cleaned_data
    
    def _extract_price(self, price_str: str) -> float:
        """
        Extract numeric price from a string
        """
        import re
        # Remove currency symbols and extract numbers
        price_match = re.search(r'[\d,]+\.?\d*', price_str.replace(',', ''))
        if price_match:
            try:
                return float(price_match.group())
            except ValueError:
                return 0.0
        return 0.0
    
    def normalize_ratings(self, ratings: List[float], from_scale: int, to_scale: int = 5) -> List[float]:
        """
        Normalize ratings from one scale to another (e.g., 10-point to 5-point scale)
        """
        normalized = []
        for rating in ratings:
            normalized_rating = (rating / from_scale) * to_scale
            normalized.append(min(to_scale, max(0, round(normalized_rating, 2))))
        
        return normalized
    
    def merge_data_sources(self, data_sources: List[List[Dict]], primary_key: str) -> List[Dict]:
        """
        Merge data from multiple sources based on a primary key
        """
        merged_dict = {}
        
        for source in data_sources:
            for item in source:
                key_value = item.get(primary_key)
                if key_value:
                    if key_value in merged_dict:
                        # Update existing item with new data
                        merged_dict[key_value].update(item)
                    else:
                        # Add new item
                        merged_dict[key_value] = item
        
        return list(merged_dict.values())
    
    def calculate_statistics(self, data: List[Dict], field: str) -> Dict[str, Any]:
        """
        Calculate basic statistics for a field in the data
        """
        values = []
        for item in data:
            if field in item and item[field] is not None:
                try:
                    values.append(float(item[field]))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    def filter_data(self, data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """
        Filter data based on specified criteria
        """
        filtered_data = []
        
        for item in data:
            match = True
            for field, criteria in filters.items():
                if field not in item:
                    match = False
                    break
                
                value = item[field]
                
                if isinstance(criteria, dict):
                    # Handle range filters like {'min': 100, 'max': 500}
                    if 'min' in criteria and value < criteria['min']:
                        match = False
                        break
                    if 'max' in criteria and value > criteria['max']:
                        match = False
                        break
                elif isinstance(criteria, list):
                    # Handle list of acceptable values
                    if value not in criteria:
                        match = False
                        break
                else:
                    # Handle exact match
                    if value != criteria:
                        match = False
                        break
            
            if match:
                filtered_data.append(item)
        
        return filtered_data

class TravelDataProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
    
    def process_flight_data(self, flights: List[Dict]) -> List[Dict]:
        """
        Process and standardize flight data
        """
        processed_flights = self.clean_data(flights)
        
        # Additional flight-specific processing
        for flight in processed_flights:
            # Convert duration string to minutes
            if 'duration' in flight:
                flight['duration_minutes'] = self._duration_to_minutes(flight['duration'])
            
            # Standardize airline names
            if 'airline' in flight:
                flight['airline'] = flight['airline'].title()
        
        return processed_flights
    
    def process_hotel_data(self, hotels: List[Dict]) -> List[Dict]:
        """
        Process and standardize hotel data
        """
        processed_hotels = self.clean_data(hotels)
        
        # Additional hotel-specific processing
        for hotel in processed_hotels:
            # Normalize ratings to 5-point scale
            if 'rating' in hotel and isinstance(hotel['rating'], (int, float)):
                hotel['rating_normalized'] = self._normalize_rating_scale(hotel['rating'])
            
            # Standardize hotel names
            if 'name' in hotel:
                hotel['name'] = hotel['name'].title()
        
        return processed_hotels
    
    def _duration_to_minutes(self, duration_str: str) -> int:
        """
        Convert duration string like '2h 30m' to total minutes
        """
        import re
        hours_match = re.search(r'(\d+)\s*h', duration_str, re.IGNORECASE)
        minutes_match = re.search(r'(\d+)\s*m', duration_str, re.IGNORECASE)
        
        total_minutes = 0
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60
        if minutes_match:
            total_minutes += int(minutes_match.group(1))
        
        return total_minutes
    
    def _normalize_rating_scale(self, rating: float) -> float:
        """
        Normalize ratings to a 5-point scale
        """
        # Assume the rating is on a 10-point scale if it's greater than 5
        if rating > 5:
            return min(5.0, max(0.0, round(rating / 2, 1)))
        return min(5.0, max(0.0, round(rating, 1)))