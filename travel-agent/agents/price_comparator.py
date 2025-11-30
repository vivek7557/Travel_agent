class PriceComparator:
    def __init__(self):
        pass
    
    def compare_prices(self, flights, hotels, car_rentals=None):
        """
        Compare prices across different options to find the best value
        """
        try:
            # Calculate total cost for different combinations
            options = self._calculate_combinations(flights, hotels, car_rentals)
            
            # Sort by best value (price vs quality)
            sorted_options = sorted(options, key=lambda x: x['value_score'], reverse=True)
            
            return sorted_options[0] if sorted_options else None
        except Exception as e:
            print(f"Error comparing prices: {str(e)}")
            return None
    
    def _calculate_combinations(self, flights, hotels, car_rentals=None):
        """
        Calculate different combinations of flights and hotels
        """
        combinations = []
        
        for flight in flights:
            for hotel in hotels:
                # Calculate total cost
                total_cost = flight['price'] + (hotel['price'] * 5)  # Assuming 5 nights
                
                # Calculate value score (simplified)
                value_score = self._calculate_value_score(flight, hotel, total_cost)
                
                combination = {
                    'flight': flight,
                    'hotel': hotel,
                    'total_cost': total_cost,
                    'value_score': value_score
                }
                
                combinations.append(combination)
        
        return combinations
    
    def _calculate_value_score(self, flight, hotel, total_cost):
        """
        Calculate a value score based on price, quality, and other factors
        """
        # Simple value calculation (price vs quality)
        flight_quality = 5 if flight['stops'] == 0 else 5 - flight['stops']
        hotel_quality = hotel['rating']
        
        # Higher rating = better value, lower price = better value
        value_score = (flight_quality + hotel_quality) / (total_cost / 100)
        
        return value_score
    
    def find_best_deals(self, service_type, options):
        """
        Find the best deals for a specific service type
        """
        if not options:
            return []
        
        # Sort by price-to-quality ratio
        sorted_options = sorted(options, 
                               key=lambda x: x.get('rating', 0) / (x.get('price', 1) or 1), 
                               reverse=True)
        
        return sorted_options