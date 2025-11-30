import requests
from typing import Dict, List

class AgentTools:
    
    def __init__(self):
        self.amadeus_key = os.environ.get("AMADEUS_API_KEY")
        self.booking_key = os.environ.get("BOOKING_API_KEY")
    
    def search_flights(self, origin: str, destination: str, 
                      date: str, passengers: int) -> List[Dict]:
        """
        Search for available flights
        """
        # Amadeus API call
        url = "https://api.amadeus.com/v2/shopping/flight-offers"
        headers = {"Authorization": f"Bearer {self.amadeus_key}"}
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": date,
            "adults": passengers
        }
        
        response = requests.get(url, headers=headers, params=params)
        flights = response.json()
        
        # Parse and return structured data
        return self._parse_flights(flights)
    
    def search_hotels(self, location: str, check_in: str, 
                     check_out: str, guests: int) -> List[Dict]:
        """
        Search for available hotels
        """
        # Booking.com API call
        url = "https://api.booking.com/v1/hotels/search"
        params = {
            "city": location,
            "checkin": check_in,
            "checkout": check_out,
            "guests": guests
        }
        
        response = requests.get(url, params=params)
        hotels = response.json()
        
        return self._parse_hotels(hotels)
    
    def create_itinerary(self, destination: str, days: int, 
                        interests: List[str]) -> Dict:
        """
        Generate day-by-day itinerary using AI
        """
        # Use Claude to generate itinerary
        prompt = f"""Create a {days}-day itinerary for {destination}.
        Interests: {', '.join(interests)}
        Include: activities, restaurants, timing, estimated costs."""
        
        # Call AI to generate structured itinerary
        itinerary = self._generate_with_ai(prompt)
        return itinerary
    
    def process_booking(self, booking_details: Dict) -> Dict:
        """
        Handle actual booking and payment
        """
        import stripe
        stripe.api_key = os.environ.get("STRIPE_API_KEY")
        
        # Process payment
        charge = stripe.Charge.create(
            amount=booking_details['total_amount'],
            currency="usd",
            source=booking_details['payment_token'],
            description=f"Travel booking - {booking_details['confirmation']}"
        )
        
        # Confirm booking with suppliers
        confirmation = self._confirm_with_suppliers(booking_details)
        
        return {
            "status": "confirmed",
            "confirmation_number": confirmation,
            "charge_id": charge.id
        }
    
    def send_confirmation_email(self, customer_email: str, 
                               booking_details: Dict):
        """
        Send booking confirmation via email
        """
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
        message = Mail(
            from_email='bookings@yourtravelagency.com',
            to_emails=customer_email,
            subject='Your Trip Confirmation',
            html_content=self._generate_email_template(booking_details)
        )
        
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        
        return response.status_code
