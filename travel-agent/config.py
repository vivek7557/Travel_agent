import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SKYSCANNER_API_KEY = os.getenv("SKYSCANNER_API_KEY")  # Example for flight API
HOTEL_API_KEY = os.getenv("HOTEL_API_KEY")  # Example for hotel API
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

# Settings
CACHE_DURATION = 3600  # 1 hour in seconds
MAX_RETRIES = 3
TIMEOUT = 30