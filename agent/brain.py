import anthropic
import os

class TravelAIAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.conversation_history = []
        
    def think(self, user_message, available_tools):
        """
        Main reasoning function
        Agent decides what to do based on user input
        """
        
        system_prompt = """You are an expert AI travel agent. Your job is to:
        1. Understand customer travel needs
        2. Search for the best travel options
        3. Create personalized itineraries
        4. Handle bookings end-to-end
        5. Provide 24/7 support
        
        You have access to these tools: {tools}
        
        Always be helpful, accurate, and proactive.
        Ask clarifying questions when needed.
        Present options clearly with prices.
        Confirm before making any bookings.
        """
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Call Claude with tool use capability
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt.format(tools=available_tools),
            messages=self.conversation_history,
            tools=available_tools
        )
        
        return response
