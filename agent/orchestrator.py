from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    customer_id: str
    current_step: str
    search_results: dict
    selected_options: dict
    booking_confirmed: bool

class TravelAgentOrchestrator:
    
    def __init__(self):
        self.brain = TravelAIAgent()
        self.tools = AgentTools()
        self.memory = AgentMemory()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """
        Define the agent's decision flow
        """
        workflow = StateGraph(AgentState)
        
        # Add nodes (steps)
        workflow.add_node("understand_request", self.understand_request)
        workflow.add_node("retrieve_memory", self.retrieve_customer_memory)
        workflow.add_node("search_options", self.search_travel_options)
        workflow.add_node("present_options", self.present_to_customer)
        workflow.add_node("process_booking", self.handle_booking)
        workflow.add_node("send_confirmation", self.send_confirmation)
        
        # Define edges (flow)
        workflow.set_entry_point("understand_request")
        workflow.add_edge("understand_request", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "search_options")
        workflow.add_edge("search_options", "present_options")
        workflow.add_conditional_edges(
            "present_options",
            self.should_book,
            {
                "book": "process_booking",
                "search_more": "search_options",
                "clarify": "understand_request"
            }
        )
        workflow.add_edge("process_booking", "send_confirmation")
        workflow.add_edge("send_confirmation", END)
        
        return workflow.compile()
    
    def understand_request(self, state: AgentState):
        """
        Use AI to understand what the customer wants
        """
        last_message = state["messages"][-1]
        
        # AI analyzes the request
        analysis = self.brain.think(
            last_message,
            available_tools=self._get_tool_definitions()
        )
        
        state["current_step"] = "understood"
        return state
    
    def retrieve_customer_memory(self, state: AgentState):
        """
        Recall customer preferences and history
        """
        customer_data = self.memory.recall_customer(state["customer_id"])
        state["customer_preferences"] = customer_data
        return state
    
    def search_travel_options(self, state: AgentState):
        """
        Search for flights, hotels, activities
        """
        # Extract search parameters from conversation
        params = self._extract_search_params(state["messages"])
        
        # Parallel search across all providers
        results = {
            "flights": self.tools.search_flights(**params["flight_params"]),
            "hotels": self.tools.search_hotels(**params["hotel_params"]),
            "activities": self.tools.search_activities(**params["activity_params"])
        }
        
        state["search_results"] = results
        return state
    
    def present_to_customer(self, state: AgentState):
        """
        AI creates a personalized presentation of options
        """
        presentation = self.brain.create_proposal(
            search_results=state["search_results"],
            customer_prefs=state.get("customer_preferences", {})
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": presentation
        })
        return state
    
    def should_book(self, state: AgentState):
        """
        Decide next action based on customer response
        """
        last_message = state["messages"][-1]
        
        if "book" in last_message.lower() or "confirm" in last_message.lower():
            return "book"
        elif "different" in last_message.lower() or "other options" in last_message.lower():
            return "search_more"
        else:
            return "clarify"
    
    def handle_booking(self, state: AgentState):
        """
        Process the actual booking
        """
        booking_result = self.tools.process_booking(
            state["selected_options"]
        )
        
        state["booking_confirmed"] = True
        state["confirmation_number"] = booking_result["confirmation_number"]
        return state
    
    def send_confirmation(self, state: AgentState):
        """
        Send confirmation and save to memory
        """
        self.tools.send_confirmation_email(
            state["customer_email"],
            state["selected_options"]
        )
        
        # Update customer memory with this trip
        self.memory.remember_customer(
            state["customer_id"],
            {"latest_booking": state["selected_options"]}
        )
        
        return state
    
    def run(self, customer_message: str, customer_id: str):
        """
        Main entry point - run the agent
        """
        initial_state = {
            "messages": [{"role": "user", "content": customer_message}],
            "customer_id": customer_id,
            "current_step": "start",
            "search_results": {},
            "selected_options": {},
            "booking_confirmed": False
        }
        
        # Execute the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return final_state
