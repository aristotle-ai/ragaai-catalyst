from tools import (
    llm_call,
    weather_tool,
    currency_converter_tool,
    flight_price_estimator_tool,
)
from config import tracer

class ItineraryAgent:
    def __init__(self, persona="Itinerary Agent"):
        self.persona = persona

    @tracer.tracer.agent
    def plan_itinerary(self, user_preferences, duration=3):
        
        # Get weather information
        weather = weather_tool(user_preferences["destination"])

        # Get currency conversion if needed
        if "budget_currency" in user_preferences and user_preferences["budget_currency"] != "USD":
            budget = currency_converter_tool(
                user_preferences["budget"], user_preferences["budget_currency"], "USD"
            )
        else:
            budget = user_preferences["budget"]

        # Get flight price estimation
        flight_price = flight_price_estimator_tool(
            user_preferences["origin"], user_preferences["destination"]
        )

        # Prepare prompt for the LLM
        prompt = f"""As a {self.persona}, create a {duration}-day itinerary for a trip to {user_preferences['destination']}.
        Weather: {weather}
        Budget: ${budget}
        Flight Price: {flight_price}
        Preferences: {user_preferences.get('preferences', 'No specific preferences')}
        
        Please provide a detailed day-by-day itinerary."""

        # Generate itinerary using LLM
        return llm_call(prompt)
