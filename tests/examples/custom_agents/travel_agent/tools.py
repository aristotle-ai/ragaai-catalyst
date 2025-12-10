import os
import random
import requests
from dotenv import load_dotenv
from openai import OpenAI

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from config import tracer

load_dotenv()

@tracer.tracer.llm
def llm_call(prompt, max_tokens=512, model="gpt-4o-mini", name="default"):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

@tracer.tracer.tool
def weather_tool(destination):
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": destination, "appid": api_key, "units": "metric"}
    print("Calculating weather for:", destination)
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return f"{data['weather'][0]['description'].capitalize()}, {data['main']['temp']:.1f}°C"
    except requests.RequestException:
        return "Weather data not available."

@tracer.tracer.tool
def currency_converter_tool(amount, from_currency, to_currency):
    api_key = os.environ.get("EXCHANGERATE_API_KEY")
    base_url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{from_currency}/{to_currency}"

    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()

        if data["result"] == "success":
            rate = data["conversion_rate"]
            return amount * rate
        else:
            return None
    except requests.RequestException:
        return None

@tracer.tracer.tool
def flight_price_estimator_tool(origin, destination):
    return f"Estimated price from {origin} to {destination}: $500-$1000"
