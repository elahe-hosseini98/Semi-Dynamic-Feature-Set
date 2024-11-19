from openai import OpenAI
import os

available_models = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.2-1b-instruct",
    "google/gemini-flash-1.5",
    "gryphe/mythomax-l2-13b",
    "microsoft/wizardlm-2-8x22b",
    "openai/o1-preview-2024-09-12"
]

model = available_models[0]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY",
                           "YOUR OPENROUTER_API_KEY HERE"),
)