import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load your OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Load CSV agent
agent = create_csv_agent(llm, "data/google_ab_data.csv", verbose=True)

# Ask your question
response = agent.run("What are the top 3 insights from this dataset?")
print(response)

