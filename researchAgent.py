import os
import certifi
from dotenv import load_dotenv

import streamlit as st

from langchain import hub
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent
from langchain.agents.agent import AgentExecutor


# ----------------------------
# Load Environment Variables
# ----------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Research Agent")
st.write("Ask anything. The agent can search the web using Tavily.")


# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)


# ----------------------------
# Initialize Search Tool
# ----------------------------
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)

tools = [search_tool]


# ----------------------------
# Load ReAct Prompt
# ----------------------------
prompt = hub.pull("hwchase17/react")


# ----------------------------
# Create Agent
# ----------------------------
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)


# ----------------------------
# Create Agent Executor
# ----------------------------
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)


# ----------------------------
# User Input
# ----------------------------
query = st.text_input(
    "Enter your question",
    placeholder="Example: Tell me the latest AI Agent concepts introduced in 2026"
)


# ----------------------------
# Generate Response
# ----------------------------
if st.button("Ask Agent"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):

            try:
                response = agent_executor.invoke(
                    {"input": query}
                )

                st.subheader("Answer")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {e}")