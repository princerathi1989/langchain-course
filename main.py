import re
from typing import List, Union
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_tavily import TavilySearch
from langchain import hub
from langchain.agents import tool
from langchain.tools import Tool
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish

from dotenv import load_dotenv

from callbacks import AgentCallbackHandler
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

@tool
def get_text_length(text: str)->int:
    """
    Returns the length of cleaned text input.
    """
    cleaned = text.strip("'\n").strip('"').strip("'")
    return len(cleaned)


# def main():
#     result = chain.invoke(input = {"input":"Search for 3 job postings for AWS,NodeJS, ANgular full stack developer with 14 years of experience in Noida in IT industry"})
#     print(result)

def find_tool_by_name(tools: List[Tool], tool_name: str)->Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found")

if __name__ == "__main__":
    print("Hello React Langchain")

    tools = [get_text_length]

    template="""
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools), 
        tool_names = ", ".join([t.name for t in tools])
    )

    llm = ChatOpenAI(temperature=0, stop = ["\nObservation:", "Observation", "Observation:"], callbacks=[AgentCallbackHandler()])

    intermediate_steps = []

    agent = (
        {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x:x["agent_scratchpad"]
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] =  agent.invoke({
            "input": "What is the length in characters of the text Lion?",
            "agent_scratchpad": intermediate_steps
        })

        print(agent_step)
        
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))

            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)

