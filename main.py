from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
# from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain.agents.react.agent import create_react_agent
from langchain.agents import AgentExecutor
from langchain_tavily import TavilySearch
from langchain import hub

from dotenv import load_dotenv

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools"]
).partial(format_instructions = output_parser.get_format_instructions())

agent = create_react_agent(
    # llm=ChatOllama(temperature=0.9, model="gemma3:270m"),
    llm=ChatOpenAI(temperature=0.9, model="gpt-4"),
    # llm=ChatOllama(temperature=0.9, model="mistral:latest"),
    prompt=react_prompt_with_format_instructions,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

extract_output = RunnableLambda(lambda x: x['output'])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))
chain = agent_executor | extract_output | parse_output

def main():
    result = chain.invoke(input = {"input":"Search for 3 job postings for AWS,NodeJS, ANgular full stack developer with 14 years of experience in Noida in IT industry"})
    print(result)


if __name__ == "__main__":
    main()
