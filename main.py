import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-course!")

    information = """
    Sir Lewis Carl Davidson Hamilton (born 7 January 1985) is a British racing driver who competes in Formula One for Ferrari. Hamilton has won a joint-record seven Formula One World Drivers' Championship titles—tied with Michael Schumacher—and holds the records for most wins (105), pole positions (104), and podium finishes (202), among others.

    Born and raised in Stevenage, Hamilton began his career in karting aged six, winning several national titles and attracting the attention of Ron Dennis, who signed him to the McLaren-Mercedes Young Driver Programme in 1998. After winning the direct-drive Karting World Cup and European Championship in 2000, Hamilton progressed to junior formulae, where his successes included winning the Formula 3 Euro Series and the GP2 Series. He subsequently signed for McLaren in 2007, becoming the first black driver to compete in Formula One at the Australian Grand Prix. In his rookie season, Hamilton won four Grands Prix and set several records as he finished runner-up to Kimi Räikkönen by one point. Hamilton won his maiden title in 2008, making a title-deciding overtake on the last lap of the last race of the season to become the then-youngest World Drivers' Champion. The Red Bull–Renault combination prevailed throughout his remaining four seasons at McLaren, with Hamilton achieving multiple race wins in each, including his involvement in a four-way title battle in 2010.

    Hamilton signed for Mercedes in 2013 to partner his old karting teammate Nico Rosberg, ending his 15-year association with McLaren. Following his maiden victory with the team at the Hungarian Grand Prix, new engine regulations the following season saw Mercedes emerge as the dominant force in Formula One. Over the next three seasons, Hamilton and Rosberg won 51 of 59 Grands Prix amidst their fierce rivalry—widely known as the Silver War—with Hamilton winning the former titles in 2014 and 2015, and Rosberg winning the latter. After Rosberg's retirement, Hamilton twice overturned mid-season point deficits to Sebastian Vettel of Ferrari to claim his fourth and fifth titles in 2017 and 2018. Hamilton won his sixth title in 2019, before breaking several records across his 2020 campaign—including the all-time win record at the Portuguese Grand Prix—to claim his record-equalling seventh. Hamilton became the first driver to surpass 100 race wins and pole positions in 2021, ending runner-up to Max Verstappen amidst a controversial finish. Following winless campaigns in 2022 and 2023, he took his record-breaking ninth British Grand Prix victory in 2024, his twelfth and final season with Mercedes. Hamilton signed for Ferrari in 2025, and is contracted to remain in the team until at least the end of 2026.

    Hamilton has been credited with furthering Formula One's global following by appealing to a broader audience outside the sport, in part due to his high-profile lifestyle, amongst his environmental and social activism. He has also become a prominent advocate in support of racial justice and increased diversity in motorsport. Hamilton was listed in the 2020 issue of Time as one of the 100 most influential people globally, and was knighted in the 2021 New Year Honours.
    """

    summary_template = """
    Give a the information {information} about a person I want you to create:
    1. A brief summary 
    2. Three interesting facts about them
    """

    summary_prompt_template =PromptTemplate(template=summary_template, input_variables=["information"])

    llm = ChatOpenAI(temperature=0.9, model="gpt-5")
    # llm = ChatOllama(temperature=0.9, model="gemma3:270m")

    chain = summary_prompt_template | llm
    response  =chain.invoke(input = {"information":information})

    print(response.content)
    



if __name__ == "__main__":
    main()
