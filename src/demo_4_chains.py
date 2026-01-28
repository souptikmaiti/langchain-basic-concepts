"""Demo 4: Complex Chains"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def run_chains_demo():
    # Chain 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a random interesting topic about {subject}"
    )
    
    # Chain 2: Write about the topic
    writing_prompt = ChatPromptTemplate.from_template(
        "Write a short paragraph about: {topic}"
    )
    
    model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    
    # Combine chains
    chain = (
        {"topic": topic_prompt | model | StrOutputParser()}
        | RunnablePassthrough.assign(
            paragraph=writing_prompt | model | StrOutputParser()
        )
    )

    """
    RunnablePassthrough.assign does two things
        Passes through the existing topic key
        Adds a new key paragraph by running another chain
    Flow:
        Input to assign: {topic: "Mars Colonization"}
        writing_prompt uses that {topic} value
        Output: {topic: "Mars Colonization", paragraph: "Mars colonization represents..."}"
    """
    
    result = chain.invoke({"subject": "space exploration"})
    
    print("=== Chains Demo ===")
    print(f"Topic: {result['topic']}")
    print(f"\nParagraph: {result['paragraph']}")

if __name__ == "__main__":
    run_chains_demo()