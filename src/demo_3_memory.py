"""Demo 3: Conversation Memory"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def run_memory_demo():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"))
    chain = prompt | model
    
    # Simulate conversation with memory
    chat_history = []
    
    # First interaction
    response1 = chain.invoke({
        "chat_history": chat_history,
        "input": "My name is Alice and I love Python programming."
    })
    chat_history.extend([
        # HumanMessage(content="My name is Alice and I love Python programming."),
        AIMessage(content=response1.content)
    ])
    
    # Second interaction - model should remember
    response2 = chain.invoke({
        "chat_history": chat_history,
        "input": "What's my name and what do I love?"
    })
    
    print("=== Memory Demo ===")
    print(f"Response: {response2.content}")

if __name__ == "__main__":
    run_memory_demo()