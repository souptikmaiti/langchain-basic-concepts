"""Demo 1: Prompt Templates"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def run_prompt_template_demo():
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains {topic} in simple terms."),
        ("human", "{question}")
    ])
    
    # Initialize model
    use_gemini = os.getenv("USE_GEMINI", "false").lower() == "true"
    if use_gemini:
        model = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash"),
            temperature=float(os.getenv("TEMPERATURE", 0.7))
        )
        print("Using Google Gemini")
    else:
        model = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            temperature=float(os.getenv("TEMPERATURE", 0.7))
        )
        print("Using OpenAI")
    
    # Create chain
    chain = prompt | model | StrOutputParser()
    
    # Invoke
    response = chain.invoke({
        "topic": "machine learning",
        "question": "What is a neural network?"
    })
    
    print("=== Prompt Template Demo ===")
    print(response)

if __name__ == "__main__":
    run_prompt_template_demo()