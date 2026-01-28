"""Demo 2: Tool Calling with Agent"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers."""
    return first_int + second_int

def run_tools_demo():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [multiply, add]
    
    # Required: Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    print("=== Tool Calling Demo ===")
    result = executor.invoke(
        {"input": "What is 25 multiplied by 4, then add 10?"}
    )
    print(f"\nFinal Answer: {result['output']}")


"""
Explanation:

("system", "You are a helpful assistant") -> Sets the AI's behavior/personality, static, always first
("placeholder", "{chat_history}") -> Placeholder for chat history, slot for inserting multiple messages
        Holds previous conversation turns (if any)
        Example: [HumanMessage("Hi"), AIMessage("Hello!")]
        Empty list [] if no history
("human", "{input}") -> The current user question/request
        Gets replaced with actual input: "What is 25 multiplied by 4, then add 10?"
("placeholder", "{agent_scratchpad}") -> Critical for agents! Stores the agent's "thinking process"
        Contains:
            Tool calls the agent made
            Tool execution results
            Intermediate reasoning steps

1. Human: "What is 25 * 4 + 10?"
2. Agent scratchpad: [ToolCall(multiply, 25, 4)]
3. Agent scratchpad: [ToolResult(100)]
4. Agent scratchpad: [ToolCall(add, 100, 10)]
5. Agent scratchpad: [ToolResult(110)]
6. Agent: "The answer is 110"

Without {agent_scratchpad}, the agent can't see tool results and gets stuck in a loop!
"""

if __name__ == "__main__":
    run_tools_demo()
