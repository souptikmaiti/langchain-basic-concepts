"""Main entry point to run all demos"""
from demo_1_prompt_template import run_prompt_template_demo
from demo_2_tools import run_tools_demo
from demo_3_memory import run_memory_demo
from demo_4_chains import run_chains_demo

def main():
    print("\n" + "="*50)
    run_prompt_template_demo()
    
    print("\n" + "="*50)
    run_tools_demo()
    
    print("\n" + "="*50)
    run_memory_demo()
    
    print("\n" + "="*50)
    run_chains_demo()
    print("\n" + "="*50)

if __name__ == "__main__":
    main()