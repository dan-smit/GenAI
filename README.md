<a id="readme-top"></a>

## AI Assistant for Financial Analysis
This project uses a hierarchical workflow involving a supervisor agent, two worker agents, and a synthesizer agent to provide financial data based on user queries. The agent names and descriptions are as follows:

| Agent Name | Role | Primary Tasks |
| ----- | ----------- | ------------ |
| Supervisor | Orchestrator & Router | Manages state, agent delegation, and conversation history |
| Price Analyst | Historical Price Analyst | Retrieves 90-day historical pricing using yfinance |
| Filing Analyst | Market Data Specialist | Accesses Edgar database for SEC filings
| Synthesizer | Summarizer | Compiles all gathered worker reports and conversation history to generate a response for the user |


## Getting Started

To run this application locally, I recommend creating a 
virtual environment (venv), and installing
dependencies, which are within the requirements.txt file.

Python version used: 3.11.13

Terminal commands:
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

Once all dependencies are installed, run the application by navigating 
to the Finance Analysis folder and running the following terminal
command:

streamlit run main.py

## Evaluation
Evaluation was targeted using CPU compute. As a result, self-hosted API LLM's were tested. 

1. Download ollama
https://ollama.com/download

2. Pull weights for Ollama models using the following commands in terminal (I used powershell):
ollama pull qwen2.5:1.5b
ollama pull phi3.5

Metrics:

1. Orchestration Evaluation - Test if the Supervisor agent makes the right "Next Agent" decision based on user query
2. End-to-End Evaluation - How many steps does it take to get a final answer?
3. Loop Rate- How often the architecture enters a loop. The loop occurs when the worker agent is unable to get a retrieve information and calls the Supervisor agent, and the Supervisor agent recalls on the worker agent.
4. Token Efficiency- Every loop adds the entire history back to the propmt. 
5. Redundancy- Does the Supervisor call the same node twice for the same data? 

Candidate LLM's (500M parameter maximum) were chosen based on performance of selected benchmarks from HuggingFace:
- IFEval for instruction following
- Big-Bench Hard (BBH) for language understanding
- Multi-Step Soft Reasoning (MUSR) for language understanding and context reasoning for long text

Models chosen:
- 1. JungZoona/T3Q-qwen2.5-14b-v1.0-e3
- 2. prithivMLmods/Galactic-Qwen-14B-Exp2
- 3. suayptalha/Lamarckvergence-14B

## Lessons Learned
- Managing dependencies is crucial for compatible workflows. pipreqs helped me troubleshoot some dependency issues
    - This goes hand-in-hand with creating separate environments for projects to ensure modularity and organization
- LLM's with hosted API services help with compute time compared to local/self-hosted LLM's running on CPU
    - Evaluation of models near 10-15 billion parameters is way too great for CPU processing


## Future Work
1. Look into hosting on the cloud with GPU usage.
2. Add error handling to the architecture. i.e. Functionality that can handle usage for self-hosted API's.
3. Assess synthesizer agent
