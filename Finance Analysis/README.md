<a id="readme-top"></a>

## AI Assistant for Financial Analysis
This project uses a hierarchical workflow involving a supervisor agent, two worker agents, and a synthesizer agent to provide financial data based on user queries. The workflow as well as agent descriptions are shown below:
<p align="center">
    <img src="AI Assistant Workflow.png" height="300px">
</p>

| Agent Name | Role | Primary Tasks |
| :-----: | :-----------: | :------------: |
| Supervisor | Orchestrator & Router | Manages state, agent delegation, and conversation history |
| Price Analyst | Historical Price Analyst | Retrieves 90-day historical pricing using yfinance |
| Filing Analyst | Market Data Specialist | Accesses Edgar database for SEC filings
| Synthesizer | Summarizer | Compiles all gathered worker reports and conversation history to generate a response for the user |


## Getting Started

To run this application locally, I recommend creating a 
virtual environment (venv), and installing
dependencies, which are within the requirements.txt file.

Python version used: 3.11.13

Terminal commands (bash):

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Once all dependencies are installed, run the application by navigating 
to the Finance Analysis folder and running the following command

```bash
streamlit run main.py
```

## Evaluation
Currently working on agentic evaluation of LLM's. 

Targeted Metrics:
1. Supervisor:
    - Orchestration Evaluation - Test if the Supervisor agent makes the right "Next Agent" decision based on user query
    - End-to-End Evaluation - How many steps does it take to get a final answer?
    - Token Efficiency- Monitor usage.
    - Redundancy- Does the Supervisor call the same node twice for the same data?
2. Price & Filing Analysts
    - Tool calling - Test if the LLM correctly calls tools
    - Token Efficiency - Monitor usage.
    - Job Fulfillment - Does the node accomplish its task correctly?

Candidate LLM's (~500M parameter maximum since we're running on CPU) will be chosen based on performance of selected benchmarks from HuggingFace:
- IFEval for instruction following
- Big-Bench Hard (BBH) for language understanding
- Multi-Step Soft Reasoning (MUSR) for language understanding and context reasoning for long text

## Lessons Learned
- Managing dependencies is crucial for compatible workflows. pipreqs helped me troubleshoot some dependency issues
    - This goes hand-in-hand with creating separate environments for projects to ensure modularity and organization
- LLM's with hosted API services help with compute time compared to local/self-hosted LLM's running on CPU, but may be limited in usage for free tier subscriptions.
    - Evaluation of models near 10-15 billion parameters is way too great for CPU processing

## Future Work
1. Look into hosting on the cloud with GPU usage.
    - This can open up locally deployed LLM's implementation
2. Add more error handling to the architecture. i.e. Functionality that can handle usage for self-hosted API's.
3. Explore containerization with Docker
