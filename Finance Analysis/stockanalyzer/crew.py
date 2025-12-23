from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

# UPDATED: Added SystemMessage here
from langchain.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from stockanalyzer.config import Config
from stockanalyzer.tools import fetch_sec_filing_sections, get_historical_stock_price

# Prompts for Agents
SUPERVISOR_PROMPT = """You are a supervisor managing a team of financial analyst agents to answer user queries.
Create a delegation plan to answer the user's query.

<current_query>
{most_recent_query}
</current_query>

<conversation_history>
{conversation_history}
</conversation_history>

<instructions>
Available agents:
- Price Analyst: Retrieves historical stock price data
- Filing Analyst: Retrieves SEC filing information (10-K, 10-Q, risk factors, MD&A)
- Synthesizer: Creates final answer from gathered information

Decision Logic:
1. **Critical Check:** If the conversation history contains a message from "SystemError", immediately delegate to Synthesizer with: "An internal data retrieval error occurred. Provide a polite and concise final answer explaining the failure."
2. If more data is needed, delegate a specific **data retrieval task** to Price Analyst or Filing Analyst (e.g., "Fetch the latest 10-Q for NVDA.")
3. If you have sufficient information, delegate to Synthesizer with: "Provide a concise final answer."

For follow-up questions: Use context from earlier in the conversation and possible call all other agents to get more information.
When calling agents, always ask them to provide the analysis for your query.
</instructions>

<formatting>
JSON object with 'next_agent' and 'question' fields
</formatting>
"""

WORKER_PROMPT = """You are the {agent_name}. Summarize tool data to answer the supervisor's request
Create a concise report that directly addresses the supervisor's instruction.

<supervisor_instruction>
{supervisor_instruction}
</supervisor_instruction>

<tool_data>
{tool_data}
</tool_data>

<instructions>
- Use bullet points for key findings
- Highlight important numbers and dates with bold
- Maximum 3-5 key points
- Include ONLY critical information relevant to the instruction
- No preamble or conclusions. Just the facts
</instructions>

Output your summary now:
"""

SYNTHESIS_PROMPT = """You are an expert financial analyst.
Provide a concise, data-driven answer to the user's query.

<user_query>
{most_recent_user_query}
</user_query>

<conversation_history>
{conversation_history}
</conversation_history>

<instructions>
LENGTH LIMITS (strictly enforce):
- Initial company analysis: 5-8 sentences maximum
- Follow-up questions: 2-4 sentences maximum
- Data requests: Present facts concisely

Every word must add value. No fluff, no hedging, no unnecessary qualifiers.
</instructions>

<output_structure>
Choose the appropriate structure based on query type:

TYPE 1 - Initial Company Analysis:
**Overview** (2-3 sentences)
Key business metrics, major risks, or opportunities

**Price Action** (2-3 sentences)
Recent trends, key price levels, volatility

**Recommendations** (1-2 sentences)
BUY/HOLD/SELL with core reasoning based on data

TYPE 2 - Follow-up Questions:
Directly answer the question (2-4 sentences)
Reference prior context only if relevant

TYPE 3 - Data Requests:
Present key numbers/facts clearly
Add brief context only if essential
</output_structure>

<formatting>
- Use markdown: **bold** for emphasis, bullets for lists
- Be direct and data-driven
- Make it scannable and easy to read
</formatting>

Write your response now:"""

# Agent Names
class AgentName(StrEnum):
    PRICE_ANALYST="Price Analyst"
    FILING_ANALYST="Filing Analyst"
    SYNTHESIZER="Synthesizer"
    SUPERVISOR="Supervisor"

# Data Classes for Agent State and Context
# Contains messages, iteration count in the workflow, and next agent to call
@dataclass
class AgentState:
    messages: Annotated[list, add_messages]
    iteration_count: int=0
    next_agent: AgentName | None = None

# Used to pass context to the runtime, including the model
# BaseChatModel is the interface for interacting with chat-based LLMs in LangChain
@dataclass
class ContextSchema:
    model: BaseChatModel

# Structured output for Supervisor's plan
# BaseModel ensures each plan has a next_agent and question attribute
class SupervisorPlan(BaseModel):
    """A structured plan for the supervisor to delegate tasks."""
        
    # Specify the next agent to delegate to
    # Literal restricts the value to specific string options
    next_agent: Literal[
        AgentName.PRICE_ANALYST, AgentName.FILING_ANALYST, AgentName.SYNTHESIZER
    ] = Field(
        description="The next agent to delegate the task to, or 'Synthesizer' if enough information is gathered"
    )
    #question attribute 
    question: str = Field(
        description="A specific, focused question or instruction for the chosen agent."
    )
    
#formats messages from LangChains data classes, converting them to a single string, added to context of conversation
def format_history(messages: list) -> str:
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User" if not hasattr(msg, "name") else msg.name
            formatted.append(f"{role}: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            continue
    return "\n\n".join(formatted)

# Supervisor node in agent workflow.
# AgentState includes conversational messages and the next agent to call
# Runtime is the execution environment with context like the LLM model
def supervisor_node(state: AgentState, runtime: Runtime[ContextSchema]):
    #obtain model and context from model
    supervisor_llm = runtime.context.model.with_structured_output(SupervisorPlan)
    
    # Extract user queries in state
    user_messages = [
        msg for msg in state.messages if isinstance(msg, HumanMessage) and not hasattr(msg, "name")
    ]
    most_recent_query = user_messages[-1].content if user_messages else ""
    
    conversation_history = format_history(state.messages)
    
    plan = supervisor_llm.invoke(
        SUPERVISOR_PROMPT.format(
            most_recent_query=most_recent_query,
            conversation_history=conversation_history
        )
    )
    
    # Question for the next agent from the supervisor's plan
    # Return the updated AgentState
    new_message = HumanMessage(content=plan.question, name="SupervisorInstruction")
    return {
        "messages": [new_message],
        "next_agent": plan.next_agent,
        "iteration_count": state.iteration_count + 1
    }

# This function is called for the pricing and filing analysts
def create_worker_node(agent_name: AgentName, tools: list):
    # FINAL REVISED System instruction: Mandates argument extraction and tool call.
    system_instruction = (
        f"You are the {agent_name}. Your SOLE task is to process the Supervisor's instruction by calling your tool(s). "
        f"You MUST extract the necessary arguments (like ticker, filing type, or section) from the instruction "
        f"and use your tool IMMEDIATELY. "
        f"The raw tool output will be summarized by a different process (the WORKER_PROMPT). "
        f"Therefore, you MUST NOT generate any text, explanations, or questions. "
        f"Your ONLY output MUST be a valid tool call object. If the tool is relevant, CALL IT NOW. "
    )
    
    def worker_node(state: AgentState, runtime: Runtime[ContextSchema]):
        # Apply the system instruction and bind tools for the initial invocation
        
        # Tells LLM which tools are available to use
        agent = runtime.context.model.bind_tools(tools)
        supervisor_instruction = state.messages[-1]
        
        # Inject the system instruction as the first message
        system_message = SystemMessage(content=system_instruction)
        
        # This is the first LLM call, where we expect a tool_call
        response = agent.invoke([system_message, supervisor_instruction])
        
        # If the LLM returns text instead of a tool call (i.e., the problematic conversational response)
            # We explicitly prevent the conversational response from entering the history.
            # Instead, we inject a non-conversational, system-level failure message.        
        if not response.tool_calls and response.content:

            error_msg = f"Worker failure: {agent_name} failed to generate a tool call. The Supervisor must re-evaluate the instruction and try again."
            print(f"Worker failure detected: {response.content}")
            return {
                "messages": [HumanMessage(content=error_msg, name="SystemError")], 
                "next_agent": AgentName.SUPERVISOR
            }
        
        # If no tool calls but no content, proceed to tool execution (shouldn't happen here, but safe)
        if not response.tool_calls:
            return {"messages": [response], "next_agent": AgentName.SUPERVISOR}
        
        #Loop through each tool call and invoke the corresponding tool
        tool_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            print(f'tool_call: {tool_call}')
            
            selected_tool = next((t for t in tools if t.name == tool_name), None)
            
            # Invoke the tool with extracted arguments
            tool_output = selected_tool.invoke(tool_call["args"])
            
            # Append the tool output as a ToolMessage
            tool_messages.append(
                ToolMessage(content=str(tool_output), 
                            tool_call_id=tool_call['id'],
                            name=tool_call["name"]) 
            )
        
        # Summarization Step (Second LLM Call)
        # The tool output is now summarized using the strict WORKER_PROMPT
        summary_response = runtime.context.model.invoke(
            WORKER_PROMPT.format(
                agent_name=agent_name,
                supervisor_instruction=supervisor_instruction.content,
                tool_data=tool_messages[0].content # Only one tool call (so far)
            )
        )
        
        return{
            "messages": [summary_response],
            "next_agent": AgentName.SUPERVISOR
        }

    return worker_node

# Synthesizer node is responsible for generating the complete, user-facing answer
def synthesizer_node(state: AgentState, runtime: Runtime[ContextSchema]):
    user_messages = [
        msg for msg in state.messages if isinstance(msg, HumanMessage) and not hasattr(msg, "name")
    ]
    
    most_recent_user_query = user_messages[-1].content if user_messages else ""
    
    conversation_history = format_history(state.messages)
    
    response = runtime.context.model.invoke(
        SYNTHESIS_PROMPT.format(
            most_recent_user_query=most_recent_user_query,
            conversation_history=conversation_history
        )
    )

    return {"messages": [response]}

# Limit the router to set number of iterations
def router(state: AgentState):
    if state.iteration_count >= Config.MAX_ITERATIONS:
        # If max iterations reached, force the flow to the Synthesizer to provide a partial answer
        return AgentName.SYNTHESIZER 
    return state.next_agent


# Create the complete agent workflow graph
# Returns the compiled StateGraph as a Runtime-executable workflow
def create_agent():
    
    #pricing agent. assign tool
    price_agent_node = create_worker_node(
        AgentName.PRICE_ANALYST, [get_historical_stock_price]
    )
    #filing agent. assign tool
    filing_agent_node = create_worker_node(
        AgentName.FILING_ANALYST, [fetch_sec_filing_sections]
    )
    
    #memory and context for workflow
    graph = StateGraph[AgentState, ContextSchema, AgentState, AgentState](
        AgentState, ContextSchema
    )
    
    #Add node for each agent. Mapping functions
    graph.add_node(AgentName.SUPERVISOR, supervisor_node)
    graph.add_node(AgentName.PRICE_ANALYST, price_agent_node)
    graph.add_node(AgentName.FILING_ANALYST, filing_agent_node)
    graph.add_node(AgentName.SYNTHESIZER, synthesizer_node)
    
    graph.set_entry_point(AgentName.SUPERVISOR)
    graph.add_conditional_edges(
        AgentName.SUPERVISOR,
        router,
        {
            AgentName.PRICE_ANALYST: AgentName.PRICE_ANALYST,
            AgentName.FILING_ANALYST: AgentName.FILING_ANALYST,
            AgentName.SYNTHESIZER: AgentName.SYNTHESIZER
        }
    )
    #Price Analyst and Filing analyst call supervisor once they complete their tasks. 
    graph.add_edge(AgentName.PRICE_ANALYST, AgentName.SUPERVISOR)
    graph.add_edge(AgentName.FILING_ANALYST, AgentName.SUPERVISOR)
    graph.add_edge(AgentName.SYNTHESIZER, END)
    
    return graph.compile()