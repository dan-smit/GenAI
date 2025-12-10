import uuid

import streamlit as st
from dotenv import load_dotenv
from edgar import set_identity
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

from stockanalyzer.crew import AgentName, AgentState, ContextSchema, create_agent
from stockanalyzer.config import Config, ModelProvider
import os

#to run the app, type the following in the terminal: streamlit run main.py


load_dotenv("vars.env")

st.set_page_config(page_title="Financial Analysis")

st.subheader("Private AI assistant for financial analysis")

#st.cache_resource stores objects across all users, sessions, and reruns. Can only be used on hashable objects
#Here we hash the workflow and model so they are not recreated on every user interaction
@st.cache_resource
def create_workflow():
    return create_agent()

#Cache the LLM model so it is not recreated on every user interaction
@st.cache_resource
def create_model():
    if Config.MODEL.provider == ModelProvider.GOOGLE_GENAI and not os.getenv("GEMINI_API_KEY"):
        st.error("FATAL ERROR: GEMINI_API_KEY environment variable is not set. Cannot initialize Google GenAI model.")
    
    parameters = {
        "temperature": Config.MODEL.temperature,
        "thinking_budget": 0
    }
    if Config.MODEL.provider == ModelProvider.OLLAMA:
        parameters["num_ctx"] = Config.CONTEXT_WINDOW
        
    return init_chat_model(
        f"{Config.MODEL.provider.value}:{Config.MODEL.name}",
        **parameters
    )
    
#st.session_state stores variables for user's sessions, allowing them to persist across script reruns
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "email" not in st.session_state:
    st.session_state.email = None
    
if not st.session_state.email:
    st.info("Welcome! To get started, please enter your email address below.")
    email_input = st.text_input(
        "Your Email Address",
        placeholder="example@domain.com",
        help="Required to access SEC EDGAR filings"
    )

    if email_input:
        if "@" in email_input and "." in email_input.split("@")[1]:
            st.session_state.email = email_input
            set_identity(email_input)
            st.success("Email set successfully! Reloading...")
            st.rerun()
        else:
            st.error("Please enter a valid email address")
    st.stop()
else:
    set_identity(st.session_state.email)
    
with st.sidebar:
    st.header("Conversation")
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.session_state.messages:
        st.caption(f"Messages in conversation: {len(st.session_state.messages)}")
        

#Escape dollar signs in markdown to prevent rendering issues
def escape_markdown(text: str) -> str:
    return text.replace("$", r"\$")


#Assign avatars and roles to messages and display them in the chat interface
for message in st.session_state.messages:
    avatar = None
    if isinstance(message, HumanMessage):
        avatar = ":material/person:"
        role = "user"
    elif isinstance(message, AIMessage):
        avatar = ":material/smart_toy:"
        role = "assistant"
    else:
        continue

    with st.chat_message(role, avatar=avatar):
        st.markdown(escape_markdown(message.content))
        
#Accept user input and process it through the agent workflow
if prompt := st.chat_input(
    "Ask about a public company's SEC filings and stock prices (e.g. 'Analyze Apple stock')"
    ):
    # Append user message to the session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)
        
    # Initialize the agent state with the current conversation messages
    initial_state: AgentState = AgentState(messages=st.session_state.messages.copy())
    
    # Process the user input through the agent workflow and stream responses
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        status_container = st.status("**Analyzing your request...**", expanded=True)
        final_answer_placeholder = st.empty()
        
        current_agent = None
        agent_containers = {}
        agent_content = {}
        
        # Workflow is created and cached above
        workflow = create_workflow() #create_workflow[AgentState, ContextSchema, AgentState, AgentState]()
        
        # Stream the workflow output in "messages" 
        # initial_state is the input to the workflow containing the user query and conversation history
        # context contains the LLM model to be used by the agents in the workflow
        # stream_mode="messages" streams the output as message chunks from the agents
        # message_chunk is a message object from the agent (HumanMessage, AIMessage, or ToolMessage)
        # metadata contains information about the message chunk, including which agent produced it
        for message_chunk, metadata in workflow.stream(
            initial_state,
            context=ContextSchema(model=create_model()),
            stream_mode="messages"
        ):
            # Extract agent name from metadata
            agent_name = metadata.get("langgraph_node", "Unknown")
            
            # Skip empty message chunks
            if not hasattr(message_chunk, "content") or not message_chunk.content:
                continue
            
            # Convert message content to string
            content = str(message_chunk.content)
            
            #Determine when the agent changes to create new sections in the UI
            if agent_name != current_agent:
                current_agent = agent_name
                agent_content[agent_name] = ""
                
                with status_container:
                    if agent_name in [
                        AgentName.SUPERVISOR,
                        AgentName.PRICE_ANALYST,
                        AgentName.FILING_ANALYST
                    ]:
                        st.markdown(f"**{agent_name}**")
                        agent_containers[agent_name] = st.empty()
                    elif agent_name == AgentName.SYNTHESIZER:
                        status_container.update(
                            label="**Synthesizing Final Answer**"
                        )
                        agent_containers[agent_name] = st.empty()
                
            # Append agent content for later display
            agent_content[agent_name] += content
            
            if agent_name in agent_containers:
                if agent_name == AgentName.SYNTHESIZER:
                    final_answer_placeholder.markdown(
                        "### Analysis\n" + escape_markdown(agent_content[agent_name])
                    )
                else:
                    with status_container:
                        if agent_name == AgentName.SUPERVISOR:
                            agent_containers[agent_name].info(
                                escape_markdown(agent_content[agent_name])
                            )
                        else:
                            agent_containers[agent_name].markdown(
                                escape_markdown(agent_content[agent_name])
                            )
        
        status_container.update(
            label="**Thoughts**",
            state="complete",
            expanded=False
        )

