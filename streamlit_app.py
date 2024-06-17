import streamlit as st
import os
import nest_asyncio
from llama_index.databricks import Databricks  # Update this line if the path is different
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Streamlit app setup
st.title("Research Agent with Databricks")
st.sidebar.title("Configuration")

# User input for API keys
llama_api_key = st.sidebar.text_input("Llama API Key", type="password")
databricks_api_key = st.sidebar.text_input("Databricks API Key", type="password")
databricks_cluster_id = st.sidebar.text_input("Databricks Cluster ID")

# Set environment variables
os.environ["LLAMA_CLOUD_API_KEY"] = llama_api_key

# Apply nest_asyncio
nest_asyncio.apply()

# Initialize models
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Databricks(
    model="databricks-meta-llama-3-70b-instruct",
    api_key=databricks_api_key,
    api_base=f"https://{databricks_cluster_id}.cloud.databricks.com/serving-endpoints",
)

Settings.llm = llm
Settings.embed_model = embed_model

st.write("Models initialized successfully.")

# File uploader for research papers
uploaded_files = st.file_uploader("Upload Research Papers", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    from llama_parse import LlamaParse
    from llama_index.core.schema import Document

    def _load_data(file_path: str) -> Document:
        parser = LlamaParse(result_type="text")
        json_objs = parser.get_json_result(file_path)
        json_list = json_objs[0]["pages"]
        docs = []
        for item in json_list:
            doc = Document(
                text=item["text"], metadata={"page_label": item["page"]}
            )
            docs.append(doc)
        return docs

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        docs = _load_data(uploaded_file.name)
        st.write(f"Parsed content from {uploaded_file.name}:")
        for doc in docs:
            st.write(doc.text)

# Convert papers to Tools
if uploaded_files:
    from llama_index.core import VectorStoreIndex, SummaryIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.tools import FunctionTool, QueryEngineTool
    from llama_index.core.vector_stores import MetadataFilters, FilterCondition
    from typing import List, Optional

    def get_doc_tools(file_path: str, name: str) -> str:
        """Get vector query and summary query tools from a document."""
        documents = _load_data(file_path)
        splitter = SentenceSplitter(chunk_size=1024)
        nodes = splitter.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(nodes)

        def vector_query(query: str, page_numbers: Optional[List[int]] = None) -> str:
            """Use to answer questions over a given paper."""
            if page_numbers:
                filters = MetadataFilters(
                    filters=[FilterCondition(key="page_label", value=page) for page in page_numbers]
                )
                return vector_index.query(query, filters=filters)
            return vector_index.query(query)

        return vector_query

    for uploaded_file in uploaded_files:
        vector_query_tool = get_doc_tools(uploaded_file.name, uploaded_file.name)
        st.write(f"Vector query tool for {uploaded_file.name} is ready to use.")

# Query input and response display
if uploaded_files:
    query = st.text_input("Enter your query:")
    if query:
        for uploaded_file in uploaded_files:
            vector_query_tool = get_doc_tools(uploaded_file.name, uploaded_file.name)
            response = vector_query_tool(query)
            st.write(f"Response for {uploaded_file.name}:")
            st.write(response)

# Setup an agent over the uploaded papers
if uploaded_files:
    from llama_index.core.agent import ReActAgentWorker, AgentRunner

    paper_to_tools_dict = {}
    for uploaded_file in uploaded_files:
        vector_tool = get_doc_tools(uploaded_file.name, uploaded_file.name)
        paper_to_tools_dict[uploaded_file.name] = [vector_tool]

    initial_tools = [t for tools in paper_to_tools_dict.values() for t in tools]

    agent_worker = ReActAgentWorker.from_tools(
        initial_tools,
        verbose=True
    )
    agent = AgentRunner(agent_worker)

    st.write("Agent setup successfully.")

    # Agent query input and response display
    agent_query = st.text_input("Enter your query for the agent:")
    if agent_query:
        agent_response = agent.query(agent_query)
        st.write("Agent response:")
        st.write(agent_response)
