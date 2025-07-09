import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Neo4jVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
import streamlit as st
import tempfile

def main():
    st.set_page_config(layout="wide", page_title="Graph RAG")
    st.sidebar.header("Graph RAG")
    st.sidebar.image('logo.png', use_column_width=True)

    with st.sidebar.expander("About"):
        st.markdown("""
        Upload a PDF → Convert it to a Neo4j knowledge graph → Ask natural language questions.
        Powered by OpenAI GPT and LangChain.
        """)

    st.title("Graph RAG: Real-time Knowledge Graph App")

    load_dotenv()

    # OpenAI API Key setup
    if 'OPENAI_API_KEY' not in st.session_state:
        st.sidebar.subheader("OpenAI API Key")
        api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            st.session_state['OPENAI_API_KEY'] = api_key
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model_name="gpt-4o")
            st.session_state['embeddings'] = embeddings
            st.session_state['llm'] = llm
    else:
        embeddings = st.session_state['embeddings']
        llm = st.session_state['llm']

    # Neo4j connection setup
    graph = None
    if 'neo4j_connected' not in st.session_state:
        st.sidebar.subheader("Connect to Neo4j")
        neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
        neo4j_username = st.sidebar.text_input("Username:", value="neo4j")
        neo4j_password = st.sidebar.text_input("Password:", type='password')
        connect = st.sidebar.button("Connect")
        if connect and neo4j_password:
            try:
                graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                st.session_state['neo4j_url'] = neo4j_url
                st.session_state['neo4j_username'] = neo4j_username
                st.session_state['neo4j_password'] = neo4j_password
                st.sidebar.success("Connected to Neo4j database.")
            except Exception as e:
                st.error(f"Neo4j connection failed: {e}")
    else:
        graph = st.session_state['graph']

    # Proceed only if connected
    if graph:
        uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

        if uploaded_pdf and 'qa_chain' not in st.session_state:
            with st.spinner("Processing PDF..."):
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_pdf_path = tmp_file.name

                loader = PyPDFLoader(tmp_pdf_path)
                pages = loader.load_and_split()
                splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                docs = splitter.split_documents(pages)

                # Prepare LangChain Documents
                langchain_docs = [Document(page_content=doc.page_content.replace("\n", ""), metadata={'source': uploaded_pdf.name}) for doc in docs]

                # Clear Neo4j Graph
                graph.query("MATCH (n) DETACH DELETE n;")

                # Allowed schema
                allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
                allowed_relationships = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]

                # Transform to graph documents
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    node_properties=False,
                    relationship_properties=False
                )

                graph_docs = transformer.convert_to_graph_documents(langchain_docs)
                graph.add_graph_documents(graph_docs, include_source=True)

                # Create vector index
                Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=st.session_state['neo4j_url'],
                    username=st.session_state['neo4j_username'],
                    password=st.session_state['neo4j_password'],
                    database="neo4j",
                    node_label="Patient",
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding",
                    index_name="vector_index",
                    keyword_index_name="entity_index",
                    search_type="hybrid"
                )

                st.success(f"Finished preparing {uploaded_pdf.name}")

                # GraphCypherQAChain setup
                schema = graph.get_schema
                prompt = PromptTemplate(
                    template="""
                    Task: Generate a Cypher query for the following question.

                    Schema:
                    {schema}

                    Rules:
                    - Use only the above schema.
                    - Output only the Cypher query.

                    Question: {question}
                    """,
                    input_variables=["schema", "question"]
                )

                qa_chain = GraphCypherQAChain.from_llm(
                    llm=llm,
                    graph=graph,
                    cypher_prompt=prompt,
                    verbose=True,
                    allow_dangerous_requests=True
                )

                st.session_state['qa_chain'] = qa_chain

    else:
        st.warning("Connect to Neo4j first.")

    # Question answering
    if 'qa_chain' in st.session_state:
        st.subheader("Ask a Question")
        with st.form(key='question_form'):
            question = st.text_input("Type your question:")
            submit = st.form_submit_button("Submit")

        if submit and question:
            with st.spinner("Generating answer..."):
                result = st.session_state['qa_chain'].invoke({"query": question})
                st.write("### Answer:")
                st.write(result['result'])

if __name__ == "__main__":
    main()
