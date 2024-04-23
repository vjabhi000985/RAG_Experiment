import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# Replace with your actual API key and model name
google_api_key = "YOUR_GOOGLE_API_KEY"
model_name = "gemini-1.5-pro-latest"

def format_context(docs):
    """Formats retrieved documents into a user-friendly, multi-line context."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(user_query):
    """Builds the RAG chain based on user input."""

    loader = WebBaseLoader("https://www.example.com/your-target-document")  # Replace with your target document URL
    data = loader.load()

    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model=f"models/embedding-001")

    db_connection = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
    db_connection.persist()

    retriever = db_connection.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(user_query)

    chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""I'm a helpful AI assistant. I can answer your questions based on relevant information I find."""),
        HumanMessagePromptTemplate.from_template("""Answer the following question, considering the provided context:

        Context:
        {context}

        Question:
        {question}

        Answer: """)
    ])

    rag_chain = (
        {"context": retriever | format_context, "question": RunnablePassthrough()}
        | chat_template
        | ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model_name)
        | StrOutputParser()
    )

    return rag_chain

st.title("RAG System: Leave No Context Behind")
st.write("This system retrieves information from a document and uses it to answer your questions in a comprehensive manner.")

user_query = st.text_input("Ask me anything about the document:")

if user_query:
    rag_chain = create_rag_chain(user_query)
    response = rag_chain.invoke(user_query)
    st.markdown(response)