import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from IPython.display import Markdown as md

# format_data function
def format_data(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# path to the existing chromadb
chroma_db_path = "./chroma_db_"

# Initialize the loaders
loader = PyPDFLoader("./pdf/Leave_No_Context_Behind_Paper.pdf")
# print(type(loader))

# Load and split the documents
data = loader.load_and_split()
text_splitter = NLTKTextSplitter(chunk_size=200, chunk_overlap=200)
chunks = text_splitter.split_documents(data)

# load the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="GOOGLE_API_KEY.txt",
                                               model="models/embedding-001")

# Connect to the existing chromadb and initialize the retriever
db_connection = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k":1})

# Define the chat template
chat_template = ChatPromptTemplate.from_messages([
        SystemMessage(content="""I'm a helpful AI assistant. I can answer your questions based on relevant information I find."""),
        HumanMessagePromptTemplate.from_template("""Answer the following question, considering the provided context:

        Context:
        {context}

        Question:
        {question}

        Answer: """)
])

# Now, define the chat model.
# first of all load the google_api_key.
# google_api_key = open("GOOGLE_API_KEY.txt","r")

chat_model = ChatGoogleGenerativeAI(google_api_key="GOOGLE_API_KEY.txt",
                                    model="gemini-1.5-pro-latest")

# Now, define the output parser
output_parser = StrOutputParser()

# Now, let's build the rag model
rag_chain = (
    {"context":retriever | format_data, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Now, let's create a streamlit app.
st.image("images.png")
st.title("RAG Model: Your Information Retrieval Assistant")
st.write("""
    The Retrieval-Augmented Generation (RAG) model is a powerful technique for information retrieval and question answering. 
    It leverages machine learning to extract relevant information from a document and then uses that information 
    to answer your questions in a comprehensive manner. It's like having a personal AI assistant
    who can access and understand the document for you! Below is the architecture of RAG:
""")

st.image("logo.webp", width=700)
st.title("🗣️ QnA Using RAG Pipeline")
st.write("Referencing : Leave No Context Behind Paper")
user_input = st.text_input("ASK a question...")
button = st.button("Ask")

if button:
    retrieved_data = rag_chain.invoke(user_input)
    st.markdown(retrieved_data, unsafe_allow_html=True)