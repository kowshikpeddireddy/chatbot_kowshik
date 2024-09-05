import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "fiass_db")

# Define the embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing vector store with the embedding function
db = FAISS.load_local(persistent_directory, embeddings,allow_dangerous_deserialization=True)


# Create the Streamlit app
st.set_page_config(page_title="kowshik's Chatbot")




st.title("🤖 Kowshik's Chatbot")

# Create the retriever, LLM, and chains outside the input block
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only. Please provide the most accurate response based on the question.
    <context> {context} </context>
    Questions: {input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# User input for the question
user_input = st.text_input ( "wanna know more about kowshik just ask....!", "" )

if st.button("Ask"):
    if user_input:
        # Get the response
        response = retrieval_chain.invoke({'input': user_input})
        # Display the response with improved styling
        st.markdown (
            f"<div style='max-width: 800px; padding: 10px; border-radius: 5px; background-color: #0000; border: 1px solid #d1d1d6;'>{response['answer']}</div>",
            unsafe_allow_html=True )

    else:
        st.warning("Please enter a question.")