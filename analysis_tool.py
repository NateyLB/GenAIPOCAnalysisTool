import getpass
import os
import bs4
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv


load_dotenv()
os.environ["OPENAI_API_KEY"]
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

llm = ChatOpenAI(model="gpt-4o-mini")

def upload_embed_store(file):
    """
    Takes in CSV file, splits it into words, embeddes, and stores for retrieval.
    """
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    with open(file.name, "wb") as f:
        f.write(file.getbuffer())
    loader = CSVLoader(file.name)
    pages = loader.load_and_split()
    docs = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    # Might want to do some custom extraction like extract a header or some information here. Uses a prompt to do this Here is an example
    # template = ("Extract all the headings from the following text: {text}. The headings should be "
    #             "in the same order as they appear in the text and should be short and concise."
    #             "Example: `dictionary`, `list comprehension`, `functions`.")

    # def extract_headings(text):
    #     prompt = template.format(text=text.page_content)
    #     result = get_ai_prompt_response(prompt)
    #     return result.strip().split("\n")

    return Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

def st_ui() -> None:
    """
    Generates a user interface that allows a user to upload a .csv file and utilize Generative AI-powered chat interface for data analysis
    """
    st.title('Generative AI-Powered Analysis Tool for Excel Data', anchor=False)
    with st.form("question"):
        uploaded_file = st.file_uploader('File uploader')
        submitted = st.form_submit_button("Generate Questions and answers")
        if submitted:
            # print(file_path.upload_url)
            if uploaded_file is not None:
                vectorstore = upload_embed_store(uploaded_file)
            # st.text(retrieve_and_answer(vectorstore,topic,difficulty, no_questions, type_of_question))

if __name__ == '__main__':
    st_ui()