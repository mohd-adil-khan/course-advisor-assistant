from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_chroma import Chroma
import os

file_path=r"\trialsyllabus.pdf"
data =UnstructuredPDFLoader(file_path)
content=data.load()

HF_token="please put your hugging face token here"
os.environ['HUGGINGFACEHUB_API_TOKEN']=HF_token

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_token , model_name='thenlper/gte-large')
vectordb=Chroma(embedding_function=embeddings)

parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
child_splitter=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=0)

retriever= ParentDocumentRetriever(
    vectorstore=vectordb ,
    docstore=InMemoryByteStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

retriever.add_documents(content,ids=None)

##Reranking
import cohere
from cohere import Client
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_cohere import CohereRerank
# cohere API KEY
api_key="put your cohere API key here"
from langchain.retrievers.document_compressors import CohereRerank
cohere_client = cohere.Client(api_key=api_key)

compressor = CohereRerank(client=cohere_client)
comp_retriever=ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever)

template="""
<|system|>
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers.Please tell 'I dont know' if user query is not  in CONTEXT
The context contains courses with their syllabus , based on syllabus similiarity return course code, name
<|user|>

CONTEXT :{context}
</s>
<|user|>
{query}
</s>
<|assistant|>
"""
from langchain.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_template(template)
model =HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature":0.1,"max_new_token":512, "max_length":64})

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"context": comp_retriever, "query": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


import streamlit as st
from PIL import Image

logo = Image.open("IITG_logo.png")
col1, col2 = st.columns([1, 6]) 
with col1:
    st.image(logo, width=80) 
with col2:
    st.markdown("<h1 style='text-align: left;'>Course Advisor Assistant</h1>", unsafe_allow_html=True)

# Instructions section
st.markdown("""
### Instructions:
1. **Step 1:** Enter the area of interest.
2. **Step 2:** Click on "Get Course Recommendation" Button.
3. **Step 3:** Check for familiar courses in the list of courses offered by IIT-G.
""", unsafe_allow_html=True)

st.markdown("""
<div style="background-color:#f8d7da; padding:10px; border-radius:5px; border-left: 5px solid #f5c2c7;">
    <strong>Warning:</strong> Please remember this advisor is in trial phase. Recheck the advises at your own level.
</div>
""", unsafe_allow_html=True)

# Custom styles for text areas and buttons
st.markdown("""
    <style>
    .stTextArea > label {
        font-size: 18px;  /* Uniform font size for labels */
    }
    .stTextArea > div > textarea {
        padding: 10px;         /* Adjust padding for height */
        font-size: 16px;       /* Adjust font size */
        width: 100%;           /* Ensure full width */
        box-sizing: border-box;
        height: 100px;         /* Set a fixed height to allow text wrapping */
        overflow-y: auto;      /* Enable vertical scroll if needed */
        white-space: pre-wrap; /* Ensure text wrapping */
    }
    .stButton > button {
        padding: 15px;         /* Consistent padding */
        font-size: 16px;       /* Adjust font size */
        width: 100%;           /* Ensure full width */
        background-color: #4CAF50; /* Green background color */
        color: white;          /* White text color */
        border: none;          /* Remove default border */
        border-radius: 5px;    /* Rounded corners */
        cursor: pointer;       /* Pointer cursor on hover */
        font-weight: bold;     /* Make text bold */
    }
    </style>
    """, unsafe_allow_html=True)

# Section 1: Enter research domain
st.markdown("""
<div style="background-color:#e3f2fd; padding:15px; border-radius:5px; margin-bottom:20px;">
    <h2 style='font-size: 24px;'>What do you want to learn:</h2>
    """, unsafe_allow_html=True)

research_domain = st.text_area(
    "Enter the research interest or topics to study here",
    height=100,  
    max_chars=256, 
    placeholder="Type your research domain or topic here..."
)

# Section 2: Recommended
st.markdown("""
<div style="background-color:#f1f8e9; padding:15px; border-radius:5px; margin-bottom:20px;">
    <h3 style='font-size: 22px;'>Course Recommendations:</h3>
    """, unsafe_allow_html=True)
submit = st.button("Get Course Recommendation")

if submit:
    st.write(f"**Processing the input, have patience**")
    response=chain.invoke(research_domain)
    st.write(response.split("<|assistant|>")[-1])
else:
    st.write("Idle Status")
st.markdown("</div>", unsafe_allow_html=True)



