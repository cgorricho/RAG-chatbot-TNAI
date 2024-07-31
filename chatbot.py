### TERMONORTE - PROYECTO DE INTELIGENCIA ARTIFICIAL ###
#
# Archivo base para webapp prototipo
# Chatbot con documentos
#
# Desarrollado por:
# HEPTAGON GenAI | AIML
# Carlos Gorricho
# cel: +57 314 771 0660
# email: cgorricho@heptagongroup.co

### IMPORTAR DEPENDENCIAS ###
import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from typing import List
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pydantic import BaseModel, Field
import dotenv
import os

# Create classes needed for MultiQueryRetrieval with custom prompt
# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


### DEFINICION DE LA PAGINA ###

# Configuración de la página
st.set_page_config(
    page_title = 'TERMONORTE - Chatbot GPT 4o',
    # page_icon = '🏭',
    layout = 'wide'
)

# carga variables de entorno de .env
dotenv.load_dotenv()

### DEFINICION DE FUNCIONES

# define funciones iniciales de procesamiento de texto

# extrae texto de PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# extrae text chunks (pedazos de texto)
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='(-*\s*\nPag\s*\d+)',
        is_separator_regex=True,
        chunk_size=2048, 
        chunk_overlap=256)
    chunks = text_splitter.create_documents([text])
    return chunks

# vector store chromadb
def get_vector_store(text_chunks):
    Chroma.from_documents(documents=text_chunks,
                          embedding=OpenAIEmbeddings(), 
                          persist_directory="./chroma_db",
                          )
    
# vector store pinecone
def vector_store_pinecone(text_chunks):
    PineconeVectorStore.from_documents(
        documents=text_chunks, 
        embedding=OpenAIEmbeddings(), 
        index_name=os.getenv('PINECONE_INDEX_NAME'))

# crea la barra lateral
with st.sidebar:
        st.image("logo_TN_small.png")
        st.title("Pasos:")
        pdf_docs = st.file_uploader('Cargue el manual de interés y haga click en "Enviar & Procesar"', 
                                    accept_multiple_files=True, 
                                    key="pdf_uploader")
        if st.button("Enviar & Procesar", 
                     key="process_button"):
            with st.spinner("Procesando..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                # get_vector_store(text_chunks)
                get_vector_store(text_chunks)
                st.success("Documento procesado")


# Crea layout para el encabezado en la página principal
col1, col2 = st.columns([1, 5])

with col1:
   st.image("logo_TN_small.png")

with col2:
   st.header('Chatbot con manuales de operación - GPT 4o')

st.markdown("""

Este chatbot esta diseñado para interactuar con los manuales de operación y mantenimiento de los gensets Hyundai, ubicados en la planta de Termonorte, en Santa Marta, Colombia.
            
### Cómo funciona:

Siga los siguientes pasos para interactuar con el chatbot:

1. **Cargue el documento**: El sistema acepta múltiples archivos PDF a la vez, analizando el contenido para proporcionar información completa. Después de cargar los documentos, haga click en el botón "Enviar & Procesar"

2. **Haga sus preguntas**: Después de procesar los documentos, haga cualquier pregunta relacionada con el contenido de los documentos que ha subido.
""")



# This is the first API key input; no need to repeat it in the main function.

def get_conversational_chain(retriever):
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Give your responses primarily in numbered lists, when relevant. Always answer the question in the language in which it is asked.
    
    {context}

    Question: {question}

    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o", 
                     temperature=0,
                     )
    chain = {"context": retriever, 
             "question": RunnablePassthrough()} | rag_prompt_custom | llm
    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = Chroma(persist_directory="./chroma_db",
                    embedding_function=embeddings,
                    )
    retriever = new_db.as_retriever()
    chain = get_conversational_chain(retriever)
    response=chain.invoke(user_question)
    st.write("Respuesta: ", "\n",response.content)

def main():
    st.header("Chatbot de IA")

    user_question = st.text_input("Realice una pregunta acerca de los manuales", key="user_question")

    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
