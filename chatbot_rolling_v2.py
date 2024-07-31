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
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
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
import regex as re

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
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['(VOLUME\s*\d+\s*-\s*Pag\s*\d+)'],
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
    

# path =

# crea la barra lateral
with st.sidebar:
        st.image("logo_TN_small.png")
        
        st.title("Manuales:")

        options = st.multiselect(
            "Seleccione los manuales de interés:",
            ["Green", "Yellow", "Red", "Blue"],
            ["Yellow", "Red"])




        st.title("Modelos:")

        modelo = st.radio('Seleccione el modelo:',
                 ['GPT 3.5', 'GPT 4', 'GPT 4o'],
                 index=2,
                 key='llm')
        
        if modelo == 'GPT 3.5':
            llm = 'gpt-3.5-turbo-0125'
        elif modelo == 'GPT 4':
            llm = 'gpt-4-turbo'
        else:
            llm = ''
                      
        
        

# Crea layout para el encabezado en la página principal
col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo_TN_small.png")

with col2:
    st.header('Chatbot con manuales de operación - GPT 4o - V2')

with st.expander('Instrucciones de uso'):

    st.markdown("""

Este chatbot esta diseñado para interactuar con los manuales de operación y mantenimiento de los gensets Hyundai, ubicados en la planta de Termonorte, en Santa Marta, Colombia.
            
### Cómo funciona:

Siga los siguientes pasos para interactuar con el chatbot:

1. **Cargue el documento**: El sistema acepta múltiples archivos PDF a la vez, analizando el contenido para proporcionar información completa. Después de cargar los documentos, haga click en el botón "Enviar & Procesar"

2. **Haga sus preguntas**: Después de procesar los documentos, haga cualquier pregunta relacionada con el contenido de los documentos que ha subido.
""")

global_llm = "gpt-4o"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def get_source(context):
    all_pages = []

    for doc in context:
        page_number = ((re.findall('(Pag\s*\d+)', doc.page_content))) 
        all_pages.extend(page_number)

        source_pages = '  - ' + '\n\n  - '.join(['Volume 1 - ' + pg for pg in sorted(set(all_pages))])
    
        return source_pages



def get_conversational_chain(retriever):
    template = """Use the following pieces of context to answer the question at the end. Always answer the question in the language in which it is asked. If you don't know the answer, just say that you don't know, don't try to make up an answer. Give your responses primarily in numbered lists.
    
    {context}

    Question: {question}

    Helpful Answer:"""
    
    rag_prompt_custom = PromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model=global_llm, 
                     temperature=0,
                     )
    
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | rag_prompt_custom
        | llm
        | StrOutputParser()
        )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, 
         "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def user_input(user_question):
    
    embeddings = OpenAIEmbeddings()
    new_db = Chroma(persist_directory="./chroma_db",
                    embedding_function=embeddings,
                    )
    
    retriever = new_db.as_retriever(
        search_type="mmr",
        )
    
    chain = get_conversational_chain(retriever)
    
    try:
        response=chain.invoke(user_question)
    except:
        response='La consulta tiene un contexto muy grande, por favor sea más específico. (The qustion has a broad context, please be more specific)'
    
    return response

def main():
    st.header("Chatbot de IA")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Realice una pregunta acerca de los manuales", key="user_question")

    if user_question:
        # agrega la siguiente pregunta al stack de mensajes
        st.session_state.messages.append({
            "role": "user", 
            "content": user_question,
            })
        
        # despliega pregunta del usuario en la ventana de chat
        with st.chat_message("user"):
            st.markdown(user_question)
        
        with st.chat_message("assistant"):
            response = user_input(user_question)
            
            response_with_source = f"""
                {response['answer']}
                \n\nFuente (Source):
                \n\n{get_source(response['context'])}
                """

            st.markdown(response_with_source)

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_with_source,
            })

        
if __name__ == "__main__":
    main()
