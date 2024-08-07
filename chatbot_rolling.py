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
        separators=['(VOLUME\s+\d+\s+-\s+Pag\s+\d+\s+)'],
        is_separator_regex=True,
        keep_separator=True,
        chunk_size=2048, 
        chunk_overlap=256,
        )
    chunk_OK = False
    ajuste = 0
    cont = 0
    print('Chunking: procesando texto...')
    while not chunk_OK:
        try:
            chunks = text_splitter.split_text(text[-(len(text)-ajuste):])
            print('Texto procesado correctamente')
            chunk_OK = True
        except:
            cont += 1
            print(f'Ajustando largo de texto. Iteración {cont}')
            ajuste += 50
        
    chunks = [chunk for chunk in chunks if len(chunk)>50]
    
    return chunks

# vector store chromadb
def get_vector_store(text_chunks):
    Chroma.from_texts(texts=text_chunks,
                          embedding=OpenAIEmbeddings(
                              model='text-embedding-3-large'), 
                          persist_directory="./chroma_db",
                          )
    
# vector store pinecone
def vector_store_pinecone(text_chunks):
    PineconeVectorStore.from_documents(
        documents=text_chunks, 
        embedding=OpenAIEmbeddings(), 
        index_name=os.getenv('PINECONE_INDEX_NAME'))
    
# # CREA BARRA LATERAL
# with st.sidebar:
#         st.image("logo_TN_small.png")
#         
#         st.title("Pasos:")
#         
#         pdf_docs = st.file_uploader('Cargue el manual de interés y # haga click en "Enviar & Procesar"', 
#                                     accept_multiple_files=True, 
#                                     key="pdf_uploader")
#        
#         if st.button("Enviar & Procesar", 
#                      key="process_button"):
#             with st.spinner("Procesando..."):
#                 print(pdf_docs)
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 get_vector_store(text_chunks)
#                 st.success("Documento procesado")


# Crea layout para el encabezado en la página principal
col1, col2 = st.columns([1, 5])

with col1:
    st.image("logo_TN_small.png")

with col2:
    st.header('Chatbot con manuales de operación - GPT 4o')

with st.expander('Instrucciones de uso'):

    st.markdown("""

Este chatbot esta diseñado para interactuar con los manuales de operación y mantenimiento de los motores para generación eléctrica marca Hyundai, ubicados en la planta de Termonorte, en Santa Marta, Colombia.
            
### Cómo funciona:

Siga los siguientes pasos para interactuar con el chatbot:

1. **Manuales**: El sistema tiene cargado en la base de datos los siguientes manuales:
    * VOLUME 1 - INSTRUCTION BOOK VOL I
    * VOLUME 2 - INSTRUCTION BOOK VOL II
    * VOLUME 3 - INSTRUCTION BOOK VOL III
    * VOLUME 4 - INSTRUCTION BOOK VOL IV
    * VOLUME 6 - MECHANICAL EQUIPMENT II
    * VOLUME 7 - MECHANICAL EQUIPMENT III
    * VOLUME 8 - MECHANICAL EQUIPMENT IV
    * VOLUME 9 - MECHANICAL EQUIPMENT V
    * VOLUME 10 - ELECTRICAL & CONTROL I
    
2. **Haga sus preguntas**: En el área designada 'Realice una pregunta acerca de los manuales', haga cualquier pregunta relacionada con el contenido de los documentos mencionados en le punto anterior.
""")

global_llm = "gpt-4o"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_source(context, n=10):
    all_pages = []

    for doc in context:
        print('**'*25)
        page = ((re.findall('(VOLUME\s+\d+\s+-\s+Pag\s+\d+)', doc.page_content))) 
        print(page)
        all_pages.extend(page)

    source_pages = '  - ' + '\n\n  - '.join([pg for pg in list(sorted(set(all_pages)))[:n]])
    
    return source_pages


def get_conversational_chain(retriever):
    template = """Use the following pieces of context to answer the question at the end. Always answer the question in the language in which it is asked. If you don't know the answer, just say that you don't know, don't try to make up an answer. Give your responses primarily in numbered lists. 
    
    If the question is related to many documents in de context, try to include the reference for the main titles in your response.
    
    Example:

    "question: procedure for the maintenance of the fuel injection pump?"

    "answer: The procedure is detailed in various documents. Here is a summary of the main steps:
    1. Disassemble de fuel injection pump (VOLUME 1 - Pag 351)
        - Step 1
        - Step 2
    2. Clean fileter (VOLUME 10 - Pag 1322)
        - Step 1
        - Step 2"
    
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
    
    embeddings = OpenAIEmbeddings(
        model='text-embedding-3-large'
        )
    
    new_db = Chroma(persist_directory="./chroma_db",
                    embedding_function=embeddings,
                    )
    
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever = new_db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 25},
            ), 
        llm = ChatOpenAI(
            model=global_llm, 
            temperature=0,
            ),
    )
    
    chain = get_conversational_chain(multi_retriever)
    
    try:
        response = chain.invoke(user_question)
    except:
        response = 'La consulta tiene un contexto muy grande, por favor sea más específico. (The qustion has a broad context, please be more specific)'
    
    print(response)
    
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
