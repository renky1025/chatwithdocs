from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain_community.llms import Ollama
import qdrant_client
import streamlit as st
from typing import Literal
from dataclasses import dataclass
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.document_loaders import WebBaseLoader
import base64, os
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain.text_splitter import SpacyTextSplitter
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


def load_image(img):
    with open(img, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')

def findAllFile(base, ext):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.'+ext):
                fullname = os.path.join(root, f)
                yield fullname

QDRANT_HOST="http://192.168.1.96:6334"
COLLECTION_NAME="luxunv2"

def insertData():
    # Creating a persistant DB
    client = qdrant_client.QdrantClient(
        url = QDRANT_HOST,
        prefer_grpc=True,
    )


    # create_collection
    collection_name = COLLECTION_NAME
    vector_config = qdrant_client.http.models.VectorParams(
        size = 1024,
        distance = qdrant_client.http.models.Distance.COSINE
        # distance = qdrant_client.http.models.Distance.DOT
    )
    client.recreate_collection(
        collection_name = collection_name,
        vectors_config = vector_config,
    )
    # print(client)

    #  load web data
    # web_links = ["https://hotels-ten.vercel.app/api/hotels"] 
    # loader = WebBaseLoader(web_links)
    # document=loader.load()
    # load pdf file
    
    # document = []
    # files = findAllFile("D:\\ai-test\\local-rag-llamaindex-qdrant\\data\\luxun", "pdf")
    # with tqdm(total=len(list(files))) as pbar:
    #     pbar.set_description('PDF Processing:')
    #     i = 0
    #     for file_path in files:
    #         loader = PyPDFLoader(file_path)
    #         docs = loader.load()
    #         document += docs
    #         i+=1
    #         pbar.update(i)
            
    loader = DirectoryLoader("D:\\ai-test\\local-rag-llamaindex-qdrant\\data\\luxun", glob="*.pdf",loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
    document = loader.load()
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = 1000, chunk_overlap=20)
    texts = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
    logging.info(" embedding docs ! =========>")

    vector_store = Qdrant(
        client=client,
        collection_name = collection_name,
        embeddings=embeddings
    )
    # vector_store.add_documents(texts)
    vector_store.add_documents(texts)
    logging.info(" add vectors completed! <=========")
    retriever=vector_store.as_retriever()

def load_db():
    client = qdrant_client.QdrantClient(
        url=QDRANT_HOST,
        prefer_grpc=True,
    )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    vector_store = Qdrant(
        client = client,
        collection_name = COLLECTION_NAME,
        embeddings = embeddings
    )
    logging.info("connection established !")
    return vector_store

def initialize_session_state() :
    vector_store = load_db()
    # Initialize a session state to track whether the initial message has been sent
    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False

    # Initialize a session state to store the input field value
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    if "chain" not in st.session_state :  

        #create custom prompt for your use case
        prompt_template = """
        You are Jarvis, you are a know-it-all robot who answers all questions.

        You will be given a context of the conversation made so far followed by a customer's question, 
        give the answer to the question using the context. 
        The answer should be short, straight and to the point. If you don't know the answer, reply that the answer is not available.

        Context: {context}

        Question: {question}
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain_type_kwargs = { "prompt" : PROMPT }
        #build your LLM
        llm = Ollama(model="qwen")
        #build your chain for RAG+C
        template = (
                """Combine the chat history and follow up question into 
                a standalone question. 
                If chat hsitory is empty, use the follow up question as it is.
                Chat History: {chat_history}
                Follow up question: {question}"""
            )
        # TRY TO ADD THE INPUT VARIABLES
        prompt = PromptTemplate.from_template(template)
        # question_generator_chain = LLMChain(llm=llm, prompt=prompt)
        logging.info("vector store loaded !")
        st.session_state.chain = ConversationalRetrievalChain.from_llm(     
            llm = llm,
            chain_type = "stuff",
            memory = ConversationSummaryMemory(llm = llm, memory_key='chat_history', input_key='question', output_key= 'answer', return_messages=True),
            retriever = vector_store.as_retriever(search_type="mmr"),
            condense_question_prompt = prompt,
            return_source_documents=False,
            combine_docs_chain_kwargs=chain_type_kwargs,
        )
        

# Streamlit header
st.set_page_config(page_title="Ollama:Chat - An LLM-powered chat bot")
st.title("Ollama-Bot")
st.write("This is a chatbot for anything (Knowledge base is limited xxx)")
# st.write("(Knowledge base is limited to Sheraton Hotel and using trial api key of cohere.)")

# Defining message class
@dataclass
class Message :
    """Class for keepiong track of chat Message."""
    origin : Literal["Customer","elsa"]
    Message : "str"


# laodinf styles.css
def load_css():
    with open("static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        # st.write(css)
        st.markdown(css, unsafe_allow_html = True)

#Callblack function which when activated calls all the other
#functions 
def on_click_callback():

    load_css()
    customer_prompt = st.session_state.customer_prompt

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):

            llm_response = st.session_state.chain(
                {"context": st.session_state.chain.memory.buffer, "question": customer_prompt}, return_only_outputs=True)
            
         

    st.session_state.history.append(
        Message("customer", customer_prompt)
    )
    st.session_state.history.append(
        Message("AI", llm_response)
    )



def main():
    initialize_session_state()
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    aiimg = load_image('./static/elsa.png')
    userimg = load_image('./static/admin.png')
    with chat_placeholder:
        for chat in st.session_state.history:
            if type(chat.Message) is dict:
                msg = chat.Message['answer']
            else:
                msg = chat.Message 
            div = f"""
            <div class = "chatRow 
            {'' if chat.origin == 'AI' else 'rowReverse'}">
                <img class="chatIcon" src = "data:image/png;base64,{aiimg if chat.origin == 'AI' else userimg}" width=32 height=32>
                <div class = "chatBubble {'adminBubble' if chat.origin == 'AI' else 'humanBubble'}">&#8203; {msg}</div>
            </div>"""
            st.markdown(div, unsafe_allow_html=True)

    with st.form(key="chat_form"):
        cols = st.columns((6, 1))
        
        # Display the initial message if it hasn't been sent yet
        if not st.session_state.initial_message_sent:
            cols[0].text_input(
                "Chat",
                placeholder="Hello, how can I assist you?",
                label_visibility="collapsed",
                key="customer_prompt",
            )  
        else:
            cols[0].text_input(
                "Chat",
                value=st.session_state.input_value,
                label_visibility="collapsed",
                key="customer_prompt",
            )

        cols[1].form_submit_button(
            "Ask",
            type="secondary",
            on_click=on_click_callback,
        )

        

    # Update the session state variable when the input field changes
    st.session_state.input_value = cols[0].text_input


if __name__ == "__main__":
    #insertData() # prepare data
    main()# run app: streamlit run qdrant_app.py