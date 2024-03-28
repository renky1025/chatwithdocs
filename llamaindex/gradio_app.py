#!/usr/bin/python
# -*- coding: UTF-8 -*-
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    download_loader,
    Prompt,
    Document,
    PromptHelper,
)
from llama_index.core.node_parser import SentenceSplitter
import yaml
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from langchain.text_splitter import SpacyTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.ollama import Ollama
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

class my_app:
    def __init__(self) -> None:
        self.chat_history: list = []
        config_file = "config.yml"
        with open(config_file, "r") as conf:
            self.config = yaml.safe_load(conf)

        self.embed_model =  LangchainEmbedding(HuggingFaceEmbeddings(model_name=self.config["embedding_model"]))
        self.llm = Ollama(model=self.config["llm_name"], base_url=self.config["llm_url"], request_timeout=300)
        # max_input_size = 4096
        # num_output = 256
        # max_chunk_overlap = 20
        # self.prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    def qdrant_client(self):
        #create a qdrant client
        client = qdrant_client.QdrantClient(url=self.config["qdrant_url"],prefer_grpc=True,)
        client._client.openapi_client.client._client.timeout = 60  # seconds
        qdrant_vector_store = QdrantVectorStore(
            prefer_grpc=True,
            client=client,
            collection_name=self.config["collection_name"],
        )
        return qdrant_vector_store

    def process_file(self,file: str = "") -> None:
        logging.info("starting file processing ========>")
        text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm", chunk_size = self.config["chunk_size"])
        if len(file)>0:
            PDFReader = download_loader("PDFReader")
            loader = PDFReader()
            documents = loader.load_data(file=file)
        else:
            documents = SimpleDirectoryReader(self.config["data_path"], required_exts=['.pdf', '.PDF']).load_data(show_progress=True)
        #documents = SimpleDirectoryReader(self.config["data_path"], required_exts=['.txt']).load_data()
        logging.info(f" docs is a {type(documents)}, of length {len(documents)}, where each element is a {type(documents[0])} object")
        documents_v2 = []
        #text_splitter = SentenceSplitter(chunk_size=200, chunk_overlap=20)
        # prompt_helper = PromptHelper(
        #     context_window=4096,
        #     num_output=256,
        #     chunk_overlap_ratio=0.1,
        #     chunk_size_limit=None,
        # )
        for doc in documents:
            if len(doc.text)>0:
                if len(doc.text)>self.config["chunk_size"]:
                    docs = text_splitter.split_text(doc.text)
                    documents_v2 += [Document(text=t,metadata= doc.metadata,) for t in docs]
                else:
                    documents_v2+=[doc]

        if len(documents_v2) ==0:
            raise Exception("Sorry, no documents were found")
        qdrant_vector_store = self.qdrant_client()
        storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
        service_context = ServiceContext.from_defaults(
            llm=None,
            embed_model=self.embed_model,
            chunk_size=self.config["chunk_size"]
        )
        ## save the vectors
        VectorStoreIndex.from_documents(
            documents_v2,
            storage_context=storage_context, service_context=service_context,show_progress=True
        )
        print("Documents saved successfully <=======")

    def load_embedder(self):
        embed_model = HuggingFaceEmbeddings(model_name=self.config['embedding_model'])
        return embed_model

    def qdrant_index(self) -> VectorStoreIndex:
        #embed_model = self.load_embedder()
        # client = qdrant_client.QdrantClient(url=self.config["qdrant_url"], prefer_grpc=True)
        # client._client.openapi_client.client._client.timeout = 60  # seconds
        # qdrant_vector_store = QdrantVectorStore(client=client, collection_name=self.config['collection_name'])
        qdrant_vector_store = self.qdrant_client()
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=self.embed_model, chunk_size=self.config["chunk_size"]
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store=qdrant_vector_store, service_context=service_context
        )
        return index

    def query_something(self, query_str):
        DEFAULT_TEXT_QA_PROMPT_TMPL = (
                "Context information is below. \n"
                "---------------------\n"
                "{context_str}"
                "\n---------------------\n"
                "Given the context information and not prior knowledge, "
                "answer the question: {query_str}\n"
            )
        QA_PROMPT = Prompt(DEFAULT_TEXT_QA_PROMPT_TMPL)
        #query_engine = index.as_query_engine(text_qa_template=QA_PROMPT, response_mode="tree_summarize")
        query_index = self.qdrant_index()
        query_engine = query_index.as_query_engine(text_qa_template=QA_PROMPT,
                                                #similarity_top_k=query.similarity_top_k,
                                                response_mode="tree_summarize",
                                                )
        response = query_engine.query(query_str)
        return response

def get_response(history, query, file):
        # add files to index
        #llamaIndexApp.process_file(file)
        result = llamaIndexApp.query_something(query)
        llamaIndexApp.chat_history += [(query, result)]
        logging.info(result)
        #history.append(result)
        #yield history,""
        for char in str(result):
           history[-1][-1] += char
           yield history,''

import fitz
from PIL import Image
def add_text(history, text: str):
    if not text:
         raise gr.Error('enter text')
    history = history + [(text,'')]
    return history

def render_file(file):
        doc = fitz.open(file.name)
        page = doc[2]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image

def render_first(file):
        doc = fitz.open(file.name)
        page = doc[2]
        #Render the page as a PNG image with a resolution of 300 DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image,[]

llamaIndexApp = my_app()
#get_response("鲁迅写了哪些小说")

#llamaIndexApp.process_file()

import gradio as gr
with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot').style(height=650)
            show_img = gr.Image(label='Upload PDF', tool='select' ).style(height=680)
    with gr.Row():
        with gr.Column(scale=0.60):
            txt = gr.Textbox(
                        show_label=False,
                        placeholder="Enter text and press enter",
                    ).style(container=False)
        with gr.Column(scale=0.20):
            submit_btn = gr.Button('submit')
        with gr.Column(scale=0.20):
            btn = gr.UploadButton("📁 upload a PDF", file_types=[".pdf"]).style()

    btn.upload(
            fn=render_first, 
            inputs=[btn], 
            outputs=[show_img,chatbot],)

    submit_btn.click(
            fn=add_text, 
            inputs=[chatbot,txt], 
            outputs=[chatbot, ], 
            queue=False).success(
            fn=get_response,
            inputs = [chatbot, txt, btn],
            outputs = [chatbot,txt])
    #         .success(
    #         fn=render_file,
    #         inputs = [btn], 
    #         outputs=[show_img]
    # )
##demo = gr.Interface(fn=get_response, inputs="text", outputs="textbox")
if __name__ == "__main__":
    demo.queue()
    demo.launch()