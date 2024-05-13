# RAG practice

One basic function project, create with LlamaIndex and LangChain framework
![](images/local-rag-architecture.png)
![](images/86971A7E-F910-4ad2-BA59-A80EB9D1EB66.png)

## Running the project

#### Starting a Qdrant docker instance

```bash
docker run -p 6333:6333 -v ~/qdrant_storage:/qdrant/storage:z qdrant/qdrant
```

#### Downloading & Indexing data

```bash
llamaIndexApp = my_app()
llamaIndexApp.process_file()
```

```bash
    insertData() # prepare data
    #main()# run app: streamlit run qdrant_app.py
```

#### Starting Ollama LLM server

Follow [this article](https://otmaneboughaba.com/posts/local-llm-ollama-huggingface/) for more infos on how to run models from hugging face locally with Ollama.


Start the model server

```bash
ollama run llama2
```

By default, Ollama runs on ```http://localhost:11434```

#### Demo running
update config.yml first and then
```bash
python gradio_app.py
```
or
```bash
streamlit run qdrant_app.py
```

## Example

#### Request

![Post Request](images/912842EB-653C-4085-A1A8-8D8C2E500DEF.png)