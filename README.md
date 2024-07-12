# PDF-Chat
## Chat with your PDF docs - RAG Tool

#### Having PDF documents too long to read? This RAG-powered AI tool will help you scan through your PDF documents and get direct answers.

#### Ask anything regarding the files you uploaded and it retrieves answers instantly. 

> No more reading through 1000+ page PDF documents! 


## **How it Works**
-----------------------
> The AI tool implements **Cohere's *Command R+* model** which is RAG-optimized to tackle huge workloads of unstructured data.


> It is then utilised together with another **Cohere product: *embed-english-v3.0*** as the embedding model


### Custom Knowledge Base
> RAG implementation requires a knowledge base (or even a database/dataset) to be the 'source of truth' from which it answers to a user's queries.

> [More on the LLM model]: (https://docs.cohere.com/docs/command-r-plus)

```
from llama_index.core import SimpleDirectoryReader

input_path = input_dir_path

loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()
```


## Chunking
> Involves breaking down large texts into smaller chunks or pieces. This function ofocurse includes a chunk size limit and chunk overlap, which prevents the output from being incomprehensible when concatenated.

> This usually depends on the embedding model to be used

> Chunking imporves retrieval speed and efficiency, as the problem is broken down into sub-problems for processing (Think of recursive algos)

```
from llama_index.core import SimpleDirectoryReader

loader = SimpleDirectoryReader(
            input_dir = input_dir_path2,
            required_exts=[".py", ".ipynb", ".js", ".ts", ".md"],
            recursive=True
        )
docs = loader.load_data()
```



## Embeddings model
> Refers to conversion of user input into numerical vectors that re better understood by computers, ML models.

> Here, we use Cohere's *embed-english-v3.0* as the embedding model.

> [More on the model]: (https://cohere.com/blog/introducing-embed-v3)

```
from llama_index.embeddings.cohere import CohereEmbedding
from transformers import AutoModel, AutoTokenizer

llm = Cohere(api_key=API_KEY, model="command-r-plus")

embed_model = CohereEmbedding(
    cohere_api_key=API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)

def load_embedding_model(model_name="facebook/bart-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name)
    return tokenizer, embedding_model
```

 
## Vector databases
> Storage for vectors that are a numerical representation of the user-generated input

```
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext

```

## Conversational UI
> **Streamlit** offers a simplistic interface for the RAG tool for uploading files and asking questions in a messaging-like UI

```
import streamlit as st

with col1:
    st.header(f"PDF-Chat")
    st.subheader(f"Chat with your PDF documents")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

```
-----------
## Conclusion

 

