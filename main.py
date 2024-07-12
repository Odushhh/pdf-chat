import nest_asyncio
from dotenv import load_dotenv
from IPython.display import Markdown, display

import os
import pypdf
import PyPDF2
from llama_index.core import SimpleDirectoryReader

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader

from llama_index.llms.cohere import Cohere
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

from llama_index.core import SimpleDirectoryReader, PromptHelper, ServiceContext, StorageContext, load_index_from_storage

# allows nested access to the event loop
nest_asyncio.apply()

API_KEY = 'zhMrOjPTMRsBlwEcx58f6rkEluu3WUiHetNW2xEX'

# add your documents in this directory, you can drag & drop
input_dir_path = '/teamspace/studios/this_studio/pdf_files'



# setup llm & embedding model
llm = Cohere(api_key=API_KEY, model="command-r-plus")

embed_model = CohereEmbedding(
    cohere_api_key=API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_query",
)
 

input_dir_path = '/teamspace/studios/this_studio/pdf_files'

if os.path.exists(input_dir_path):
  print(f"Directory '{input_dir_path} 'exists.")

  files = os.listdir(input_dir_path)
  print("Files in the directory: ", files)

  pdf_files = [f for f in files if f.endswith(".pdf")]
  if pdf_files:
    print("PDF files found:", pdf_files)
  else:
    print("No PDF files found.")
else:
  print(f"Directory '{input_dir_path}' does not exist.")


# load data
loader = SimpleDirectoryReader(
            input_dir = input_dir_path,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()

# Creating an index over loaded data
Settings.embed_model = embed_model
index = VectorStoreIndex.from_documents(docs, show_progress=True)

# Create a cohere reranker 
cohere_rerank = CohereRerank(
                        model='rerank-english-v3.0',
                        api_key=API_KEY,
                    )
# Create the query engine, where we use a cohere reranker on the fetched nodes
Settings.llm = llm
query_engine = index.as_query_engine(node_postprocessors=[cohere_rerank])

# Generate the response
response = query_engine.query("What exactly is DSPy?",)


display(Markdown(str(response)))