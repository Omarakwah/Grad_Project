from llama_index.core import SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import os
import logging
import sys

import gradio as gr
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
import torch
from llama_index.core.prompts.prompts import SimpleInputPrompt
from llama_index.core import ServiceContext
import os
from dotenv import load_dotenv, find_dotenv
import openai
import sys
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
reader= SimpleDirectoryReader(
    input_files=["[The Oxford Series in Electrical and Computer Engineering] Ding, Zhi; Lathi, Bhagwandas Pannalal - Modern digital and analog communication systems (2018;2019, Oxford University Press, USA)(Z-Lib.io).pdf"])
docs = reader.load_data()
openai.api_key  = os.environ["OPENAI_API_KEY"]
index = VectorStoreIndex.from_documents(docs)
retriever = index.as_retriever(similarity_top_k=10)
query_engine = index.as_query_engine()
system_prompt = "You are a communication Engineer  assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided." 

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.3, "do_sample": True},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",

    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.bfloat16}
)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    node_parser=SentenceSplitter(chunk_size=512, chunk_overlap=20),
    num_output=512,
    context_window=2048,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
)
def predict(input, history):
  response = query_engine.query(input)
  return str(response)
gr.ChatInterface(predict).launch(share=True)
