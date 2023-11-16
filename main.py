from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch

from langchain.document_loaders import TextLoader

# Load model
from langchain import LLMChain
from transformers import pipeline
import transformers
from langchain.llms import HuggingFacePipeline
from instruct_pipeline import InstructionTextGenerationPipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline

import torch

def setup_vector_db(file_name)
  # Document Loader
  loader = TextLoader(file_name)
  docs= loader.load()
  text_split = RecursiveCharacterTextSplitter(chunk_size = 100, chunk_overlap = 0)
  text = text_split.split_documents(docs)
  hf_embeddings = HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")
  db = FAISS.from_documents(text, hf_embeddings)
  return db

def setup_model():
  generate_text = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16,
                           trust_remote_code=True, device_map="auto", return_full_text=True)


  hf_pipeline = HuggingFacePipeline(pipeline=generate_text)
  chain = load_qa_chain(hf_pipeline, chain_type="stuff")
  return chain


# Start flask app and set to ngrok
db = setup_vector_db("Turbidity-2013.txt")
chain = setup_model()
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_ansert():
  prompt = request.form['prompt-input']
  print(f"Generating a response of {prompt}")

  docs = db.similarity_search(prompt)
  answer = chain.run(input_documents = docs, question = prompt)

  print("Sending response ...")
  return render_template('index.html', response=answer)


if __name__ == '__main__':
  app.run()
