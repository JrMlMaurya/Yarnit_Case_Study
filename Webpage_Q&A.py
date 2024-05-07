#Importing all the requires libraries 
import openai
from openai import OpenAI
from langchain_community.vectorstores import FAISS
import pickle
import faiss

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI

from fastapi import FastAPI

#setting up the openai api key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" #replace with your openai api key

#Defining a function for the Quary answering
def Ques_ans(url, Quary):
  urls = [url]

  loader = UnstructuredURLLoader(urls=urls)
  data = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
                                        chunk_size=1000,
                                        chunk_overlap = 200)

  docs = text_splitter.split_documents(data)

  embeddings = OpenAIEmbeddings()

  vectorStore_openAI = FAISS.from_documents(docs, embeddings)

  chain = load_qa_chain(OpenAI(), chain_type = "stuff")

  Ans_score = vectorStore_openAI.similarity_search_with_relevance_scores(Quary)
  Ans = vectorStore_openAI.similarity_search(Quary)

  if Ans_score[0][1]>0.70:
    answer=chain.run(input_documents=Ans, question = Quary)
  else:
    answer = "I don't Know the answer."

  return answer


#Building the fastapi

app =FastAPI()

@app.post("/Ques_ans")
def get_inputs(url: str, Quary: str):

  answer = Ques_ans(url, Quary)

  Input = {"URL": url,
           "Question": Quary}
  Output = {"Answer":answer}

  return {"Input":Input,
          "Output":Output}
