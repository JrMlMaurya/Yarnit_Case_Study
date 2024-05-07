#Import all the required libraries 

import os
import uvicorn
from fastapi import FastAPI
from openai import OpenAI

#Setting the openai api key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" #replace with your openai api key
client = OpenAI()

#Defining function for the content genration 
def Content_genration(format, topic):

    prompt = f'''
    Genrate a marketing content on the given topic and the given format.

    format:{format}
    topic:{topic}
    '''
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    # response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a skilled marketing content writer like blogs, emails, social media posts etc."},
        {"role": "user","content":prompt}
    ]
    )
    text = completion.choices[0].message.content

    return text

#Building the FastAPI
app = FastAPI()

@app.post("/text")
def get_inputs(format: str, topic:str):

    Output = Content_genration(format, topic)

    Input = {"Format": format,
            "Topic": topic}


    return {"Input":Input, "Output":{"Content": Output}}
