from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
import base64
from IPython.display import Image, display
from matplotlib import pyplot as plt
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from env import OPEN_AI_KEY

os.environ["OPENAI_API_KEY"] = OPEN_AI_KEY


template = """Statement: {statement}

For the above statement, break down the sentences into steps
Then, Generate mermaid js structured markup for {type}.

Do not add any special characters in the markup which break the rules to create mermaid markup.
And only return the markup in your response.

"""

template = """Statement: {statement}

1. Break down the Statement into a series of steps in English.
2. Then convert these steps into Mermaidjs structured markup. The diagram may be a {type}.
Keep the following points in mind while generating the markup:
1. Do not add any special characters in the markup which break the rules to create mermaid markup.

Your Response should only contain the generated markup, Do not append or prepend any other word/sentences.
"""
prompt = PromptTemplate(template=template, input_variables=["statement", "type", "orientation"])
llm = OpenAI(model_name="text-davinci-003", temperature=0.9)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def get_flow_diagram(statement):
    input = {
        "statement":statement,
        "orientation": "Top Bottom",
        "type": "graph"
    }

    markup_response = llm_chain.run(input)
    graph_generate(markup_response)


def download_and_display_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Flow Diagram", use_column_width=True)
        else:
            st.error(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")


def graph_generate(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    st.info("https://mermaid.ink/img/" + base64_string)
    flow_diagram_url = "https://mermaid.ink/img/" + base64_string
    download_and_display_image(flow_diagram_url)


st.title('Natural Language Mermaid Flow Diagrams')
with st.form('my_form'):
  statement = st.text_area('Enter text:', 'How to make tea?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    get_flow_diagram(statement)



