import os
import discord
from discord import Intents
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import pinecone 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from langchain.vectorstores import Pinecone
from langchain.prompts.chat import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI

# initializes pincone and fetches api key + pinecone env from .env file, user must also provide an index_name 
pinecone.init(
api_key=os.getenv('PINECONE_API_KEY'),
environment=os.getenv('PINECONE_ENV')
        )
index_name="pyneconeapp"

# gets API key from .env
openai.api_key = os.getenv("OPENAI_API_KEY")

#loads env file 

from flask import Flask, request
app = Flask(__name__)

# Setup Discord client
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.all()
intents.messages = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='question', help='Ask a question about our documentation')
async def ask_question(ctx, *, question):
    answer = get_answer(question)
    await ctx.send(answer)

def process_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, length_function=len)
    chunks = splitter.split_text(content)

    embeddings = OpenAIEmbeddings()
    embeddings_list = embeddings.embed_documents(chunks)

    index = pinecone.Index(index_name)
    index.upsert(vectors=[(f"document-{i}", embedding, {"page_content": chunk}) for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list))])

def get_similar_docs(query, k=5):
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_documents([query])

    index = pinecone.Index(index_name)
    query_result = index.query(queries=[query_embedding[0]], top_k=k, include_metadata=True)

    similar_docs = []
    for match in query_result.results[0].matches:
        id = match.id
        metadata = match.metadata
        if 'page_content' in metadata:
            similar_docs.append(Document(id=id, page_content=metadata['page_content']))

    return similar_docs

def get_answer(query):
    top_documents = get_similar_docs(query, k=5)
    model_name = "gpt-3.5-turbo"
    model = ChatOpenAI(model_name=model_name)

    chat_history = []
    question = HumanMessage(content=str(query)) 

    messages = [SystemMessage(content=doc.page_content) for doc in top_documents] + [question]
    response = model(messages)

    return response.content

# Process and index your documentation
# process_text_file("/path/to/your/documentation.txt")

bot.run(TOKEN)

@bot.command(name='generate', help='Generate text with OpenAI GPT-3')
async def generate_text(ctx, *, prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=200  # Some arbitrary number; adjust as needed
        )
        await ctx.send(response.choices[0].text.strip())
    except Exception as e:
        await ctx.send(str(e))

@bot.command(name='retrieve', help='Retrieve similar documents from Pinecone index')
async def retrieve_text(ctx, *, query):
    try:
        similar_docs = get_similar_docs(query, k=5)
        await ctx.send('\n'.join([doc.page_content for doc in similar_docs]))
    except Exception as e:
        await ctx.send(str(e))

# Run the bot in a separate thread
import threading
threading.Thread(target=bot.run, args=(TOKEN,)).start()

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()

        if 'prompt' not in data or 'max_tokens' not in data:
            return {"error": "Missing required parameters"}, 400

        # Generate the text using OpenAI's GPT-3 model
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=data['prompt'],
            max_tokens=data['max_tokens']
        )

        return {"generated_text": response.choices[0].text.strip()}, 200
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/retrieve', methods=['GET'])
def retrieve_text():
    try:
        query = request.args.get('query')

        if not query:
            return {"error": "No query provided"}, 400

        # Retrieve similar documents from the Pinecone index
        similar_docs = get_similar_docs(query, k=5)

        return {"similar_docs": [doc.text for doc in similar_docs]}, 200
    except Exception as e:
        return {"error": str(e)}, 500
    
if __name__ == "__main__":
        app.run(port=5000)

