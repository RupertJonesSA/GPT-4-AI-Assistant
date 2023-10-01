ASTRA_DB_SECURE_BUNDLE_PATH=r"C:\Users\baybl\OneDrive\Documents\Python Scripts\Vector Embeddings\secure-connect-vector-database.zip"
ASTRA_DB_APPLICATION_TOKEN = "AstraCS:DtAxWMIpeSkOQJmKnQhOYscA:0a0bda51453935a8b8e89674f998edd7b05d740fcbbe8d78d63d327a22036ac0" #"AstraCS:..."
ASTRA_DB_CLIENT_ID = "DtAxWMIpeSkOQJmKnQhOYscA"
ASTRA_DB_CLIENT_SECRET = "8QpxKObLyTEUQreZa9mnLrhW,b9EKNy4,negnrvKTn4MLUcqpUduPSZQnK69tt-wOv-LsZx3e2.uewmRmwU.UZDWret3modZ+qsX3p.ZACmHg-,j_QKTYZ_,tA2fL8O," 
ASTRA_DB_KEYSPACE="search"
OPENAI_API_KEY = "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"

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from datasets import load_dataset

# Creating a cluster
# Configuration to be able to communicate with the database created with
# DataStax Astra.
cloud_config= {
   'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH 
}
auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

# Connect to OpenAI using OpenAI API key. 
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
myEmbedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

myCassandraVStore = Cassandra(
    embedding=myEmbedding,
    session=astraSession,
    keyspace=ASTRA_DB_KEYSPACE,
    table_name="qa_mini_demo",
)

print("Loading data from huggingface")
myDataset = load_dataset("Biddls/Onion_News", split="train")
headlines = myDataset["text"][:50]

print("\nGenerating embeddings and storing in AstraDB")
myCassandraVStore.add_texts(headlines)

print("Inserted %i headlnes. \n" % len(headlines))

vectorIndex = VectorStoreIndexWrapper(vectorstore=myCassandraVStore)

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ")
        first_question = False
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ")

    if query_text.lower() == 'quit':
        break

    print("Quesiton: \"s%\"" % query_text)
    answer = vectorIndex.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)

    print("DOCUMENTS BY RELEVANCE:")
    for doc, score in myCassandraVStore.similarity_search_with_score(query_text, k=4):
        print(" %0.4f \"%s ...\"" % (score, doc.page_content[:60]))
