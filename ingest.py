# ingest.py
# PySpark imports
from pyspark.sql import SparkSession

# Other imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import os
import sys
from google.cloud import storage  # GCP client

spark = SparkSession.builder.appName("RAGIngest").getOrCreate()
sc = spark.sparkContext
file_path = sys.argv[1]
output_path = sys.argv[2]

# Load text corpus (can be a single large file or many small ones)
rdd = sc.textFile(file_path)
# rdd = sc.textFile("Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt")

# Define a function to chunk text
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Apply splitting in parallel
chunked_rdd = rdd.flatMap(lambda line: split_text(line))

# Remove chunks that are empty
chunked_rdd = chunked_rdd.filter(lambda x: len(x.strip()) > 0)
# print(chunked_rdd.take(5))

# create the vectorstore from the chunked RDD
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(
    texts=chunked_rdd.collect(), 
    embedding=embeddings,
    )

# Save FAISS index locally
local_dir = "/tmp/faiss_index"
os.makedirs(local_dir, exist_ok=True)
vectorstore.save_local(local_dir)

# Upload to GCS
client = storage.Client()
bucket_name = output_path.split("/")[2]           # gs://bucket_name/path
prefix = "/".join(output_path.split("/")[3:])     # path inside bucket
bucket = client.bucket(bucket_name)

for filename in os.listdir(local_dir):
    local_path = os.path.join(local_dir, filename)
    blob_path = f"{prefix}/faiss_index/{filename}"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {filename} to gs://{bucket_name}/{blob_path}")

sc.stop()
