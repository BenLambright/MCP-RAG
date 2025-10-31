from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import FloatType
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys

spark = SparkSession.builder.appName("NaiveRAG").getOrCreate()
sc = spark.sparkContext

file_path = sys.argv[1]
output_path = sys.argv[2]
# file_path = "Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt"
rdd = sc.textFile(file_path)

# first thing we need to do is chunk the text so we have narrower chunks of embeddings to search from
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Apply chunking
chunked_rdd = rdd.flatMap(lambda line: split_text(line))
chunked_rdd = chunked_rdd.filter(lambda x: len(x.strip()) > 0)  # remove empty chunks

# create a dataframe of our chunks with doc_id's
df = chunked_rdd.zipWithIndex().map(lambda x: (x[1], x[0])).toDF(["doc_id", "chunk_text"])

# compute embeddings using TF-IDF
chunks_list = [row.chunk_text for row in df.collect()]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunks_list)

def to_list(row_idx):
    return tfidf_matrix[row_idx].toarray().flatten().tolist()

df_with_embeds = df.rdd.map(lambda row: (row['doc_id'], row['chunk_text'], to_list(row['doc_id']))).toDF(
    ["doc_id", "chunk_text", "embedding"]
)

# create the query and embed it
query = "Alice falls down a rabbit hole"
query_embedding = vectorizer.transform([query]).toarray().flatten()
query_broadcast = sc.broadcast(query_embedding)

# define our cosine similarity function to compare the query to each chunk
def cosine_similarity_udf(chunk_emb):
    chunk_emb = np.array(chunk_emb)
    q_emb = query_broadcast.value
    norm_chunk = np.linalg.norm(chunk_emb)
    norm_q = np.linalg.norm(q_emb)
    if norm_chunk == 0 or norm_q == 0:
        return 0.0
    return float(np.dot(chunk_emb, q_emb) / (norm_chunk * norm_q))

cosine_udf = F.udf(cosine_similarity_udf, FloatType())

df_with_scores = df_with_embeds.withColumn("score", cosine_udf("embedding"))

# retrieve the closest chunks
top_k = df_with_scores.orderBy(F.desc("score")).limit(3)

# Print top-k chunks
for row in top_k.collect():
    print(f"doc_id: {row['doc_id']}, score: {row['score']}")
    print(f"chunk: {row['chunk_text'][:200]}...\n")  # print first 200 chars

# save the top results
top_rdd = top_k.rdd.map(lambda row: f"doc_id: {row['doc_id']}, score: {row['score']}\n{row['chunk_text']}")
top_rdd.coalesce(1).saveAsTextFile(output_path)

spark.stop()
