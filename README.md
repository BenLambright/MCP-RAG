# Big Data Analytics — Term Paper

Benjamin Lambright & Pearlrose Nwuke

## Goal
A comparison of running a simple RAG-style information retrieval how we've been doing in class on PySpark versus running it on a Docker image to simulate a simple MCP server.

---

## Repository layout (quick overview)
- README.md
- Dockerfile — Where I set up the docker image for the cloud
- faiss_index/ — Before I switched to just using TF-IDF, I calculated FAISS indices using an off-the-shelf model from hugging-face. I couldn't get that to work on the cloud, though.
- ingest.py — ingest file when I wasy using FAISS, didn't work on the cloud
- client.py - client script to query the embedding space that had been run on the docker image
- cloudbuild.yaml - build yaml file so that I could run docker image in the cloud terminal
- query-cntrl.py - the control, a simpler version of assignment4
- query-mcp.py - the app for embedding documents on a container
- requirements.txt

---

## Important notes
The following files are depreciated (not used for the final draft), but I kept them to show you guys my progress:
`ingest.py`, `install-deps.sh`, anything with "faiss" in the name

The *main* scripts are `query-cntrl.py` and `query-mcp.py`. All other scripts are supporting scripts. See relevant sections for more details.

---

## Development & environment (short)
- Python 3.8+ recommended.
- Install dependencies: pip install -r requirements.txt
- Use Docker for reproducible runtime (see docker/Dockerfile).

---

## query-cntrl.py (placeholder)
Purpose:
- Similar to assignment 4, used TF-IDF and PySpark to parse and chunk a document and get the embeddings. Then, I queried those embeddings using cosign similarity. 

What to document here:
- You should be able to just run this on a cluster with at least 4 nodes with our dummy txt file
- Inputs: our dummy txt file
- Outputs: a path for the output result 
- Dependencies: Pyspark

Ideally I would have continued using a model from Huggingface to do state of the art embeddings, and then query an LLM, but several hours of trying, it seems like we lack the permissions to access dependencies on GCP that require internet (so in this case downloading the weights of models like an LLM for usage on GCP)

---

## docker/Dockerfile
Purpose:
- Container image used to run components reproducibly and simulate a basic MCP server, see reproduce MCP for more details.

---

## query-mcp.py (placeholder)
Purpose:
- Created a basic app that effectively does the same thing as the control, only the retrieval side is wrapped in an API call, so we can continuously make new calls. In theory, you could add more commands to this to save the results to use them for other things, or to just save them in a database.

Notes:
I had trouble using PySpark on a Docker image in the cloud, potentially because of compute issues but I'm not 100% sure. Learning how to build a docker image on the GCP's Dataproc was incredibly challenging, so I had to make this as simplistic as possible. That's the reason why this version doesn't even use PySpark, but it's also good for comparing what in theory might be a fast PySpark application versus a potentially slower containerized server that allows you to do a lot more with your requests.

---

## Reproduce MCP
Below summarizes the steps taken to deploy the FastAPI RAG server onto the cluster using Google Cloud Build and Docker.

### Phase 1: Preparation and Image Build (via GCP Cloud Shell)

The initial environment setup and image creation were handled from the Google Cloud Shell terminal. You have to pull this up on the website.

### File Upload
All required files were uploaded to the Cloud Shell working directory. There is an option to upload files to the terminal in GCP:

- `Dockerfile`
- `query-mcp.py`
- `requirements.txt`
- `cloudbuild.yaml`
- `Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt`

After you've added all of those, you can execute the following command:
```bash
gcloud builds submit --config cloudbuild.yaml --project assignment-4-475110
```

From this, you should be able to find the image as a repository in the Artifact Registery for your project. Click on the path and copy the path.

### Phase 2: Deployment and Execution (via SSH to Master Node)

After the image was built, start up a cluster and SSH into the master node for it.

### SSH Access
The cluster's master node was accessed via SSH.

### Stopping Stale Containers
Before running the new image, the previously failed container was forcefully stopped and removed (if you have one):

```bash
sudo docker rm -f rag-server
```

### Running the New Image
The container was launched in detached mode, explicitly using the latest image tag and binding the container's port 8080 to the master node's port 8080:

```bash
sudo docker run \
    --detach \
    --publish 8080:8080 \
    --name rag-server \
    <IMAGE_URI>:<TAG>
```

### Phase 3: Testing and Querying the Live Service

Once the container was running (`docker ps` showed the status as Up), bring in our client script to run and actually do the queries.

### Client Script Creation
A simple Python client script (`client.py`) was created using `nano`:

```bash
nano client.py
```

You literally just copy and paste the client script from github and put it in here to run. There's probaby a better way to do this but this is what I found works.

### Execution
Finally, you just run the client script!

```bash
python client.py
# URL: http://localhost:8080/rag
```

Notes
- Written by Benjamin Lambright
