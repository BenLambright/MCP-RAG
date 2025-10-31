FROM python:3.12-slim

WORKDIR /app
COPY ./Alices-Adventures-in-Wonderland-by-Lewis-Carroll.txt /app/
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["uvicorn", "query-mcp:app", "--host", "0.0.0.0", "--port", "8080"]