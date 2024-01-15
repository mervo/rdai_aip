docker build -t fastapi-pytorch .
docker run -p 8000:8000 fastapi-pytorch

http://localhost:8000/docs to test the /predict/ endpoint.
