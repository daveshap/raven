# Embeddings Microservice

A very simple service to embed input strings

This follows a simple app architecture
* routers contain all routes organized by need (in this case embeddings)
* schemas contain pydantic classes that define request and response structures
```
├── embeddings_pytorch
│   ├── __init__.py
│   ├── main.py
│   └── routers
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   └── schemas
│       ├── __init__.py
│       └── embeddings.py
```

See [here](https://fastapi.tiangolo.com/tutorial/bigger-applications/) for more information


## Development

Create a python environment
```
python -m venv .venv
source .venv/bin/activate
```
Install requirements
```
pip install -r requirements.txt
```
Run app
```
python main.py
```
Go to the [hosted docs](http://localhost:8088/docs) to try it
or POST
```
curl -X 'POST' \
  'http://localhost:8088/embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "strings": [
    "Hello",
    "你好",
    "Bonjour",
    "今日は",
    "नमस्ते",
    "Hallo",
    "Hola"
  ]
}'
```
