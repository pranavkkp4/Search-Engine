# Micro Search Engine

This is a lightweight, self‑contained web search application designed to showcase basic
information retrieval and natural language processing skills.  It builds a simple
“semantic” search engine over a small dataset using TF‑IDF vectorization and
cosine similarity.  The project includes a minimal front‑end and back‑end
implemented with Python and Flask, making it easy to understand and extend.

## Features

* **Search over a custom dataset.**  A small dataset of topics and descriptions
  is provided out of the box (see `data/dataset.csv`).  You can replace this
  file with your own CSV of documents.  The search engine indexes the
  descriptions and ranks results according to TF‑IDF cosine similarity.
* **Interactive web interface.**  A simple HTML page lets you enter a query and
  browse the top search results.  Each result shows the title, description and
  a similarity score.
* **Minimal dependencies.**  Only a handful of common Python packages are used
  (`Flask`, `pandas`, `scikit‑learn`).  The project runs out of the box in
  environments that already have these libraries; otherwise they can be
  installed via `pip install -r requirements.txt`.
* **Clear structure.**  The code is organised in a single `app.py` entry
  point, with templates and static files separated.  This makes it easy to
  modify the front‑end or replace the search logic with more advanced models
  (e.g. sentence embeddings or vector databases) in the future.

## Getting Started

### Prerequisites

* Python 3.8 or later
* The following Python packages: `Flask`, `pandas`, `scikit‑learn`.  These
  dependencies are listed in `requirements.txt` and can be installed with

```bash
python -m pip install -r requirements.txt
```

### Running the Application

1. Clone or unpack this repository on your local machine.
2. Install the required dependencies as described above.
3. From within the project directory, run:

```bash
python app.py
```

4. Open your browser and navigate to <http://localhost:5000>.  You should see
   the Micro Search Engine home page.  Enter a query and click “Search” to see
   the ranked results.

### Replacing the Dataset

The search engine reads its data from `data/dataset.csv`.  Each row in the
CSV should have three columns:

| column       | description                              |
|--------------|------------------------------------------|
| `id`         | A unique identifier for the document     |
| `title`      | A short title for the document           |
| `description`| The text content to be indexed and search|

If you wish to search over your own documents, replace `dataset.csv` with
your own file following the same structure.  Restart the server and the
documents will be reindexed automatically on startup.

### Extending the Search Engine

This project is deliberately simple, but it is a great starting point for
experimenting with more sophisticated information retrieval techniques.  Here
are a few ideas:

* Replace the TF‑IDF vectorizer with a sentence embedding model (e.g. from
  the `sentence‑transformers` library) to capture semantic similarity rather
  than relying on exact word overlap.  You will need to download a model and
  compute embeddings for the dataset and queries.
* Store vectors in a specialised vector database such as Faiss or Annoy to
  scale to millions of documents.
* Add filtering, sorting or pagination to the result list.  For example, you
  could display only results whose similarity score exceeds a threshold.
* Provide a RESTful API endpoint that returns search results in JSON instead
  of rendering HTML.  This would make it easy to integrate the engine into
  other applications.

## Project Structure

```
micro_search_engine/
├── app.py               # Flask application with search logic
├── data/
│   └── dataset.csv      # Sample dataset
├── requirements.txt     # List of dependencies
├── templates/
│   └── index.html       # Front‑end template
├── static/
│   └── style.css        # Optional CSS styling
└── README.md            # This document
```

## License

This project is licensed under the MIT License.  See the `LICENSE` file
provided with this repository for details.
