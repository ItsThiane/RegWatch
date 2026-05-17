FROM python:3.11-slim

# Variables environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Dossier travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Installer dépendances Python
RUN pip install -r requirements.txt

# Pré-download du modèle embedding
# (évite téléchargement au runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-large')"

# Copier projet
COPY . .

# Streamlit config
EXPOSE 8501

# Commande lancement
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]