import os
import json
import argparse
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle

class FAISSIndexer:
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        dimension: Optional[int] = None
    ):
        """
        Initialize the FAISS indexer.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            index_type: Type of FAISS index ('flat', 'ivf', 'pq')
            dimension: Dimension of the embeddings (if None, determined from model)
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Determine embedding dimension if not provided
        if dimension is None:
            dimension = self.model.get_sentence_embedding_dimension()
        
        self.dimension = dimension
        print(f"Embedding dimension: {self.dimension}")
        
        # Create the appropriate index based on type
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "ivf":
            # IVF index with flat quantizer (faster search, slight accuracy loss)
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = 100  # number of clusters
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train_needs_objects = True  # Flag indicating training is needed
        elif index_type == "pq":
            # Product Quantization (more compression, more accuracy loss)
            m = 8  # number of subquantizers
            self.index = faiss.IndexPQ(self.dimension, m, 8)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.document_lookup = []  # Stores document texts or metadata
        self.trained = False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None, batch_size: int = 32):
        """
        Add texts to the index.
        
        Args:
            texts: List of text strings to index
            metadatas: Optional list of metadata dictionaries corresponding to each text
            batch_size: Batch size for generating embeddings
        """
        if not texts:
            print("No texts provided")
            return
        
        if metadatas and len(texts) != len(metadatas):
            raise ValueError("Number of texts and metadata entries must match")
        
        # Create lookup entries
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            self.document_lookup.append({
                "text": text,
                "metadata": metadata
            })
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings).astype('float32')
        
        # Train the index if needed
        if hasattr(self.index, 'train_needs_objects') and not self.trained:
            print("Training the index...")
            self.index.train(embeddings)
            self.trained = True
        
        # Add vectors to the index
        print(f"Adding {len(embeddings)} vectors to the index")
        self.index.add(embeddings)
        print(f"Index now contains {self.index.ntotal} vectors")
    
    def save_index(self, directory: str):
        """
        Save the FAISS index and document lookup to disk.
        
        Args:
            directory: Directory where the index will be saved
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save document lookup
        lookup_path = os.path.join(directory, "document_lookup.pkl")
        with open(lookup_path, 'wb') as f:
            pickle.dump(self.document_lookup, f)
        
        # Save model name and dimension
        config_path = os.path.join(directory, "config.json")
        config = {
            "model_name": self.model.model_card_data["base_model"],
            "dimension": self.dimension,
            "document_count": len(self.document_lookup)
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Index saved to {directory}")
    
    @classmethod
    def load_index(cls, directory: str, model_name: str = None):
        """
        Load a saved FAISS index from disk.
        
        Args:
            directory: Directory where the index is saved
            model_name: Optional model name to override the saved config
            
        Returns:
            FAISSIndexer instance with loaded index
        """
        # Load configuration
        config_path = os.path.join(directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Use provided model name or load from config
        if model_name is None:
            model_name = config.get("model_name")
        
        # Create indexer instance
        indexer = cls(model_name=model_name, dimension=config["dimension"])
        
        # Load FAISS index
        index_path = os.path.join(directory, "faiss_index.bin")
        indexer.index = faiss.read_index(index_path)
        
        # Load document lookup
        lookup_path = os.path.join(directory, "document_lookup.pkl")
        with open(lookup_path, 'rb') as f:
            indexer.document_lookup = pickle.load(f)
        
        print(f"Loaded index with {indexer.index.ntotal} vectors and {len(indexer.document_lookup)} documents")
        return indexer
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search the index for the most similar documents to the query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing the retrieved documents and scores
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.document_lookup):
                continue  # Skip invalid indices
            
            doc = self.document_lookup[idx]
            results.append({
                "id": idx,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(1.0 / (1.0 + dist))  # Convert distance to similarity score
            })
        
        return results

def concatenate_paper_fields(paper: Dict) -> str:
    """
    Concatenate title, abstract, and results from a paper dictionary.
    
    Args:
        paper: Dictionary containing paper data
        
    Returns:
        Concatenated text
    """
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    results = paper.get("results", "")
    
    # Concatenate with separator sections
    concatenated = ""
    if title:
        concatenated += f"TITLE: {title}\n\n"
    if abstract:
        concatenated += f"ABSTRACT: {abstract}\n\n"
    if results:
        concatenated += f"RESULTS: {results}\n\n"
    
    return concatenated.strip()

def index_medical_papers(
    source_dir: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_type: str = "flat",
    batch_size: int = 32
):
    """
    Index medical papers from all JSON files in a directory and save the FAISS index.
    
    Args:
        source_dir: Path to directory containing JSON files with medical papers
        output_dir: Directory where the index will be saved
        model_name: Name of the SentenceTransformer model to use
        index_type: Type of FAISS index ('flat', 'ivf', 'pq')
        batch_size: Batch size for generating embeddings
    """
    # Create indexer
    indexer = FAISSIndexer(model_name=model_name, index_type=index_type)
    
    # Get all JSON files in the directory
    json_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        raise ValueError(f"No JSON files found in directory: {source_dir}")
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file and collect papers
    all_papers = []
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            
            # Handle different JSON structures
            if not isinstance(papers, list):
                # If it's not a list, try to see if it's a dictionary with a list field
                if isinstance(papers, dict) and any(isinstance(papers.get(key), list) for key in papers):
                    # Find the first list field
                    for key in papers:
                        if isinstance(papers[key], list):
                            papers = papers[key]
                            break
                else:
                    # If it's a single paper, wrap it in a list
                    papers = [papers]
            
            # Add file path as source metadata
            for paper in papers:
                if isinstance(paper, dict):
                    paper['_source_file'] = json_file
            
            all_papers.extend(papers)
            print(f"Loaded {len(papers)} papers from {json_file}")
        
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print(f"Loaded a total of {len(all_papers)} papers from all files")
    
    # Concatenate fields and prepare for indexing
    texts = []
    metadatas = []
    
    for paper in all_papers:
        # Create the text to embed by concatenating fields
        text = concatenate_paper_fields(paper)
        if not text:
            print(f"Warning: Empty text for paper: {paper.get('title', 'Unknown title')}")
            continue
        
        texts.append(text)
        metadatas.append(paper)  # Use the entire paper as metadata
    
    print(f"Prepared {len(texts)} papers for indexing")
    
    # Add texts to the index
    indexer.add_texts(texts, metadatas, batch_size=batch_size)
    
    # Save the index
    indexer.save_index(output_dir)

if __name__ == "__main__":
    SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
    SOURCE_DIR  = os.path.join(SCRIPT_DIR, "source_data")
    OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "faiss_index")

    
    index_medical_papers(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR
    )