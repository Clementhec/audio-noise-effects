"""
Embedding utilities using HuggingFace Sentence Transformers

This module provides semantic embedding generation using local models from HuggingFace.
No API keys required - runs entirely locally.

Model: sentence-transformers/all-MiniLM-L6-v2
- 384 dimensions
- Fast and efficient
- Good for semantic similarity tasks
- Multilingual support
"""

import textwrap as tr
from typing import List, Optional, Union
import warnings

import matplotlib.pyplot as plt
import plotly.express as px
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Global model instance (lazy loaded)
_model_cache = {}


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or load the sentence transformer model.

    Args:
        model_name: Name of the model from sentence-transformers
                   Default: "all-MiniLM-L6-v2" (384-dim, fast, good quality)

    Returns:
        SentenceTransformer model instance
    """
    if model_name not in _model_cache:
        print(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {_model_cache[model_name].get_sentence_embedding_dimension()}")

    return _model_cache[model_name]


def get_embedding(
    text: str,
    model: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    **kwargs
) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Input text to embed
        model: Model name (default: "all-MiniLM-L6-v2")
        normalize: Whether to normalize embeddings (recommended for cosine similarity)
        **kwargs: Additional arguments passed to model.encode()

    Returns:
        List of floats representing the embedding vector
    """
    # Replace newlines, which can negatively affect performance
    text = text.replace("\n", " ")

    # Get the model
    encoder = get_model(model)

    # Generate embedding
    embedding = encoder.encode(
        text,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        **kwargs
    )

    return embedding.tolist()


def get_embeddings(
    list_of_text: List[str],
    model: str = "all-MiniLM-L6-v2",
    normalize: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
    **kwargs
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts (batch processing).

    Args:
        list_of_text: List of texts to embed
        model: Model name (default: "all-MiniLM-L6-v2")
        normalize: Whether to normalize embeddings (recommended for cosine similarity)
        batch_size: Number of texts to process at once
        show_progress: Show progress bar
        **kwargs: Additional arguments passed to model.encode()

    Returns:
        List of embedding vectors (each as List[float])
    """
    if len(list_of_text) == 0:
        return []

    # Replace newlines
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    # Get the model
    encoder = get_model(model)

    # Generate embeddings in batch
    embeddings = encoder.encode(
        list_of_text,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        **kwargs
    )

    # Convert to list of lists
    return [emb.tolist() for emb in embeddings]


def cosine_similarity(a: Union[List[float], np.ndarray], b: Union[List[float], np.ndarray]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score (0 to 1 for normalized vectors)
    """
    a = np.array(a) if not isinstance(a, np.ndarray) else a
    b = np.array(b) if not isinstance(b, np.ndarray) else b

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def distances_from_embeddings(
    query_embedding: Union[List[float], np.ndarray],
    embeddings: List[Union[List[float], np.ndarray]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculate distances between a query embedding and a list of embeddings.

    Args:
        query_embedding: Query vector
        embeddings: List of vectors to compare against
        distance_metric: One of "cosine", "L1", "L2", "Linf"

    Returns:
        List of distances
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(f"Unknown distance metric: {distance_metric}. Choose from {list(distance_metrics.keys())}")

    # Convert to numpy arrays if needed
    query_embedding = np.array(query_embedding) if not isinstance(query_embedding, np.ndarray) else query_embedding
    embeddings = [np.array(emb) if not isinstance(emb, np.ndarray) else emb for emb in embeddings]

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Return indices of nearest neighbors from a list of distances.

    Args:
        distances: List of distance values

    Returns:
        Numpy array of indices sorted by distance (nearest first)
    """
    return np.argsort(distances)


def pca_components_from_embeddings(
    embeddings: List[List[float]], n_components: int = 2
) -> np.ndarray:
    """
    Reduce embedding dimensionality using PCA.

    Args:
        embeddings: List of embedding vectors
        n_components: Number of dimensions to reduce to

    Returns:
        Numpy array of reduced embeddings
    """
    pca = PCA(n_components=n_components)
    array_of_embeddings = np.array(embeddings)
    return pca.fit_transform(array_of_embeddings)


def tsne_components_from_embeddings(
    embeddings: List[List[float]], n_components: int = 2, **kwargs
) -> np.ndarray:
    """
    Reduce embedding dimensionality using t-SNE.

    Args:
        embeddings: List of embedding vectors
        n_components: Number of dimensions to reduce to
        **kwargs: Additional arguments for TSNE

    Returns:
        Numpy array of reduced embeddings
    """
    # Use better defaults if not specified
    if "init" not in kwargs.keys():
        kwargs["init"] = "pca"
    if "learning_rate" not in kwargs.keys():
        kwargs["learning_rate"] = "auto"

    tsne = TSNE(n_components=n_components, **kwargs)
    array_of_embeddings = np.array(embeddings)
    return tsne.fit_transform(array_of_embeddings)


def chart_from_components(
    components: np.ndarray,
    labels: Optional[List[str]] = None,
    strings: Optional[List[str]] = None,
    x_title: str = "Component 0",
    y_title: str = "Component 1",
    mark_size: int = 5,
    **kwargs,
):
    """
    Create interactive 2D chart of embedding components.

    Args:
        components: 2D array of components
        labels: Optional labels for coloring
        strings: Optional hover text
        x_title: X-axis title
        y_title: Y-axis title
        mark_size: Size of markers

    Returns:
        Plotly figure
    """
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter(
        data,
        x=x_title,
        y=y_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


def chart_from_components_3D(
    components: np.ndarray,
    labels: Optional[List[str]] = None,
    strings: Optional[List[str]] = None,
    x_title: str = "Component 0",
    y_title: str = "Component 1",
    z_title: str = "Component 2",
    mark_size: int = 5,
    **kwargs,
):
    """
    Create interactive 3D chart of embedding components.

    Args:
        components: 3D array of components
        labels: Optional labels for coloring
        strings: Optional hover text
        x_title: X-axis title
        y_title: Y-axis title
        z_title: Z-axis title
        mark_size: Size of markers

    Returns:
        Plotly figure
    """
    empty_list = ["" for _ in components]
    data = pd.DataFrame(
        {
            x_title: components[:, 0],
            y_title: components[:, 1],
            z_title: components[:, 2],
            "label": labels if labels else empty_list,
            "string": ["<br>".join(tr.wrap(string, width=30)) for string in strings]
            if strings
            else empty_list,
        }
    )
    chart = px.scatter_3d(
        data,
        x=x_title,
        y=y_title,
        z=z_title,
        color="label" if labels else None,
        symbol="label" if labels else None,
        hover_data=["string"] if strings else None,
        **kwargs,
    ).update_traces(marker=dict(size=mark_size))
    return chart


def plot_multiclass_precision_recall(
    y_score, y_true_untransformed, class_list, classifier_name
):
    """
    Precision-Recall plotting for a multiclass problem.

    Plots average precision-recall, per class precision recall and reference f1 contours.
    Code slightly modified, but heavily based on https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    """
    n_classes = len(class_list)
    y_true = pd.concat(
        [(y_true_untransformed == class_list[i]) for i in range(n_classes)], axis=1
    ).values

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true.ravel(), y_score.ravel()
    )
    average_precision_micro = average_precision_score(y_true, y_score, average="micro")
    print(
        str(classifier_name)
        + " - Average precision score over all classes: {0:0.2f}".format(
            average_precision_micro
        )
    )

    # setup plot details
    plt.figure(figsize=(9, 10))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append("iso-f1 curves")
    (l,) = plt.plot(recall_micro, precision_micro, color="gold", lw=2)
    lines.append(l)
    labels.append(
        "average Precision-recall (auprc = {0:0.2f})" "".format(average_precision_micro)
    )

    for i in range(n_classes):
        (l,) = plt.plot(recall[i], precision[i], lw=2)
        lines.append(l)
        labels.append(
            "Precision-recall for class `{0}` (auprc = {1:0.2f})"
            "".format(class_list[i], average_precision[i])
        )

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{classifier_name}: Precision-Recall curve for each class")
    plt.legend(lines, labels)


# ============================================================================
# Model Information
# ============================================================================

def get_embedding_dimension(model: str = "all-MiniLM-L6-v2") -> int:
    """
    Get the embedding dimension for a given model.

    Args:
        model: Model name

    Returns:
        Embedding dimension
    """
    encoder = get_model(model)
    return encoder.get_sentence_embedding_dimension()


# Available models (can be extended)
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "description": "Fast, good quality, recommended default",
        "languages": "50+ languages"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "description": "Higher quality, slower",
        "languages": "English"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "description": "Multilingual, good for non-English",
        "languages": "50+ languages"
    }
}


if __name__ == "__main__":
    """Quick test of embedding functionality."""
    print("Testing HuggingFace Embeddings")
    print("=" * 60)

    # Test single embedding
    text = "The dog is barking loudly"
    emb = get_embedding(text)
    print(f"Text: '{text}'")
    print(f"Embedding dimension: {len(emb)}")
    print(f"First 5 values: {emb[:5]}")
    print()

    # Test batch embeddings
    texts = [
        "The dog is barking loudly",
        "A canine is making noise",
        "Thunder echoes across the valley",
        "Lightning strikes nearby"
    ]
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = get_embeddings(texts, show_progress=True)
    print(f"Generated {len(embeddings)} embeddings")
    print()

    # Test similarity
    print("Similarity scores:")
    print(f"  Dog barking vs Canine noise: {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
    print(f"  Dog barking vs Thunder: {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
    print(f"  Thunder vs Lightning: {cosine_similarity(embeddings[2], embeddings[3]):.3f}")
