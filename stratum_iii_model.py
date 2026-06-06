"""
MaHealthBiasAudit - Stratum III: Model Bias Audit
Model-based analysis using embeddings and clustering
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from config import PRIMARY_LANGUAGES
from utils import setup_logger


class ModelBiasAuditor:
    """Model-based bias detection using embeddings"""
    
    def __init__(self):
        self.logger = setup_logger('model')
    
    def compute_language_clustering(self, 
                                   embeddings: np.ndarray,
                                   labels: List[str]) -> Dict[str, Any]:
        """Analyse how well embeddings cluster by language"""
        if len(embeddings) == 0:
            return {'error': 'No embeddings available'}
        
        # Dimensionality reduction for visualisation
        pca = PCA(n_components=50, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # t-SNE for 2D visualisation
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings_pca[:, :50])
        
        # Get unique languages
        unique_langs = list(set(labels))
        
        # Compute cluster purity by language
        # Try different cluster numbers
        best_k = None
        best_silhouette = -1
        
        for k in range(2, min(8, len(embeddings) // 10 + 2)):
            if k >= len(embeddings):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            if len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(embeddings, cluster_labels)
                if sil_score > best_silhouette:
                    best_silhouette = sil_score
                    best_k = k
        
        # Compute language separation
        language_centroids = {}
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            if lang_indices:
                language_centroids[lang] = np.mean(embeddings[lang_indices], axis=0)
        
        # Compute pairwise distances between language centroids
        lang_names = list(language_centroids.keys())
        centroid_distances = np.zeros((len(lang_names), len(lang_names)))
        
        for i, lang1 in enumerate(lang_names):
            for j, lang2 in enumerate(lang_names):
                if language_centroids[lang1] is not None and language_centroids[lang2] is not None:
                    centroid_distances[i, j] = np.linalg.norm(language_centroids[lang1] - language_centroids[lang2])
        
        # Compute intra-language vs inter-language distances
        intra_distances = []
        inter_distances = []
        
        for i, emb in enumerate(embeddings):
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                dist = np.linalg.norm(emb - emb2)
                if labels[i] == labels[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)
        
        separation_ratio = np.mean(inter_distances) / max(np.mean(intra_distances), 0.001) if intra_distances else 0
        
        results = {
            'embeddings_2d': embeddings_2d,
            'pca_explained_variance': pca.explained_variance_ratio_[:10].tolist(),
            'optimal_clusters': best_k,
            'best_silhouette_score': best_silhouette,
            'language_centroids': {k: v.tolist() if v is not None else None for k, v in language_centroids.items()},
            'centroid_distances': centroid_distances.tolist(),
            'centroid_distance_labels': lang_names,
            'separation_ratio': separation_ratio,
            'intra_language_variance': np.var(intra_distances) if intra_distances else 0,
            'inter_language_variance': np.var(inter_distances) if inter_distances else 0
        }
        
        return results
    
    def compute_translation_consistency(self, 
                                       embeddings: np.ndarray,
                                       labels: List[str],
                                       questions_by_lang: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute translation consistency scores between languages"""
        consistency_scores = {}
        
        if not embeddings.size:
            return {}
        
        # For each non-English language, compute similarity to English
        english_indices = [i for i, l in enumerate(labels) if l == 'English']
        
        for lang in PRIMARY_LANGUAGES:
            if lang == 'English' or lang not in questions_by_lang:
                continue
            
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            
            if not english_indices or not lang_indices:
                consistency_scores[lang] = 0
                continue
            
            # Compute pairwise similarities
            english_embs = embeddings[english_indices]
            lang_embs = embeddings[lang_indices]
            
            # Cosine similarity matrix
            sim_matrix = np.dot(lang_embs, english_embs.T)
            sim_matrix = np.clip(sim_matrix, -1, 1)
            
            # For each language embedding, find best match (max similarity)
            max_similarities = np.max(sim_matrix, axis=1)
            consistency_scores[lang] = np.mean(max_similarities)
        
        return consistency_scores
    
    def analyse_embedding_biases(self, 
                                embeddings: np.ndarray,
                                labels: List[str]) -> List[Dict]:
        """Identify potential biases in embedding space"""
        biases = []
        
        if embeddings.size == 0:
            return biases
        
        # Check for language-specific clusters
        unique_langs = list(set(labels))
        
        # Compute within-language and between-language distances
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            if len(lang_indices) < 2:
                continue
            
            lang_embeddings = embeddings[lang_indices]
            
            # Compute within-cluster variance
            centroid = np.mean(lang_embeddings, axis=0)
            within_var = np.mean([np.linalg.norm(emb - centroid) ** 2 for emb in lang_embeddings])
            
            # Compare to global variance
            global_centroid = np.mean(embeddings, axis=0)
            global_var = np.mean([np.linalg.norm(emb - global_centroid) ** 2 for emb in embeddings])
            
            variance_ratio = within_var / max(global_var, 0.001)
            
            if variance_ratio < 0.5:
                biases.append({
                    'Language': lang,
                    'Type': 'Overclustering',
                    'Severity': 'High' if variance_ratio < 0.3 else 'Moderate',
                    'Description': f"Language forms very tight cluster (variance ratio: {variance_ratio:.2f})",
                    'Recommendation': 'May indicate lack of semantic diversity or translation issues'
                })
            elif variance_ratio > 1.5:
                biases.append({
                    'Language': lang,
                    'Type': 'Dispersion',
                    'Severity': 'Moderate',
                    'Description': f"Language embeddings are widely dispersed (variance ratio: {variance_ratio:.2f})",
                    'Recommendation': 'May indicate inconsistency in responses'
                })
        
        # Check for English-centrism
        english_indices = [i for i, l in enumerate(labels) if l == 'English']
        if english_indices:
            english_centroid = np.mean(embeddings[english_indices], axis=0)
            
            for lang in unique_langs:
                if lang == 'English':
                    continue
                
                lang_indices = [i for i, l in enumerate(labels) if l == lang]
                if lang_indices:
                    lang_centroid = np.mean(embeddings[lang_indices], axis=0)
                    dist_to_english = np.linalg.norm(lang_centroid - english_centroid)
                    
                    # Check if this language is unusually close or far
                    if dist_to_english < 0.3:
                        biases.append({
                            'Language': lang,
                            'Type': 'English_Proximity',
                            'Severity': 'Low',
                            'Description': f"Language embeddings are very close to English (dist: {dist_to_english:.2f})",
                            'Recommendation': 'May indicate translations are too literal'
                        })
                    elif dist_to_english > 1.2:
                        biases.append({
                            'Language': lang,
                            'Type': 'English_Distance',
                            'Severity': 'High',
                            'Description': f"Language embeddings are far from English (dist: {dist_to_english:.2f})",
                            'Recommendation': 'May indicate cultural or semantic divergence'
                        })
        
        return biases
    
    def compute_semantic_similarity_matrix(self, 
                                          embeddings: np.ndarray,
                                          labels: List[str]) -> pd.DataFrame:
        """Compute average semantic similarity between languages"""
        unique_langs = list(set(labels))
        similarity_matrix = np.zeros((len(unique_langs), len(unique_langs)))
        
        for i, lang1 in enumerate(unique_langs):
            indices1 = [idx for idx, l in enumerate(labels) if l == lang1]
            if not indices1:
                continue
            
            emb1 = embeddings[indices1]
            
            for j, lang2 in enumerate(unique_langs):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    continue
                
                indices2 = [idx for idx, l in enumerate(labels) if l == lang2]
                if not indices2:
                    continue
                
                emb2 = embeddings[indices2]
                
                # Compute average pairwise cosine similarity
                sim_matrix = np.dot(emb1, emb2.T)
                sim_matrix = np.clip(sim_matrix, -1, 1)
                similarity_matrix[i, j] = np.mean(sim_matrix)
        
        return pd.DataFrame(similarity_matrix, index=unique_langs, columns=unique_langs)
    
    def run_full_audit(self,
                      questions_by_lang: Dict[str, List[str]],
                      answers_by_lang: Dict[str, List[str]]) -> Dict:
        """Run complete model bias audit"""
        self.logger.info("Starting model bias audit")
        
        # For this audit, we need embeddings from preprocessing
        # The embeddings will be passed in via the preprocessing results
        # For now, we'll return a structure indicating embeddings needed
        
        results = {
            'status': 'embeddings_required',
            'message': 'Embeddings must be computed in preprocessing stage',
            'summary': {
                'languages': list(questions_by_lang.keys()),
                'embeddings_available': False
            }
        }
        
        return results
    
    def run_audit_with_embeddings(self,
                                 embeddings: np.ndarray,
                                 labels: List[str],
                                 questions_by_lang: Dict[str, List[str]]) -> Dict:
        """Run model audit with pre-computed embeddings"""
        self.logger.info("Running model audit with embeddings")
        
        # Compute clustering analysis
        clustering_results = self.compute_language_clustering(embeddings, labels)
        
        # Compute translation consistency
        consistency_scores = self.compute_translation_consistency(embeddings, labels, questions_by_lang)
        
        # Compute semantic similarity matrix
        similarity_matrix = self.compute_semantic_similarity_matrix(embeddings, labels)
        
        # Analyse embedding biases
        bias_flags = self.analyse_embedding_biases(embeddings, labels)
        
        # Summary
        summary = {
            'embeddings_shape': embeddings.shape,
            'languages_analyzed': list(set(labels)),
            'optimal_clusters': clustering_results.get('optimal_clusters', 0),
            'separation_ratio': clustering_results.get('separation_ratio', 0),
            'translation_consistency': consistency_scores,
            'bias_flags_count': len(bias_flags)
        }
        
        results = {
            'clustering_analysis': clustering_results,
            'translation_consistency': consistency_scores,
            'semantic_similarity_matrix': similarity_matrix,
            'embedding_biases': bias_flags,
            'summary': summary
        }
        
        self.logger.info(f"Model audit complete: {summary}")
        return results