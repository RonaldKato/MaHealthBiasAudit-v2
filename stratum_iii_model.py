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
        
        self.logger.info(f"Computing clustering on {len(embeddings)} embeddings")
        
        # PCA for dimensionality reduction
        pca = PCA(n_components=min(50, len(embeddings)), random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)
        
        # t-SNE for 2D visualization
        perplexity = min(30, max(5, len(embeddings) - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_pca[:, :min(50, embeddings_pca.shape[1])])
        
        unique_langs = list(set(labels))
        
        # Find optimal number of clusters
        best_k = None
        best_silhouette = -1
        
        max_k = min(8, len(embeddings) // 5 + 2)
        for k in range(2, max_k):
            if k >= len(embeddings):
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            if len(set(cluster_labels)) > 1:
                try:
                    sil_score = silhouette_score(embeddings, cluster_labels)
                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_k = k
                except:
                    pass
        
        # Compute language centroids
        language_centroids = {}
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            if lang_indices:
                language_centroids[lang] = np.mean(embeddings[lang_indices], axis=0)
        
        # Compute centroid distances
        lang_names = list(language_centroids.keys())
        centroid_distances = np.zeros((len(lang_names), len(lang_names)))
        
        for i, lang1 in enumerate(lang_names):
            for j, lang2 in enumerate(lang_names):
                if language_centroids[lang1] is not None and language_centroids[lang2] is not None:
                    centroid_distances[i, j] = np.linalg.norm(language_centroids[lang1] - language_centroids[lang2])
        
        # Compute intra/inter distances
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
            'embeddings_2d': embeddings_2d.tolist(),
            'pca_explained_variance': pca.explained_variance_ratio_[:10].tolist(),
            'optimal_clusters': best_k,
            'best_silhouette_score': best_silhouette,
            'language_centroids': {k: v.tolist() if v is not None else None for k, v in language_centroids.items()},
            'centroid_distances': centroid_distances.tolist(),
            'centroid_distance_labels': lang_names,
            'separation_ratio': separation_ratio,
            'intra_language_variance': float(np.var(intra_distances)) if intra_distances else 0,
            'inter_language_variance': float(np.var(inter_distances)) if inter_distances else 0,
            'num_samples': len(embeddings)
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
        
        english_indices = [i for i, l in enumerate(labels) if l == 'English']
        
        for lang in PRIMARY_LANGUAGES:
            if lang == 'English' or lang not in questions_by_lang:
                continue
            
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            
            if not english_indices or not lang_indices:
                consistency_scores[lang] = 0
                continue
            
            english_embs = embeddings[english_indices]
            lang_embs = embeddings[lang_indices]
            
            sim_matrix = np.dot(lang_embs, english_embs.T)
            sim_matrix = np.clip(sim_matrix, -1, 1)
            
            max_similarities = np.max(sim_matrix, axis=1)
            consistency_scores[lang] = float(np.mean(max_similarities))
            
            # Also compute standard deviation
            consistency_scores[f"{lang}_std"] = float(np.std(max_similarities))
        
        return consistency_scores
    
    def analyse_embedding_biases(self, 
                                embeddings: np.ndarray,
                                labels: List[str]) -> List[Dict]:
        """Identify potential biases in embedding space"""
        biases = []
        
        if embeddings.size == 0:
            return biases
        
        unique_langs = list(set(labels))
        
        for lang in unique_langs:
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            if len(lang_indices) < 2:
                continue
            
            lang_embeddings = embeddings[lang_indices]
            centroid = np.mean(lang_embeddings, axis=0)
            within_var = np.mean([np.linalg.norm(emb - centroid) ** 2 for emb in lang_embeddings])
            
            global_centroid = np.mean(embeddings, axis=0)
            global_var = np.mean([np.linalg.norm(emb - global_centroid) ** 2 for emb in embeddings])
            
            variance_ratio = within_var / max(global_var, 0.001)
            
            if variance_ratio < 0.3:
                biases.append({
                    'Language': lang,
                    'Type': 'Overclustering_Critical',
                    'Severity': 'Critical',
                    'Description': f"Language forms extremely tight cluster (variance ratio: {variance_ratio:.2f})",
                    'Recommendation': 'URGENT: May indicate severe lack of semantic diversity or translation issues'
                })
            elif variance_ratio < 0.5:
                biases.append({
                    'Language': lang,
                    'Type': 'Overclustering',
                    'Severity': 'High',
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
        
        # English distance analysis
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
                    
                    if dist_to_english < 0.2:
                        biases.append({
                            'Language': lang,
                            'Type': 'English_Proximity_Critical',
                            'Severity': 'Critical',
                            'Description': f"Language embeddings are extremely close to English (dist: {dist_to_english:.2f})",
                            'Recommendation': 'URGENT: May indicate translations are too literal or copied'
                        })
                    elif dist_to_english < 0.3:
                        biases.append({
                            'Language': lang,
                            'Type': 'English_Proximity',
                            'Severity': 'Moderate',
                            'Description': f"Language embeddings are very close to English (dist: {dist_to_english:.2f})",
                            'Recommendation': 'May indicate translations are too literal'
                        })
                    elif dist_to_english > 1.2:
                        biases.append({
                            'Language': lang,
                            'Type': 'English_Distance_Critical',
                            'Severity': 'Critical',
                            'Description': f"Language embeddings are very far from English (dist: {dist_to_english:.2f})",
                            'Recommendation': 'URGENT: May indicate severe cultural or semantic divergence'
                        })
                    elif dist_to_english > 0.9:
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
                
                sim_matrix = np.dot(emb1, emb2.T)
                sim_matrix = np.clip(sim_matrix, -1, 1)
                similarity_matrix[i, j] = float(np.mean(sim_matrix))
        
        return pd.DataFrame(similarity_matrix, index=unique_langs, columns=unique_langs)
    
    def run_audit_with_embeddings(self,
                                 embeddings: np.ndarray,
                                 labels: List[str],
                                 questions_by_lang: Dict[str, List[str]]) -> Dict:
        """Run model audit with pre-computed embeddings"""
        self.logger.info("="*50)
        self.logger.info("STARTING MODEL AUDIT")
        self.logger.info("="*50)
        
        self.logger.info("Computing language clustering...")
        clustering_results = self.compute_language_clustering(embeddings, labels)
        
        self.logger.info("Computing translation consistency...")
        consistency_scores = self.compute_translation_consistency(embeddings, labels, questions_by_lang)
        
        self.logger.info("Computing semantic similarity matrix...")
        similarity_matrix = self.compute_semantic_similarity_matrix(embeddings, labels)
        
        self.logger.info("Analysing embedding biases...")
        bias_flags = self.analyse_embedding_biases(embeddings, labels)
        
        summary = {
            'embeddings_shape': list(embeddings.shape),
            'languages_analyzed': list(set(labels)),
            'optimal_clusters': clustering_results.get('optimal_clusters', 0),
            'separation_ratio': clustering_results.get('separation_ratio', 0),
            'translation_consistency': consistency_scores,
            'bias_flags_count': len(bias_flags),
            'critical_flags': sum(1 for f in bias_flags if f.get('Severity') == 'Critical')
        }
        
        results = {
            'clustering_analysis': clustering_results,
            'translation_consistency': consistency_scores,
            'semantic_similarity_matrix': similarity_matrix,
            'embedding_biases': bias_flags,
            'summary': summary
        }
        
        self.logger.info("="*50)
        self.logger.info("MODEL AUDIT COMPLETE")
        self.logger.info(f"  Languages analyzed: {summary['languages_analyzed']}")
        self.logger.info(f"  Optimal clusters: {summary['optimal_clusters']}")
        self.logger.info(f"  Bias flags: {summary['bias_flags_count']}")
        self.logger.info("="*50)
        
        return results