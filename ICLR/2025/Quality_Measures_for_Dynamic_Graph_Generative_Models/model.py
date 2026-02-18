"""
Quality Measures for Dynamic Graph Generation
动态图生成质量度量
"""

import torch
import torch.nn as nn

class GraphQualityMetric(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 度分布相似度
        self.degree_scorer = nn.Linear(128, 1)
        
        # 聚类系数
        self.cluster_scorer = nn.Linear(128, 1)
        
        # 最短路径
        self.path_scorer = nn.Linear(128, 1)
        
    def compute_degree_dist(self, adj):
        """计算度分布"""
        return adj.sum(dim=-1)
    
    def compute_clustering(self, adj):
        """计算聚类系数"""
        # 简化的聚类计算
        tri = (adj @ adj @ adj).diagonal()
        deg = adj.sum(dim=-1)
        deg[deg < 2] = 1
        return tri / (deg * (deg - 1))
    
    def forward(self, graph1, graph2):
        """计算两个图的质量相似度"""
        d1 = self.compute_degree_dist(graph1)
        d2 = self.compute_degree_dist(graph2)
        
        degree_sim = torch.exp(-torch.abs(d1 - d2).mean())
        
        c1 = self.compute_clustering(graph1)
        c2 = self.compute_clustering(graph2)
        
        cluster_sim = torch.exp(-torch.abs(c1 - c2).mean())
        
        return {
            'degree_similarity': degree_sim,
            'clustering_similarity': cluster_sim,
            'overall': (degree_sim + cluster_sim) / 2
        }
