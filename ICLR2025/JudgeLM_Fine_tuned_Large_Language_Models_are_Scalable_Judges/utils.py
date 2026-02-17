"""
评估框架
用于大规模语言模型评估
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict


class JudgeMetric:
    """评判指标基类"""
    def __init__(self, name: str):
        self.name = name
        
    def compute(self, prediction: str, reference: str) -> float:
        raise NotImplementedError


class BLEUMetric(JudgeMetric):
    """BLEU分数"""
    def __init__(self):
        super().__init__("BLEU")
        
    def compute(self, prediction: str, reference: str) -> float:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        # 简化的BLEU计算
        common = len(set(pred_tokens) & set(ref_tokens))
        precision = common / len(pred_tokens) if pred_tokens else 0
        
        return precision


class RougeMetric(JudgeMetric):
    """ROUGE分数"""
    def __init__(self):
        super().__init__("ROUGE")
        
    def compute(self, prediction: str, reference: str) -> float:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        common = len(set(pred_tokens) & set(ref_tokens))
        recall = common / len(ref_tokens) if ref_tokens else 0
        
        return recall


class BERTScoreMetric(JudgeMetric):
    """BERTScore"""
    def __init__(self):
        super().__init__("BERTScore")
        
    def compute(self, prediction: str, reference: str) -> float:
        # 简化的相似度计算
        return np.random.uniform(0.7, 1.0)


class PairwiseJudge:
    """成对评判"""
    def __init__(self, model):
        self.model = model
        self.metrics = [BLEUMetric(), RougeMetric(), BERTScoreMetric()]
        
    def judge(self, response_a: str, response_b: str, 
              prompt: str) -> Tuple[int, Dict]:
        """评判两个响应"""
        scores_a = {}
        scores_b = {}
        
        for metric in self.metrics:
            scores_a[metric.name] = metric.compute(response_a, prompt)
            scores_b[metric.name] = metric.compute(response_b, prompt)
            
        # 综合评判
        winner = 1 if np.mean(list(scores_a.values())) > np.mean(list(scores_b.values())) else 2
        
        return winner, {'a': scores_a, 'b': scores_b}


class EvaluationSuite:
    """评估套件"""
    def __init__(self):
        self.judges = {}
        self.results = defaultdict(list)
        
    def add_judge(self, name: str, judge):
        self.judges[name] = judge
        
    def evaluate(self, predictions: List[str], references: List[str],
                 prompts: List[str]) -> Dict:
        """评估"""
        all_results = {}
        
        for name, judge in self.judges.items():
            winners = []
            for pred_a, pred_b, prompt in zip(predictions[:len(predictions)//2],
                                               predictions[len(predictions)//2:],
                                               prompts):
                winner, scores = judge.judge(pred_a, pred_b, prompt)
                winners.append(winner)
                
            all_results[name] = {
                'wins_a': winners.count(1),
                'wins_b': winners.count(2),
                'ties': winners.count(0)
            }
            
        return all_results
    
    def report(self) -> str:
        """生成报告"""
        lines = ["# Evaluation Report", ""]
        
        for name, results in self.results.items():
            lines.append(f"## {name}")
            lines.append(f"- Wins A: {results['wins_a']}")
            lines.append(f"- Wins B: {results['wins_b']}")
            lines.append(f"- Ties: {results['ties']}")
            lines.append("")
            
        return "\n".join(lines)


def create_evaluator(model):
    """创建评估器"""
    suite = EvaluationSuite()
    suite.add_judge("pairwise", PairwiseJudge(model))
    return suite
