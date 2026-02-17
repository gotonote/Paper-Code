"""
SciArena: Scientific Literature Evaluation Platform
科学文献评估平台
"""

import torch
import torch.nn as nn

class SciArenaEvaluator:
    def __init__(self, model_name="scibert"):
        self.model_name = model_name
        self.metrics = {
            'accuracy': [],
            'f1': [],
            'citation_recall': []
        }
        
    def evaluate(self, predictions, references, task_type="qa"):
        """评估"""
        if task_type == "qa":
            return self.evaluate_qa(predictions, references)
        elif task_type == "citation":
            return self.evaluate_citation(predictions, references)
        else:
            return {}
    
    def evaluate_qa(self, preds, refs):
        """问答评估"""
        correct = sum(p == r for p, r in zip(preds, refs))
        accuracy = correct / len(preds)
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(preds)
        }
    
    def evaluate_citation(self, preds, refs):
        """引用预测评估"""
        # 召回率
        true_pos = sum(any(p in r for p in pred) for pred, ref in zip(preds, refs))
        recall = true_pos / len(refs)
        
        return {
            'citation_recall': recall,
            'predicted_citations': sum(len(p) for p in preds),
            'true_citations': sum(len(r) for r in refs)
        }
    
    def add_result(self, task, result):
        """添加结果"""
        for key, value in result.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def summary(self):
        """汇总"""
        return {k: sum(v)/len(v) if v else 0 for k, v in self.metrics.items()}
