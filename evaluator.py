from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class SystemEvaluator:
    def __init__(self):
        pass
    
    def evaluate_transcription(self, ground_truth: List[Dict], predicted: List[Dict]) -> Dict:
        """Evaluate transcription accuracy"""
        # Simple word error rate calculation
        gt_text = " ".join([utt["text"] for utt in ground_truth])
        pred_text = " ".join([utt["text"] for utt in predicted])
        
        gt_words = gt_text.split()
        pred_words = pred_text.split()
        
        # Calculate word error rate (simplified)
        correct = 0
        min_len = min(len(gt_words), len(pred_words))
        for i in range(min_len):
            if gt_words[i] == pred_words[i]:
                correct += 1
        
        wer = 1 - (correct / max(len(gt_words), len(pred_words)))
        
        # Speaker diarization accuracy
        speaker_acc = accuracy_score(
            [utt["speaker"] for utt in ground_truth][:min_len],
            [utt["speaker"] for utt in predicted][:min_len]
        )
        
        return {
            "word_error_rate": wer,
            "speaker_accuracy": speaker_acc
        }
    
    def evaluate_analysis(self, ground_truth: Dict, predicted: Dict) -> Dict:
        """Evaluate sentiment/tonality/intent analysis"""
        metrics = {}
        
        # Sentiment evaluation
        if "sentiment" in ground_truth:
            metrics["sentiment_f1"] = f1_score(
                ground_truth["sentiment"],
                predicted["sentiment"],
                average="weighted"
            )
        
        # Tonality evaluation
        if "tonality" in ground_truth:
            metrics["tonality_f1"] = f1_score(
                ground_truth["tonality"],
                predicted["tonality"],
                average="weighted"
            )
        
        # Intent evaluation
        if "intent" in ground_truth:
            metrics["intent_accuracy"] = accuracy_score(
                ground_truth["intent"],
                predicted["intent"]
            )
        
        return metrics
    
    def evaluate_summary(self, ground_truth: str, predicted: str) -> Dict:
        """Evaluate summary quality using simple similarity"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer().fit_transform([ground_truth, predicted])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        
        return {
            "summary_similarity": similarity
        }