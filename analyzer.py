from transformers import pipeline
from config import Config
import torch
from typing import List, Dict
from collections import defaultdict

class ConversationAnalyzer:
    def __init__(self):
        self.config = Config()
        self._init_models()

    def _init_models(self):
        """Initialize analysis models"""
        device = 0 if torch.cuda.is_available() else -1

        self.sentiment_pipeline = pipeline(
            "text-classification",
            model=self.config.SENTIMENT_MODEL,
            device=device
        )

        self.tonality_pipeline = pipeline(
            "text-classification",
            model=self.config.TONALITY_MODEL,
            device=device
        )

        self.intent_pipeline = pipeline(
            "zero-shot-classification",
            model=self.config.INTENT_MODEL,
            device=device
        )

    def normalize_sentiment(self, label: str) -> str:
        """Map model-specific sentiment label to POSITIVE, NEGATIVE, or NEUTRAL"""
        label = label.strip().lower()

        if label in {"positive", "4 stars", "5 stars", "LABEL_2"}:
            return "POSITIVE"
        elif label in {"negative", "1 star", "2 stars", "LABEL_0"}:
            return "NEGATIVE"
        elif label in {"neutral", "3 stars", "LABEL_1"}:
            return "NEUTRAL"
        else:
            return "NEUTRAL"  # fallback for unknown

    def analyze_conversation(self, conversation: List[Dict]) -> List[Dict]:
        """Analyze conversation for sentiment, tonality and intent"""
        analysis_results = []
        intent_candidate_labels = ["complaint", "query", "feedback", "sales", "technical support"]

        for utterance in conversation:
            text = utterance["text"]
            speaker = utterance["speaker"]

            sentiment = self.sentiment_pipeline(text)[0]
            tonality = self.tonality_pipeline(text)[0]
            intent = self.intent_pipeline(
                text,
                candidate_labels=intent_candidate_labels,
                multi_label=True
            )

            analysis_results.append({
                "speaker": speaker,
                "text": text,
                "sentiment": {
                    "label": sentiment["label"],
                    "score": sentiment["score"]
                },
                "tonality": {
                    "label": tonality["label"],
                    "score": tonality["score"]
                },
                "intent": {
                    "labels": intent["labels"],
                    "scores": intent["scores"]
                }
            })

        return analysis_results


    def get_conversation_summary_stats(self, analysis_results: List[Dict]) -> Dict:
        """Generate summary statistics for the conversation"""
        sentiment_counts = defaultdict(int)
        tonality_counts = defaultdict(int)
        intent_scores = defaultdict(float)

        for result in analysis_results:
            sentiment_raw = result["sentiment"]["label"]
            sentiment = self.normalize_sentiment(sentiment_raw)
            sentiment_counts[sentiment] += 1

            tonality = result["tonality"]["label"]
            tonality_counts[tonality] += 1

            for label, score in zip(result["intent"]["labels"], result["intent"]["scores"]):
                intent_scores[label] += score

        total_intent_score = sum(intent_scores.values()) or 1.0
        normalized_intents = {k: v / total_intent_score for k, v in intent_scores.items()}

        return {
            "sentiment_distribution": dict(sentiment_counts),
            "tonality_distribution": dict(tonality_counts),
            "intent_distribution": normalized_intents
        }