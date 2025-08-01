from google.generativeai import configure, GenerativeModel
from config import Config
from typing import List, Dict

class ConversationSummarizer:
    def __init__(self):
        self.config = Config()
        configure(api_key=self.config.GEMINI_API_KEY)
        self.model = GenerativeModel("gemini-2.0-flash")
    
    def generate_summary(self, conversation: List[Dict], analysis_results: Dict) -> str:
        """Generate a summary of the conversation"""
        try:
            # Format conversation for summarization
            conv_text = "\n".join(
                [f"{utt['speaker']}: {utt['text']}" for utt in conversation]
            )
            
            summary_stats = analysis_results
            
            prompt = f"""
            Analyze the following customer support conversation and provide a concise summary 
            based on the given analysis. Focus on key points, customer emotions, and agent responses.
            
            Conversation:
            {conv_text}
            
            Analysis:
            - Overall sentiment: {summary_stats['sentiment_distribution']}
            - Tonality patterns: {summary_stats['tonality_distribution']}
            - Primary intents: {summary_stats['intent_distribution']}
            
            Provide a 3-4 sentence summary highlighting:
            1. The main purpose of the call
            2. Customer's emotional state
            3. Key resolution points
            4. Any follow-up needed
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            raise Exception(f"Summarization error: {str(e)}")