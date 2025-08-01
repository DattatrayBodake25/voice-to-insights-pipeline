from transcriber import Transcriber
from analyzer import ConversationAnalyzer
from summarizer import ConversationSummarizer
from rag_agent import RAGAgent
from evaluator import SystemEvaluator
import json
import os

import warnings
warnings.filterwarnings('ignore')

class CustomerVoiceAnalyticsPipeline:
    def __init__(self):
        self.transcriber = Transcriber()
        self.analyzer = ConversationAnalyzer()
        self.summarizer = ConversationSummarizer()
        self.rag_agent = RAGAgent()
        self.evaluator = SystemEvaluator()
    
    def process_call(self, audio_path: str, output_dir: str = "output"):
        """Process a single call through the entire pipeline"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Transcribe audio with diarization
        print("Transcribing audio...")
        transcription = self.transcriber.transcribe_with_diarization(audio_path)
        transciption_path = os.path.join(output_dir, "transcription.json")
        self.transcriber.save_transcription(transcription, transciption_path)
        
        # Step 2: Analyze conversation
        print("Analyzing conversation...")
        analysis = self.analyzer.analyze_conversation(transcription)
        analysis_path = os.path.join(output_dir, "analysis.json")
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        
        # Step 3: Generate summary
        print("Generating summary...")
        summary_stats = self.analyzer.get_conversation_summary_stats(analysis)
        summary = self.summarizer.generate_summary(transcription, summary_stats)
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        
        # Step 4: Get follow-up actions
        print("Generating follow-up actions...")
        followup_actions = self.rag_agent.get_followup_actions(summary)
        followup_path = os.path.join(output_dir, "followup_actions.txt")
        with open(followup_path, "w") as f:
            f.write(followup_actions)
        
        print(f"Processing complete. Results saved to {output_dir}")
        
        return {
            "transcription": transcription,
            "analysis": analysis,
            "summary": summary,
            "followup_actions": followup_actions
        }

if __name__ == "__main__":
    pipeline = CustomerVoiceAnalyticsPipeline()
    
    # Example usage
    audio_file = r"C:\Users\bodak\Desktop\customer_voice_analytics\data\test_audios\call_recording_15.wav"
    results = pipeline.process_call(audio_file)
    
    # Print summary of results
    print("\n=== Conversation Summary ===")
    print(results["summary"])
    
    print("\n=== Follow-up Actions ===")
    print(results["followup_actions"])