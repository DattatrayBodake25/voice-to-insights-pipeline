# Voice-to-Insights Pipeline

This project is an end-to-end AI pipeline that transforms **customer-agent phone calls** into **actionable business insights** using automatic speech recognition (ASR), large language models (LLMs), and retrieval-augmented generation (RAG). It supports **multilingual input** (English, Hindi, Hinglish), **speaker diarization**, **sentiment/tonality/intent analysis**, and **summary generation** with context-aware **follow-up action recommendations**.

---

## Features

- Transcribe customer support audio recordings using Whisper
- Diarize speakers (agent vs. customer)
- Analyze each utterance for:
- **Sentiment** (Positive, Neutral, Negative)
- **Tonality** (e.g., Angry, Polite, Calm)
- **Intent** (e.g., Complaint, Query, Feedback)  
- Summarize full conversations using Gemini LLM  
- Suggest contextual follow-up actions using a RAG pipeline
- Evaluate transcription, insights, and summaries
---

## Project Structure
```
├── audio_processor.py                          # Audio loading, normalization, silence removal
├── analyzer.py                                 # Sentiment, Tonality, Intent analysis
├── config.py                                   # Centralized configuration and model paths
├── evaluator.py                                # Metrics: WER, F1, cosine similarity, etc.
├── rag_agent.py                                # FAISS + Gemini-based follow-up action generator
├── summarizer.py                               # Gemini-based conversation summarizer
├── transcriber.py                              # Whisper STT + pyannote diarization
├── utils.py                                    # Helpers: clean folders, generate sample docs
├── data/
│ ├── documents/                                # Sample documents for vector DB (RAG)
│ ├── test_audios/                              # 3–5 sample customer call recordings
│ └── vector_db/                                # FAISS vector DB generated from docs
├── outputs/
│ ├── transcription.json                        # Diarized and transcribed conversation
│ ├── analysis.json                             # Per-utterance sentiment/intent/tonality
│ ├── summary.txt                               # Generated summary
│ └── followup.txt                              # Suggested follow-up actions
└── .env                                        # API keys (not committed)
```

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [HuggingFace account](https://huggingface.co) (for Whisper + pyannote)
- [Google API Key for Gemini](https://ai.google.dev/)
- GPU (recommended for diarization)

Install dependencies:

```bash
pip install -r requirements.txt
```
.env file (required):
```bash
HF_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_google_api_key
```

### 2. Input Requirements
 - Upload 3–5 customer support call recordings to data/test_audios/
 - Format: WAV or MP3 (mono/stereo, max 5 minutes)
 - Optional: Add sample FAQs/policies as .txt files in data/documents/

### 3. Run the Pipeline (Example Flow)
```python
# Step 1: Transcribe and diarize audio
from transcriber import Transcriber
t = Transcriber()
transcription = t.transcribe_with_diarization("data/test_audios/sample.wav")
t.save_transcription(transcription, "outputs/transcription.json")

# Step 2: Analyze sentiments, tonality, and intents
from analyzer import ConversationAnalyzer
analyzer = ConversationAnalyzer()
analysis = analyzer.analyze_conversation(transcription)
summary_stats = analyzer.get_conversation_summary_stats(analysis)

# Step 3: Generate summary
from summarizer import ConversationSummarizer
summarizer = ConversationSummarizer()
summary = summarizer.generate_summary(transcription, summary_stats)

# Step 4: Generate follow-up actions using RAG
from rag_agent import RAGAgent
rag = RAGAgent()
followups = rag.get_followup_actions(summary)

# Save outputs
import json
with open("outputs/analysis.json", "w") as f: json.dump(analysis, f, indent=2)
with open("outputs/summary.txt", "w") as f: f.write(summary)
with open("outputs/followup.txt", "w") as f: f.write(followups)
```

### 4. Evaluation
Evaluate transcription, summary, and insight quality using:
```python
from evaluator import SystemEvaluator
evaluator = SystemEvaluator()

# Example: Evaluate transcription quality
metrics = evaluator.evaluate_transcription(ground_truth, predicted)
print(metrics)  # {'word_error_rate': ..., 'speaker_accuracy': ...}
```

### Models Used
```
Task	                 Model
STT	                 openai/whisper-tiny (via HuggingFace Transformers)
Diarization	         pyannote/speaker-diarization
Sentiment	           nlptown/bert-base-multilingual-uncased-sentiment
Tonality	            bhadresh-savani/bert-base-go-emotion
Intent Detection	    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
RAG Embeddings	      sentence-transformers/all-MiniLM-L6-v2
LLM	                 gemini-2.0-flash (Google Generative AI)
```

##  Highlights
- Multilingual Audio Support: Hindi, English, Hinglish
- Speaker-Aware Transcription: Diarization with time segments
- Zero-shot Intent Analysis
- LLM-Based Summary & Suggestions
- RAG for Realistic, Contextual Responses

## Example Output (Real Sample)
Transcription:
```json
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.03,
    "end": 23.47,
    "text": "I'm just calling to say I'm absolutely blown away by the camera pro one..."
  }
]
```

## Summary:
Ashley Perez called to express overwhelming positive feedback regarding the "camera pro one"...

## Follow-up Suggestions:
- Send thank you email
- Ask for testimonial
- Offer discount (optional)

## Future Improvements
- Add CLI or Streamlit UI for user-friendly interaction
- Support longer recordings with chunked transcription
- Integrate automated feedback loop to improve model selection
- Add multilingual prompt translation for Gemini
