# CyberGuard — ML Bullying Detector

Cyberbullying & toxic content detection using an ensemble ML model.
Training data includes real-world profanity and abusive language patterns.

## Quick Start

  pip install -r requirements.txt
  python app.py
  # Open http://localhost:5050

## Files
  app.py              — Flask backend + model logic
  model.pkl           — Pre-trained model (97.4% accuracy, ready to use)
  static/index.html   — Frontend UI (same design as reference)
  requirements.txt    — Python dependencies

## API
  POST /predict  { "text": "..." }          → analysis result
  POST /batch    { "texts": ["...", ...] }  → up to 20 results
  GET  /stats                               → model info

## Notes
  - Delete model.pkl to force retrain on startup
  - For educational and research use only
