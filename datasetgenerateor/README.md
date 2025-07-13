# SMS Dataset Labeling Workflow

This directory contains scripts and tools for labeling SMS messages as Legit, Spam, or Fraud.

## Process:
1. Sample random messages from full dataset
2. Label small batch manually or with AI
3. Train basic classifier
4. Auto-label remaining messages
5. Review and iterate

## Files:
- `sample_data.py` - Extract random sample for labeling
- `auto_labeler.py` - AI-powered labeling script
- `train_classifier.py` - Train basic model
- `review_labels.py` - Review and correct predictions
