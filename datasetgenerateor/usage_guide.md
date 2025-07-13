# SMS Dataset Labeling - Usage Guide

## Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

## Step-by-Step Process

### 1. Sample Data for Labeling
Extract a random sample from your SMS dataset:

```bash
python sample_data.py your_sms_data.csv -n 1000 -o sample_for_labeling.csv
```

**Parameters:**
- `your_sms_data.csv`: Your full SMS dataset (must have 'body' column)
- `-n 1000`: Number of messages to sample (default: 1000)
- `-o sample_for_labeling.csv`: Output file name

### 2. Label the Sample

**Option A: Manual Labeling**
1. Open `sample_for_labeling.csv` in Excel or Google Sheets
2. Fill in the 'label' column with: `legit`, `spam`, or `fraud`
3. Save the file as `sample_labeled.csv`

**Option B: Auto-Labeling with AI**
```bash
python auto_labeler.py sample_for_labeling.csv -o sample_auto_labeled.csv
```

This will automatically label messages using AI rules. Review the output, especially messages with low confidence scores.

### 3. Train a Basic Classifier

Once you have labeled data, train a classifier:

```bash
python train_classifier.py sample_labeled.csv -o sms_classifier.pkl
```

**Parameters:**
- `sample_labeled.csv`: Your labeled data
- `-o sms_classifier.pkl`: Output model file
- `--test-size 0.2`: Fraction of data for testing (default: 0.2)
- `--cross-validation`: Enable cross-validation

### 4. Label Remaining Data

Use your trained model to label the rest of your dataset:

```bash
python label_remaining.py your_sms_data.csv sms_classifier.pkl -o fully_labeled.csv
```

### 5. Review and Iterate

1. Review the auto-labeled messages, especially those with low confidence
2. Correct any mistakes
3. Add corrected messages to your training set
4. Retrain the model (go back to step 3)
5. Repeat until satisfied with quality

## File Formats

### Input CSV Format
Your SMS data should have at least these columns:
- `body`: The SMS message text
- `address` or `sender`: The sender (optional but helpful)
- `date`: Timestamp (optional)

### Label Format
Use exactly these labels (case-insensitive):
- `legit`: Legitimate messages (personal, work, OTPs, etc.)
- `spam`: Unsolicited marketing, offers, etc.
- `fraud`: Scams, phishing, suspicious international numbers

## Tips for Better Results

1. **Quality over Quantity**: 500 well-labeled messages are better than 2000 poorly labeled ones
2. **Balanced Dataset**: Try to have roughly equal numbers of each class
3. **Review Uncertain Cases**: Always review messages with confidence < 0.6
4. **Iterate**: The more you correct and retrain, the better your model becomes
5. **Keep Notes**: Document your labeling criteria for consistency

## Troubleshooting

### Common Issues:
- **"Column not found"**: Make sure your CSV has the required columns
- **"No labeled data"**: Check that your label column contains valid labels
- **Low accuracy**: You may need more training data or better labeling

### Getting Help:
- Check the error messages - they usually explain what's wrong
- Make sure your CSV file is properly formatted
- Verify that labels are spelled correctly: `legit`, `spam`, `fraud` 