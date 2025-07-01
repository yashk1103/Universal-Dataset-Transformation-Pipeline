# Dataset Format Converter for Multi-Turn RAG Evaluation

A comprehensive toolkit for converting popular dataset formats into the standardized format required by multi-turn RAG evaluation systems. This converter enables you to use existing datasets from research papers and public repositories with your custom evaluation pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Supported Dataset Formats](#supported-dataset-formats)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Dataset Converter Usage](#dataset-converter-usage)
6. [Format Adapter Usage](#format-adapter-usage)
7. [Output Format Specification](#output-format-specification)
8. [Integration with RAG Evaluation Systems](#integration-with-rag-evaluation-systems)
9. [Conversion Examples](#conversion-examples)
10. [Batch Processing](#batch-processing)
11. [Validation and Quality Checks](#validation-and-quality-checks)
12. [Troubleshooting](#troubleshooting)
13. [Advanced Usage](#advanced-usage)

## Overview

### The Complete Pipeline
This toolkit provides a **2-script pipeline** for complete dataset processing:

```
Original Dataset → Dataset Converter → Format Adapter → RAG System Ready
     (SQuAD)           (converts)         (adapts)       (compatible)
```

### Required Scripts

**You need BOTH scripts for the complete workflow:**

| **Script** | **Purpose** | **Function** | **Status** |
|------------|-------------|--------------|------------|
| **`dataset_converter.py`** | Convert various formats | SQuAD → JSON, FAQ CSV → JSON, etc. | REQUIRED |
| **`format_adapter.py`** | Ensure perfect compatibility | Fixes converter output for benchmark | REQUIRED |

### The Challenge
Multi-turn RAG evaluation systems require datasets in a specific format with conversation structures, but most existing datasets come in various formats (SQuAD, CoQA, MS MARCO, etc.). Converting these manually is time-consuming and error-prone.

### The Solution
This toolkit provides:
- **Dataset Converter**: Converts 15+ popular dataset formats to a standard intermediate format
- **Format Adapter**: Ensures full compatibility with RAG evaluation benchmarks
- **Automatic Format Detection**: Intelligently detects input format types
- **Validation Tools**: Ensures converted data quality and structure

### Complete Workflow Steps

**STEP 1: Convert Dataset Format**
```bash
python dataset_converter.py \
  --input "./squad_data.json" \
  --format "squad" \
  --dataset_name "my_squad" \
  --max_conversations 50
```

**STEP 2: Adapt to Perfect Format**
```bash
python format_adapter.py \
  --input "./custom_datasets/my_squad/conversations.json" \
  --output "./ready/squad_ready.json"
```

### Integration Points
The converted datasets work seamlessly with:
- **Custom Model + Custom Dataset** evaluation
- **Domain-Specific evaluation** with external corpora
- **Single-turn to Multi-turn** conversion workflows
- **Existing Multi-turn Datasets** with custom datasets

## Supported Dataset Formats

### Research Dataset Formats

| **Format** | **Description** | **Source** | **Conversion Quality** |
|------------|-----------------|------------|------------------------|
| **SQuAD** | Stanford Question Answering Dataset | Research papers, Hugging Face | Full Support |
| **CoQA** | Conversational Question Answering | Research, Official releases | Full Support |
| **MS MARCO QA** | Microsoft Machine Reading Comprehension | Microsoft Research | Basic Support |
| **Natural Questions** | Real Google search queries | Google AI | Basic Support |
| **HotpotQA** | Multi-hop reasoning dataset | Research papers | Basic Support |
| **Wizard of Wikipedia** | Knowledge-grounded conversations | Facebook AI | Basic Support |
| **PersonaChat** | Personality-based conversations | Facebook AI | Basic Support |
| **Blended Skill Talk** | Multi-skill conversations | Facebook AI | Basic Support |

### Generic Formats

| **Format** | **Description** | **Use Case** | **Conversion Quality** |
|------------|-----------------|--------------|------------------------|
| **Custom QA Pairs** | Simple question-answer pairs | Custom datasets, APIs | Full Support |
| **FAQ Format** | Frequently Asked Questions | Knowledge bases, websites | Full Support |
| **Dialogue Format** | General conversation format | Chatbot logs, transcripts | Basic Support |
| **Interview Format** | Interview transcripts | Podcasts, interviews | Basic Support |
| **Support Tickets** | Customer support conversations | Help desk systems | Basic Support |
| **Conversational QA** | Generic multi-turn QA | Various sources | Basic Support |
| **Chitchat** | Casual conversation data | Social media, forums | Basic Support |

**Legend**:
- **Full Support**: Complete implementation with all features
- **Basic Support**: Core functionality implemented, may need customization

## Installation

### Requirements

```bash
# Install from requirements.txt (recommended)
pip install -r requirements.txt

# Or install manually
pip install pandas pyyaml
```

**Note**: `pathlib`, `argparse`, and `json` are part of Python's standard library and don't need separate installation.

### Complete Script Collection

**You need BOTH scripts for the complete ecosystem:**

```
format_adapter/
├── dataset_converter.py      # SCRIPT 1: Converts 15+ dataset formats
├── format_adapter.py         # SCRIPT 2: Ensures benchmark compatibility  
└── README.md                # This documentation
```

### Script Functions

**Script 1: Dataset Converter** (`dataset_converter.py`)
```
Input:  SQuAD, CoQA, MS MARCO, FAQ CSV, etc.
Output: JSON format (close to benchmark format)
Purpose: Converts various dataset formats to standardized JSON
```

**Script 2: Format Adapter** (`format_adapter.py`)  
```
Input:  Output from dataset converter
Output: Perfect format for RAG benchmark
Purpose: Adds required fields (topic, context), fixes structure
```

### What Each Script Does

**Dataset Converter** - The Format Translator:
```
SQuAD Dataset → Converts → JSON Format
FAQ CSV → Converts → JSON Format  
CoQA Dataset → Converts → JSON Format
... (15+ formats supported)
```

**Format Adapter** - The Compatibility Fixer:
```
Converter Output → Adapts → Perfect Benchmark Format
(adds topic field, context field, fixes structure)
```

### Verify Complete Installation

```bash
# Test Dataset Converter
python dataset_converter.py --list_formats

# Test Format Adapter
python format_adapter.py --help
```

## Quick Start

### Complete 2-Step Workflow

**STEP 1: Convert Dataset Format**
```bash
python dataset_converter.py \
  --input squad_train.json \
  --format squad \
  --output_dir ./converted_datasets \
  --dataset_name squad_converted \
  --max_conversations 100
```

**STEP 2: Adapt for Perfect Compatibility**
```bash
python format_adapter.py \
  --input ./converted_datasets/squad_converted/conversations.json \
  --output ./converted_datasets/squad_converted/conversations_adapted.json
```

### Quick Test of Complete Pipeline

```bash
# Test the complete ecosystem
# 1. List supported formats
python dataset_converter.py --list_formats

# 2. Convert a sample dataset
python dataset_converter.py \
  --input "./sample_data.csv" \
  --format "custom_qa_pairs" \
  --dataset_name "test_conversion" \
  --max_conversations 5

# 3. Adapt format
python format_adapter.py \
  --input "./custom_datasets/test_conversion/conversations.json"
```

### Alternative: Auto-Detection Workflow

```bash
# Let the system detect format automatically
python dataset_converter.py \
  --input unknown_dataset.json \
  --detect_format \
  --dataset_name auto_detected \
  --max_conversations 50

# Adapt the output
python format_adapter.py \
  --input ./custom_datasets/auto_detected/conversations.json
```

### Why You Need Both Scripts

**Using Only Dataset Converter:**
```bash
# This creates JSON, but it's NOT fully compatible
python dataset_converter.py --input squad.json --format squad
# Output may be missing required fields (topic, context), incompatible structure
```

**Skipping Dataset Converter:**
```bash
# This only works if your data is already in near-perfect format
python format_adapter.py --input raw_data.json  # LIKELY TO FAIL
# Problems: Can't handle SQuAD, CoQA, MS MARCO, and other research formats
```

**Using Complete Pipeline:**
```bash
# This ensures 100% compatibility and success
python dataset_converter.py --input squad.json --format squad
python format_adapter.py --input ./converted/conversations.json  
# Result: Perfect format, all fields present, guaranteed compatibility
```

## Dataset Converter Usage

### Basic Commands

```bash
# List all supported formats
python dataset_converter.py --list_formats

# Auto-detect format and convert
python dataset_converter.py \
  --input dataset.json \
  --detect_format \
  --output_dir ./converted

# Convert with specific format
python dataset_converter.py \
  --input dataset.json \
  --format squad \
  --output_dir ./converted \
  --dataset_name my_dataset
```

### Command Line Options

| **Option** | **Description** | **Example** |
|------------|-----------------|-------------|
| `--input` | Input dataset file (required) | `dataset.json` |
| `--output_dir` | Output directory | `./converted_datasets` |
| `--format` | Input format name | `squad`, `coqa`, `custom_qa_pairs` |
| `--dataset_name` | Output dataset name | `my_converted_dataset` |
| `--max_conversations` | Limit conversations converted | `100`, `500` |
| `--list_formats` | Show supported formats | N/A |
| `--detect_format` | Auto-detect input format | N/A |
| `--validate` | Validate converted output | N/A |

### Format-Specific Usage

**SQuAD Format**:
```bash
python dataset_converter.py \
  --input squad-dev-v2.0.json \
  --format squad \
  --max_conversations 200 \
  --dataset_name squad_dev_converted
```

**CoQA Format**:
```bash
python dataset_converter.py \
  --input coqa-train-v1.0.json \
  --format coqa \
  --max_conversations 150 \
  --dataset_name coqa_train_converted
```

**Custom QA Pairs (CSV)**:
```bash
python dataset_converter.py \
  --input qa_dataset.csv \
  --format custom_qa_pairs \
  --dataset_name custom_qa_converted
```

**FAQ Format**:
```bash
python dataset_converter.py \
  --input company_faq.csv \
  --format faq_format \
  --dataset_name company_faq_converted
```

### Auto-Detection Examples

```bash
# Let the system detect format automatically
python dataset_converter.py \
  --input unknown_format.json \
  --detect_format \
  --output_dir ./auto_converted

# Validate detection results
python dataset_converter.py \
  --input unknown_format.json \
  --detect_format \
  --validate
```

## Format Adapter Usage

### Purpose
The Format Adapter ensures converter output is fully compatible with RAG evaluation systems by:
- Adding required fields (`context`, `topic`)
- Standardizing conversation structure
- Extracting metadata for topic classification
- Validating output format

### Basic Commands

```bash
# Adapt single file
python format_adapter.py \
  --input converted_dataset/conversations.json \
  --output adapted_conversations.json

# Batch adapt directory
python format_adapter.py \
  --input ./converted_datasets \
  --output ./adapted_datasets \
  --batch
```

### Single File Adaptation

**Input**: Converter output format
```json
{
  "conversations": [
    {
      "conversation_id": "conv_1",
      "turns": [
        {"turn_id": 0, "question": "...", "answer": "..."}
      ],
      "metadata": {"title": "Science", "category": "education"}
    }
  ]
}
```

**Output**: RAG benchmark compatible format
```json
{
  "conversations": [
    {
      "conversation_id": "conv_1", 
      "turns": [
        {
          "turn_id": 0,
          "question": "...",
          "answer": "...",
          "context": ""
        }
      ],
      "topic": "Science"
    }
  ]
}
```

### Batch Processing

```bash
# Process entire directory tree
python format_adapter.py \
  --input ./all_converted_datasets \
  --output ./all_adapted_datasets \
  --batch

# This processes all conversations.json files found recursively
```

## Output Format Specification

### Standard Converted Format

```json
{
  "conversations": [
    {
      "conversation_id": "unique_identifier",
      "turns": [
        {
          "turn_id": 0,
          "question": "What is machine learning?",
          "answer": "Machine learning is a method of data analysis...",
          "context": "Previous conversation context (if any)"
        },
        {
          "turn_id": 1, 
          "question": "How does it work?",
          "answer": "It works by using algorithms to find patterns...",
          "context": "Cumulative conversation context"
        }
      ],
      "topic": "Technology",
      "metadata": {
        "source": "original_dataset_name",
        "original_id": "original_identifier"
      }
    }
  ],
  "corpus": {
    "doc_id_1": "Document text for retrieval evaluation...",
    "doc_id_2": "Another document for corpus..."
  },
  "metadata": {
    "conversion_info": {
      "source_format": "squad",
      "converted_at": "2024-01-15T10:30:00",
      "conversations_count": 150,
      "corpus_size": 150
    }
  }
}
```

### Directory Structure

```
converted_datasets/
└── dataset_name/
    ├── conversations.json          # Main converted data
    ├── metadata.yaml              # Conversion metadata
    └── conversations_adapted.json  # RAG benchmark ready (after adaptation)
```

**metadata.yaml Example**:
```yaml
name: squad_converted
source_file: squad-dev-v2.0.json
source_format: squad
converted_at: '2024-01-15T10:30:00'
conversations_count: 200
corpus_size: 200
```

## Integration with RAG Evaluation Systems

### 1. Custom Model + Custom Dataset

```bash
# Step 1: Convert your dataset
python dataset_converter.py \
  --input your_dataset.json \
  --format custom_qa_pairs \
  --dataset_name custom_converted

# Step 2: Adapt for compatibility
python format_adapter.py \
  --input ./custom_datasets/custom_converted/conversations.json

# The output is now ready for any RAG evaluation system
```

### 2. Domain-Specific Evaluation

```bash
# Convert domain-specific dataset
python dataset_converter.py \
  --input finance_qa.csv \
  --format faq_format \
  --dataset_name finance_conversations

# Adapt format
python format_adapter.py \
  --input ./converted_datasets/finance_conversations/conversations.json

# Output is ready for domain-specific RAG evaluation
```

### 3. Single-turn to Multi-turn Conversion

```bash
# Convert single-turn dataset to multi-turn conversations
python dataset_converter.py \
  --input single_turn_qa.json \
  --format custom_qa_pairs \
  --dataset_name multi_turn_converted \
  --max_conversations 50

# The converter automatically groups single Q&A pairs into multi-turn conversations
```

### 4. Existing Multi-turn + Custom Datasets

```bash
# Convert your custom dataset
python dataset_converter.py \
  --input custom_conversations.csv \
  --format conversational_qa \
  --dataset_name custom_conv

# Adapt for use alongside existing datasets
python format_adapter.py \
  --input ./converted_datasets/custom_conv/conversations.json
```

## Conversion Examples

### Example 1: Converting SQuAD to Multi-turn

**Input (SQuAD format)**:
```json
{
  "data": [
    {
      "title": "Machine Learning",
      "paragraphs": [
        {
          "context": "Machine learning is a method of data analysis that automates analytical model building...",
          "qas": [
            {
              "question": "What is machine learning?",
              "answers": [{"text": "a method of data analysis", "answer_start": 25}]
            },
            {
              "question": "What does it automate?", 
              "answers": [{"text": "analytical model building", "answer_start": 67}]
            }
          ]
        }
      ]
    }
  ]
}
```

**Command**:
```bash
python dataset_converter.py \
  --input squad_sample.json \
  --format squad \
  --dataset_name squad_ml_sample \
  --validate
```

**Output**:
```json
{
  "conversations": [
    {
      "conversation_id": "squad_0",
      "turns": [
        {
          "turn_id": 0,
          "question": "What is machine learning?",
          "answer": "a method of data analysis"
        },
        {
          "turn_id": 1,
          "question": "What does it automate?",
          "answer": "analytical model building"
        }
      ],
      "metadata": {
        "title": "Machine Learning",
        "context": "Machine learning is a method of data analysis..."
      }
    }
  ],
  "corpus": {
    "squad_0_context": "Machine learning is a method of data analysis that automates analytical model building..."
  }
}
```

### Example 2: Converting CSV QA Pairs

**Input (CSV format)**:
```csv
conversation_id,turn_id,question,answer,category
conv_1,0,What is AI?,Artificial Intelligence refers to...,Technology
conv_1,1,How does it learn?,AI learns through algorithms...,Technology
conv_2,0,What is blockchain?,Blockchain is a distributed ledger...,Technology
conv_2,1,Why is it secure?,It uses cryptographic hashing...,Technology
```

**Command**:
```bash
python dataset_converter.py \
  --input tech_qa.csv \
  --format custom_qa_pairs \
  --dataset_name tech_conversations \
  --max_conversations 50
```

### Example 3: Converting FAQ to Conversations

**Input (FAQ CSV)**:
```csv
category,question,answer
Billing,How do I pay my bill?,You can pay through our website...,
Billing,When is payment due?,Payment is due on the 15th...,
Support,How do I reset password?,Click on 'Forgot Password'...,
Support,Who do I contact?,Contact support@company.com...,
```

**Command**:
```bash
python dataset_converter.py \
  --input company_faq.csv \
  --format faq_format \
  --dataset_name company_support
```

**Result**: Creates multi-turn conversations grouped by category (Billing, Support, etc.)

## Batch Processing

### Converting Multiple Datasets

```bash
# Process all datasets in directory
for dataset in datasets/*.json; do
    python dataset_converter.py \
      --input "$dataset" \
      --detect_format \
      --output_dir ./batch_converted \
      --validate
done

# Batch adapt all converted datasets
python format_adapter.py \
  --input ./batch_converted \
  --output ./batch_adapted \
  --batch
```

### Automated Pipeline Script

```bash
#!/bin/bash
# automated_conversion.sh

INPUT_DIR="./raw_datasets"
CONVERTED_DIR="./converted_datasets" 
ADAPTED_DIR="./adapted_datasets"

echo "Starting batch dataset conversion..."

# Convert all datasets
for file in "$INPUT_DIR"/*.{json,csv,jsonl}; do
    if [[ -f "$file" ]]; then
        filename=$(basename "$file" | cut -d. -f1)
        echo "Converting $filename..."
        
        python dataset_converter.py \
          --input "$file" \
          --detect_format \
          --output_dir "$CONVERTED_DIR" \
          --dataset_name "$filename" \
          --validate
    fi
done

# Batch adapt all converted datasets
echo "Adapting all converted datasets..."
python format_adapter.py \
  --input "$CONVERTED_DIR" \
  --output "$ADAPTED_DIR" \
  --batch

echo "Conversion pipeline completed!"
echo "Adapted datasets ready in: $ADAPTED_DIR"
```

Usage:
```bash
chmod +x automated_conversion.sh
./automated_conversion.sh
```

## Validation and Quality Checks

### Built-in Validation

```bash
# Validate during conversion
python dataset_converter.py \
  --input dataset.json \
  --format squad \
  --validate

# Validate after adaptation
python format_adapter.py \
  --input conversations.json \
  --output adapted.json
# Automatic validation included
```

### Manual Validation Checks

**Check Conversation Structure**:
```bash
python -c "
import json
with open('conversations.json', 'r') as f:
    data = json.load(f)
    
conversations = data.get('conversations', [])
print(f'Total conversations: {len(conversations)}')

for i, conv in enumerate(conversations[:3]):
    turns = conv.get('turns', [])
    topic = conv.get('topic', 'unknown')
    print(f'Conversation {i}: {len(turns)} turns, topic: {topic}')
"
```

**Check Data Quality**:
```bash
python -c "
import json
with open('conversations.json', 'r') as f:
    data = json.load(f)
    
# Check for empty questions/answers
empty_count = 0
for conv in data['conversations']:
    for turn in conv['turns']:
        if not turn.get('question', '').strip() or not turn.get('answer', '').strip():
            empty_count += 1

print(f'Empty question/answer pairs: {empty_count}')

# Check conversation lengths
lengths = [len(conv['turns']) for conv in data['conversations']]
print(f'Avg turns per conversation: {sum(lengths)/len(lengths):.1f}')
print(f'Min turns: {min(lengths)}, Max turns: {max(lengths)}')
"
```

### Quality Metrics

**Good Conversion Indicators**:
- Conversations have 2+ turns
- Questions and answers are non-empty
- Topics are properly extracted
- Corpus documents match conversations
- Metadata is preserved

**Warning Signs**:
- Many single-turn conversations
- High percentage of empty answers
- Missing topic information
- Very short or very long conversations

## Troubleshooting

### Common Issues

**Format Detection Fails**:
```bash
# Solution: Specify format manually
python dataset_converter.py \
  --input dataset.json \
  --format squad \
  --output_dir ./converted
```

**Empty Conversations Generated**:
```
Cause: Dataset doesn't have enough Q&A pairs per context
Solution: Adjust grouping logic or use different format
```

**Memory Issues with Large Datasets**:
```bash
# Solution: Process in chunks
python dataset_converter.py \
  --input large_dataset.json \
  --format squad \
  --max_conversations 100 \
  --dataset_name chunk_1
```

**Missing Required Fields**:
```
Error: KeyError: 'question'
Solution: Check your dataset has the expected field names
```

### Debug Information

**Enable Verbose Output**:
```python
# Add at the beginning of converter functions
print(f"Processing {len(data)} items...")
print(f"Sample item keys: {list(data[0].keys()) if data else 'No data'}")
```

**Check Intermediate Output**:
```bash
# Convert small sample first
python dataset_converter.py \
  --input dataset.json \
  --format custom_qa_pairs \
  --max_conversations 5 \
  --validate
```

### Error Recovery

**Partial Conversion Failure**:
```python
# Resume from last successful conversation
# Modify max_conversations to skip processed items
python dataset_converter.py \
  --input dataset.json \
  --format squad \
  --max_conversations 200 \
  --dataset_name recovery_attempt
```

**Invalid JSON Output**:
```bash
# Validate JSON syntax
python -m json.tool conversations.json > /dev/null
echo "JSON is valid"

# Fix common JSON issues
python -c "
import json
import re

with open('conversations.json', 'r') as f:
    content = f.read()

# Fix common JSON issues
content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
content = re.sub(r',\s*]', ']', content)

with open('conversations_fixed.json', 'w') as f:
    f.write(content)
"
```

## Advanced Usage

### Custom Format Implementation

To add support for a new dataset format:

1. **Add format to supported_formats dict**:
```python
self.supported_formats['my_custom_format'] = self._convert_my_custom_format
```

2. **Implement converter function**:
```python
def _convert_my_custom_format(self, file_path: str, max_conversations: int) -> Dict:
    """Convert My Custom Format"""
    print("Converting My Custom Format...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    conversations = []
    corpus = {}
    
    # Your conversion logic here
    for item in data:
        # Extract conversations and corpus
        pass
    
    return {
        "conversations": conversations,
        "corpus": corpus
    }
```

### Integration with External Tools

**Hugging Face Datasets Integration**:
```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("squad", split="validation")

# Convert to JSON for processing
dataset.to_json("squad_validation.json")

# Convert with our tool
python dataset_converter.py \
  --input squad_validation.json \
  --format squad \
  --dataset_name squad_hf_converted
```

**API Data Integration**:
```python
import requests
import json

# Fetch data from API
response = requests.get("https://api.example.com/qa-data")
data = response.json()

# Save to file
with open("api_data.json", "w") as f:
    json.dump(data, f)

# Convert
python dataset_converter.py \
  --input api_data.json \
  --format custom_qa_pairs \
  --dataset_name api_converted
```

### Performance Optimization

**Large Dataset Processing**:
```bash
# Process in parallel chunks
split -l 1000 large_dataset.jsonl chunk_
for chunk in chunk_*; do
    python dataset_converter.py \
      --input "$chunk" \
      --format custom_qa_pairs \
      --dataset_name "chunk_$(basename $chunk)" &
done
wait

# Merge results
python -c "
import json
import glob

all_conversations = []
all_corpus = {}

for file in glob.glob('converted_datasets/chunk_*/conversations.json'):
    with open(file, 'r') as f:
        data = json.load(f)
        all_conversations.extend(data['conversations'])
        all_corpus.update(data.get('corpus', {}))

merged_data = {
    'conversations': all_conversations,
    'corpus': all_corpus
}

with open('merged_dataset.json', 'w') as f:
    json.dump(merged_data, f, indent=2)
"
```

This comprehensive toolkit enables seamless integration of existing datasets with your multi-turn RAG evaluation systems, supporting research reproducibility and enabling evaluation on diverse, high-quality conversational datasets.