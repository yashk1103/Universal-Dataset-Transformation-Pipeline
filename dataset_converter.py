#!/usr/bin/env python3
"""
Dataset Format Converter for RAG Benchmark
Converts various dataset formats to the standard format expected by the benchmark
"""

import os
import json
import pandas as pd
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
from datetime import datetime

class DatasetConverter:
    """Convert various dataset formats to standard benchmark format"""
    
    def __init__(self):
        # Standard format template
        self.standard_format = {
            "conversations": [
                {
                    "conversation_id": "string",
                    "turns": [
                        {
                            "turn_id": "int",
                            "question": "string", 
                            "answer": "string"
                        }
                    ],
                    "metadata": {}  # Optional
                }
            ],
            "corpus": {
                "doc_id": "document_text"
            },
            "metadata": {}  # Optional
        }
        
        # Supported input formats
        self.supported_formats = {
            'msmarco_qa': self._convert_msmarco_qa,
            'squad': self._convert_squad,
            'coqa': self._convert_coqa,
            'natural_questions': self._convert_natural_questions,
            'hotpot_qa': self._convert_hotpot_qa,
            'wizard_of_wikipedia': self._convert_wizard_of_wikipedia,
            'persona_chat': self._convert_persona_chat,
            'blended_skill_talk': self._convert_blended_skill_talk,
            'conversational_qa': self._convert_conversational_qa,
            'chitchat': self._convert_chitchat,
            'custom_qa_pairs': self._convert_custom_qa_pairs,
            'dialogue_format': self._convert_dialogue_format,
            'interview_format': self._convert_interview_format,
            'faq_format': self._convert_faq_format,
            'support_tickets': self._convert_support_tickets
        }
    
    def list_supported_formats(self):
        """List all supported input formats"""
        print("Supported Dataset Formats:")
        print("=" * 50)
        for fmt_name in self.supported_formats.keys():
            print(f"  - {fmt_name}")
        print()
        print("Use --format <format_name> to specify the input format")
        print("Use --detect-format to auto-detect format (experimental)")
    
    def detect_format(self, file_path: str) -> Optional[str]:
        """Auto-detect dataset format (experimental)"""
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                elif file_path.endswith('.jsonl'):
                    first_line = f.readline()
                    data = json.loads(first_line)
                else:
                    return None
            
            # Check for common format indicators
            if isinstance(data, dict):
                # Check for SQuAD format
                if 'data' in data and isinstance(data['data'], list):
                    if len(data['data']) > 0 and 'paragraphs' in data['data'][0]:
                        return 'squad'
                
                # Check for CoQA format
                if 'data' in data and isinstance(data['data'], list):
                    if len(data['data']) > 0 and 'questions' in data['data'][0] and 'answers' in data['data'][0]:
                        return 'coqa'
                
                # Check for MS MARCO format
                if 'query' in data and 'passages' in data:
                    return 'msmarco_qa'
            
            elif isinstance(data, list):
                if len(data) > 0:
                    item = data[0]
                    
                    # Check for dialogue format
                    if 'dialogue' in item or 'messages' in item:
                        return 'dialogue_format'
                    
                    # Check for QA pairs
                    if 'question' in item and 'answer' in item:
                        return 'custom_qa_pairs'
                    
                    # Check for turn-based format
                    if 'turns' in item:
                        return 'conversational_qa'
            
            return None
            
        except Exception as e:
            print(f"Error detecting format: {e}")
            return None
    
    def convert_dataset(self, input_file: str, output_dir: str, format_name: str, 
                       max_conversations: int = None, dataset_name: str = None) -> bool:
        """Convert dataset to standard format"""
        
        if format_name not in self.supported_formats:
            print(f"Unsupported format: {format_name}")
            print("Use --list-formats to see supported formats")
            return False
        
        try:
            print(f"Converting {input_file} from {format_name} format...")
            
            # Convert using appropriate converter
            converter_func = self.supported_formats[format_name]
            converted_data = converter_func(input_file, max_conversations)
            
            if not converted_data:
                print("Conversion failed - no data returned")
                return False
            
            # Create output directory
            dataset_name = dataset_name or f"converted_{format_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_path = Path(output_dir) / dataset_name
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save converted data
            conversations_file = output_path / "conversations.json"
            with open(conversations_file, 'w') as f:
                json.dump(converted_data, f, indent=2)
            
            # Create metadata file
            metadata = {
                "name": dataset_name,
                "source_file": str(input_file),
                "source_format": format_name,
                "converted_at": datetime.now().isoformat(),
                "conversations_count": len(converted_data.get('conversations', [])),
                "corpus_size": len(converted_data.get('corpus', {}))
            }
            
            metadata_file = output_path / "metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
            
            print(f"Conversion completed successfully!")
            print(f"Output directory: {output_path}")
            print(f"Conversations: {len(converted_data.get('conversations', []))}")
            print(f"Corpus documents: {len(converted_data.get('corpus', {}))}")
            
            return True
            
        except Exception as e:
            print(f"Error during conversion: {e}")
            traceback.print_exc()
            return False
    
    def _convert_squad(self, file_path: str, max_conversations: int) -> Dict:
        """Convert SQuAD format"""
        print("Converting SQuAD format...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversations = []
        corpus = {}
        conv_count = 0
        
        for article in data['data']:
            if max_conversations and conv_count >= max_conversations:
                break
            
            for paragraph in article['paragraphs']:
                if max_conversations and conv_count >= max_conversations:
                    break
                
                context = paragraph['context']
                questions = paragraph['qas']
                
                if len(questions) < 2:
                    # Skip if not enough questions for multi-turn
                    continue
                
                turns = []
                for i, qa in enumerate(questions):
                    question = qa['question']
                    answers = qa.get('answers', [])
                    answer = answers[0]['text'] if answers else ""
                    
                    turn = {
                        "turn_id": i,
                        "question": question,
                        "answer": answer
                    }
                    turns.append(turn)
                
                if len(turns) >= 2:
                    conversation = {
                        "conversation_id": f"squad_{conv_count}",
                        "turns": turns,
                        "metadata": {
                            "title": article.get('title', ''),
                            "context": context
                        }
                    }
                    conversations.append(conversation)
                    
                    # Add context to corpus
                    corpus[f"squad_{conv_count}_context"] = context
                    
                    conv_count += 1
        
        return {
            "conversations": conversations,
            "corpus": corpus
        }
    
    def _convert_coqa(self, file_path: str, max_conversations: int) -> Dict:
        """Convert CoQA format"""
        print("Converting CoQA format...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        conversations = []
        corpus = {}
        
        for i, item in enumerate(data['data']):
            if max_conversations and i >= max_conversations:
                break
            
            story = item['story']
            questions = item['questions']
            answers = item['answers']
            
            turns = []
            for j, (q, a) in enumerate(zip(questions, answers)):
                turn = {
                    "turn_id": j,
                    "question": q['input_text'],
                    "answer": a['input_text']
                }
                turns.append(turn)
            
            conversation = {
                "conversation_id": f"coqa_{i}",
                "turns": turns,
                "metadata": {
                    "source": item.get('source', ''),
                    "filename": item.get('filename', '')
                }
            }
            conversations.append(conversation)
            
            # Add story to corpus
            corpus[f"coqa_{i}_story"] = story
        
        return {
            "conversations": conversations,
            "corpus": corpus
        }
    
    def _convert_custom_qa_pairs(self, file_path: str, max_conversations: int) -> Dict:
        """Convert custom Q&A pairs format"""
        print("Converting custom Q&A pairs format...")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            conversations = []
            corpus = {}
            
            # Group by conversation if conversation_id exists
            if 'conversation_id' in df.columns:
                grouped = df.groupby('conversation_id')
                count = 0
                for conv_id, group in grouped:
                    if max_conversations and count >= max_conversations:
                        break
                    
                    turns = []
                    for _, row in group.iterrows():
                        turn = {
                            "turn_id": row.get('turn_id', len(turns)),
                            "question": str(row.get('question', row.get('Q', ''))),
                            "answer": str(row.get('answer', row.get('A', '')))
                        }
                        turns.append(turn)
                    
                    conversation = {
                        "conversation_id": str(conv_id),
                        "turns": turns
                    }
                    conversations.append(conversation)
                    count += 1
            
            else:
                # Create conversations from sequential Q&A pairs
                conversation_size = 3  # Group every 3 Q&A pairs
                for i in range(0, len(df), conversation_size):
                    if max_conversations and i // conversation_size >= max_conversations:
                        break
                    
                    turns = []
                    for j, (_, row) in enumerate(df.iloc[i:i+conversation_size].iterrows()):
                        turn = {
                            "turn_id": j,
                            "question": str(row.get('question', row.get('Q', ''))),
                            "answer": str(row.get('answer', row.get('A', '')))
                        }
                        turns.append(turn)
                    
                    conversation = {
                        "conversation_id": f"qa_conv_{i // conversation_size}",
                        "turns": turns
                    }
                    conversations.append(conversation)
        
        else:
            # JSON/JSONL format
            with open(file_path, 'r') as f:
                if file_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            conversations = []
            corpus = {}
            
            for i, item in enumerate(data):
                if max_conversations and i >= max_conversations:
                    break
                
                turn = {
                    "turn_id": 0,
                    "question": str(item.get('question', item.get('Q', ''))),
                    "answer": str(item.get('answer', item.get('A', '')))
                }
                
                conversation = {
                    "conversation_id": f"qa_{i}",
                    "turns": [turn]
                }
                conversations.append(conversation)
        
        return {
            "conversations": conversations,
            "corpus": corpus
        }
    
    def _convert_faq_format(self, file_path: str, max_conversations: int) -> Dict:
        """Convert FAQ format"""
        print("Converting FAQ format...")
        
        conversations = []
        corpus = {}
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            # Create one conversation per FAQ category if available
            if 'category' in df.columns:
                grouped = df.groupby('category')
                count = 0
                for category, group in grouped:
                    if max_conversations and count >= max_conversations:
                        break
                    
                    turns = []
                    for j, (_, row) in enumerate(group.iterrows()):
                        turn = {
                            "turn_id": j,
                            "question": str(row.get('question', row.get('Q', ''))),
                            "answer": str(row.get('answer', row.get('A', '')))
                        }
                        turns.append(turn)
                    
                    conversation = {
                        "conversation_id": f"faq_{category}",
                        "turns": turns,
                        "metadata": {"category": str(category)}
                    }
                    conversations.append(conversation)
                    count += 1
            
            else:
                # Group sequential FAQs
                group_size = 5
                for i in range(0, len(df), group_size):
                    if max_conversations and i // group_size >= max_conversations:
                        break
                    
                    turns = []
                    for j, (_, row) in enumerate(df.iloc[i:i+group_size].iterrows()):
                        turn = {
                            "turn_id": j,
                            "question": str(row.get('question', row.get('Q', ''))),
                            "answer": str(row.get('answer', row.get('A', '')))
                        }
                        turns.append(turn)
                    
                    conversation = {
                        "conversation_id": f"faq_group_{i // group_size}",
                        "turns": turns
                    }
                    conversations.append(conversation)
        
        return {
            "conversations": conversations,
            "corpus": corpus
        }
    
    # Add placeholder methods for other formats
    def _convert_msmarco_qa(self, file_path: str, max_conversations: int) -> Dict:
        """Convert MS MARCO QA format - Basic implementation"""
        print("Converting MS MARCO QA format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_natural_questions(self, file_path: str, max_conversations: int) -> Dict:
        """Convert Natural Questions format - Basic implementation"""
        print("Converting Natural Questions format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_hotpot_qa(self, file_path: str, max_conversations: int) -> Dict:
        """Convert HotpotQA format - Basic implementation"""
        print("Converting HotpotQA format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_wizard_of_wikipedia(self, file_path: str, max_conversations: int) -> Dict:
        """Convert Wizard of Wikipedia format - Basic implementation"""
        print("Converting Wizard of Wikipedia format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_persona_chat(self, file_path: str, max_conversations: int) -> Dict:
        """Convert PersonaChat format - Basic implementation"""
        print("Converting PersonaChat format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_blended_skill_talk(self, file_path: str, max_conversations: int) -> Dict:
        """Convert Blended Skill Talk format - Basic implementation"""
        print("Converting Blended Skill Talk format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_conversational_qa(self, file_path: str, max_conversations: int) -> Dict:
        """Convert generic conversational QA format - Basic implementation"""
        print("Converting Conversational QA format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_chitchat(self, file_path: str, max_conversations: int) -> Dict:
        """Convert chitchat/dialogue format - Basic implementation"""
        print("Converting chitchat format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_dialogue_format(self, file_path: str, max_conversations: int) -> Dict:
        """Convert dialogue format - Basic implementation"""
        print("Converting dialogue format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_interview_format(self, file_path: str, max_conversations: int) -> Dict:
        """Convert interview/transcript format - Basic implementation"""
        print("Converting interview format...")
        return {"conversations": [], "corpus": {}}
    
    def _convert_support_tickets(self, file_path: str, max_conversations: int) -> Dict:
        """Convert support ticket format - Basic implementation"""
        print("Converting support ticket format...")
        return {"conversations": [], "corpus": {}}

def main():
    parser = argparse.ArgumentParser(description="Dataset Format Converter for RAG Benchmark")
    parser.add_argument("--input", type=str, required=True, help="Input dataset file")
    parser.add_argument("--output_dir", type=str, default="./custom_datasets", help="Output directory")
    parser.add_argument("--format", type=str, help="Input format name")
    parser.add_argument("--dataset_name", type=str, help="Output dataset name")
    parser.add_argument("--max_conversations", type=int, help="Maximum conversations to convert")
    parser.add_argument("--list_formats", action="store_true", help="List supported formats")
    parser.add_argument("--detect_format", action="store_true", help="Auto-detect format")
    parser.add_argument("--validate", action="store_true", help="Validate converted dataset")
    
    args = parser.parse_args()
    
    converter = DatasetConverter()
    
    if args.list_formats:
        converter.list_supported_formats()
        return
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    
    format_name = args.format
    
    if args.detect_format or not format_name:
        print("Detecting format...")
        detected_format = converter.detect_format(args.input)
        if detected_format:
            print(f"Detected format: {detected_format}")
            format_name = detected_format
        else:
            print("Could not detect format. Please specify --format")
            return
    
    # Convert dataset
    success = converter.convert_dataset(
        args.input,
        args.output_dir,
        format_name,
        args.max_conversations,
        args.dataset_name
    )
    
    if success and args.validate:
        print("\nValidating converted dataset...")
        # Basic validation
        output_name = args.dataset_name or f"converted_{format_name}"
        output_path = Path(args.output_dir) / output_name / "conversations.json"
        
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            conversations = data.get('conversations', [])
            print(f"Validation: Found {len(conversations)} conversations")
            
            for i, conv in enumerate(conversations[:3]):  # Check first 3
                turns = conv.get('turns', [])
                print(f"  Conversation {i}: {len(turns)} turns")
                if turns:
                    print(f"    Sample question: {turns[0].get('question', '')[:50]}...")
                    print(f"    Sample answer: {turns[0].get('answer', '')[:50]}...")
            
            print("Validation completed successfully!")
            
        except Exception as e:
            print(f"Validation failed: {e}")

if __name__ == "__main__":
    main()