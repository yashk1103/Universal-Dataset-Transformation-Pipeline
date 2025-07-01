#!/usr/bin/env python3
"""
Format Adapter for Dataset Converter Output
Ensures converter output is fully compatible with RAG benchmark
"""

import json
import argparse
from pathlib import Path

def adapt_converter_output(input_file: str, output_file: str = None):
    """
    Adapt dataset converter output to be fully compatible with RAG benchmark
    """
    
    if output_file is None:
        output_file = input_file.replace('.json', '_adapted.json')
    
    print(f"Adapting {input_file} to {output_file}")
    
    try:
        # Load converter output
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Original conversations: {len(data.get('conversations', []))}")
        print(f"Original corpus: {len(data.get('corpus', {}))}")
        
        # Adapt format
        adapted_conversations = []
        
        for conv in data.get('conversations', []):
            adapted_conv = {
                "conversation_id": conv.get('conversation_id', ''),
                "turns": []
            }
            
            # Extract topic from metadata if available
            metadata = conv.get('metadata', {})
            if metadata:
                # Try different metadata fields that could be topic
                topic = (metadata.get('title', '') or 
                        metadata.get('category', '') or 
                        metadata.get('chosen_topic', '') or 
                        metadata.get('source', '') or
                        'unknown')
                adapted_conv['topic'] = str(topic)
            
            # Adapt turns
            for turn in conv.get('turns', []):
                adapted_turn = {
                    "turn_id": turn.get('turn_id', 0),
                    "question": turn.get('question', ''),
                    "answer": turn.get('answer', ''),
                    "context": turn.get('context', '')  # Add empty context if missing
                }
                adapted_conv['turns'].append(adapted_turn)
            
            adapted_conversations.append(adapted_conv)
        
        # Create adapted data structure
        adapted_data = {
            "conversations": adapted_conversations,
            "corpus": data.get('corpus', {})
        }
        
        # Save adapted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(adapted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Adaptation completed successfully!")
        print(f"Adapted conversations: {len(adapted_conversations)}")
        print(f"Output file: {output_file}")
        
        # Validation
        print("\nValidation:")
        for i, conv in enumerate(adapted_conversations[:3]):
            turns = conv.get('turns', [])
            topic = conv.get('topic', 'unknown')
            print(f"  Conversation {i}: {len(turns)} turns, topic: '{topic}'")
            if turns:
                print(f"    Turn 0: Q='{turns[0]['question'][:50]}...'")
                print(f"    Turn 0: A='{turns[0]['answer'][:50]}...'")
        
        return True
        
    except Exception as e:
        print(f"Error during adaptation: {e}")
        return False

def batch_adapt_directory(input_dir: str, output_dir: str = None):
    """
    Adapt all conversation.json files in a directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path.parent / f"{input_path.name}_adapted"
    
    output_path.mkdir(exist_ok=True)
    
    print(f"Batch adapting from {input_path} to {output_path}")
    
    # Find all conversations.json files
    conv_files = list(input_path.rglob("conversations.json"))
    
    print(f"Found {len(conv_files)} conversation files")
    
    for conv_file in conv_files:
        # Create corresponding output directory structure
        rel_path = conv_file.relative_to(input_path)
        output_file = output_path / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {rel_path}")
        success = adapt_converter_output(str(conv_file), str(output_file))
        
        if success:
            print(f" Adapted: {output_file}")
        else:
            print(f" Failed: {conv_file}")

def main():
    parser = argparse.ArgumentParser(description="Adapt dataset converter output for RAG benchmark")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, help="Output file or directory")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_adapt_directory(args.input, args.output)
    else:
        adapt_converter_output(args.input, args.output)

if __name__ == "__main__":
    main()