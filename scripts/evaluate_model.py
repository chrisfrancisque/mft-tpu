"""
Main script for evaluating models on all benchmarks.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from pathlib import Path

from evaluate.evaluator import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--domain', type=str, choices=['math', 'coding', 'instruction', 'all'],
                       default='all', help='Domain to evaluate')
    parser.add_argument('--dataset', type=str,
                       help='Specific dataset to evaluate')
    parser.add_argument('--max_samples', type=int,
                       help='Maximum samples to evaluate per dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results to disk')
    args = parser.parse_args()
    
    logger.info(f"Evaluating model: {args.model_path}")
    
    # Initialize evaluator
    evaluator = Evaluator(
        model_path=args.model_path,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    if args.domain == 'all':
        logger.info("Evaluating on all domains")
        results = evaluator.evaluate_all_domains(
            max_samples=args.max_samples,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        for domain, domain_results in results.items():
            print(f"\n{domain.upper()}:")
            for result in domain_results:
                print(f"  {result}")
    else:
        results = evaluator.evaluate(
            domain=args.domain,
            dataset=args.dataset,
            max_samples=args.max_samples,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for result in results:
            print(f"\n{result}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()