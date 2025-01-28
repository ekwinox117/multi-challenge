import argparse
from dotenv import load_dotenv
import os
import json
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.result_parser import ResultParser
from src.models.factory import ModelFactory

def parse_provider_args(provider_args):
    """Parse key-value pairs from --provider-args."""
    args_dict = {}
    if provider_args:
        for arg in provider_args:
            key, value = arg.split('=')
            args_dict[key] = value
    return args_dict

def main():
    load_dotenv(dotenv_path="./.env",override=True)

    parser = argparse.ArgumentParser(description="Run LLM benchmark for conversational ability.")
    
    parser.add_argument('--output-file', type=str, required=True,
                        help="Path to save the final evaluation stats and scores.")
    parser.add_argument('--responses-file', type=str,
                        help="Path to the JSONL file containing model responses.")
    parser.add_argument('--model-provider', type=str,
                        help="Specify the model provider for generating responses.")
    parser.add_argument('--provider-args', type=str, nargs='*',
                        help="Provider-specific arguments in key=value format.")
    parser.add_argument('--attempts', type=int, default=1,
                        help="Number of attempts to generate for each conversation")
    parser.add_argument('--max-workers_response_gen', type=int, default=1,
                        help="Number of parallel workers to use for response generation.")
    parser.add_argument('--max-workers_eval', type=int, default=1,
                        help="Number of parallel workers to use for evaluation.")
    parser.add_argument('--raw', type=str,
                        help="Path to save detailed raw output including all responses and evaluations")

    args = parser.parse_args()

    # Validate the --raw argument
    if args.raw:
        if not args.raw.lower().endswith('.csv'):
            parser.error("The --raw argument must specify a file with a .csv extension")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(args.raw), exist_ok=True)

    input_file = './data/benchmark_questions.jsonl'

    data_loader = DataLoader(input_file)
    data_loader.load_data()

    if args.responses_file:
        data_loader.load_responses(args.responses_file)
    else:
        if not args.model_provider:
            raise ValueError("You must specify a --model-provider if generating responses.")
        
        provider_args = parse_provider_args(args.provider_args)
        model_provider = ModelFactory.get_provider(args.model_provider, **provider_args)
        data_loader.generate_responses(model_provider, attempts=args.attempts, max_workers = args.max_workers_response_gen)
    
    responses = data_loader.get_responses()
    conversations = data_loader.get_conversations()

    evaluator = Evaluator(conversations, responses)
    evaluation_results = evaluator.evaluate(max_workers=args.max_workers_eval)

    parser = ResultParser(evaluation_results)
    scores = parser.calculate_scores()

    # Save summary scores
    with open(args.output_file, 'w') as f:
        f.write(f"Attempts: {args.attempts} Overall Score: {scores['overall_score']:.2f}%\n")
        f.write("\nAxis Scores:\n")
        for axis, score in scores['axis_scores'].items():
            f.write(f"{axis}: {score:.2f}%\n")

    # Save detailed raw output if requested
    if args.raw:
        parser.save_raw_output(
            output_file=args.raw,
            conversations=conversations,
            responses=responses,
            attempts=args.attempts
        )

    print(f"Evaluation complete. Results saved to {args.output_file}")
    if args.raw:
        print(f"Detailed raw output saved to {args.raw}")

if __name__ == '__main__':
    main()