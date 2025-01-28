from collections import defaultdict
import json
from typing import List, Dict
import textwrap
import csv

class ResultParser:
    def __init__(self, evaluation_results):
        self.evaluation_results = evaluation_results

    def calculate_scores(self):
        # Group results by question_id
        question_results = defaultdict(list)
        for result in self.evaluation_results:
            question_results[result['question_id']].append(result['passed'])  # Use 'passed' instead of checking verdict

        # Count questions that pass in at least one attempt
        total_passed = sum(1 for passes in question_results.values() if any(passes))
        total_questions = len(question_results)
        overall_score = (total_passed / total_questions) * 100 if total_questions > 0 else 0

        # Initialize axis_counts with a set for tracking counted questions
        axis_counts = defaultdict(lambda: {'passed': 0, 'total': 0, 'counted': set()})
        
        for result in self.evaluation_results:
            question_id = result['question_id']
            axis = result['axis']
            
            # Only count each question once per axis
            if question_id not in axis_counts[axis]['counted']:
                axis_counts[axis]['total'] += 1
                axis_counts[axis]['counted'].add(question_id)
                
                # If any attempt for this question passed, count it as passed
                if any(r['passed'] for r in self.evaluation_results 
                      if r['question_id'] == question_id and r['axis'] == axis):
                    axis_counts[axis]['passed'] += 1

        return {
            "overall_score": overall_score,
            "axis_scores": {
                axis: (scores['passed'] / scores['total']) * 100 
                for axis, scores in axis_counts.items()
            }
        }
    
    def save_raw_output(self, output_file: str, conversations: List, responses: Dict, attempts: int):
        """Save detailed raw output including all conversations, responses, and evaluations to a CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question_id', 'axis', 'original_conversation', 'target_question', 'pass_criteria', 
                        'attempt_number', 'model_response', 'judge_verdict', 'passed', 'reasoning', 'final_result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            # Group evaluation results by question_id
            results_by_question = defaultdict(list)
            for result in self.evaluation_results:
                results_by_question[result['question_id']].append(result)

            # Process each conversation
            for conv in conversations:
                question_id = conv.question_id
                conv_responses = responses.get(question_id, ['N/A'] * attempts)  # Default to 'N/A' if no responses
                conv_results = results_by_question.get(question_id, [])

                # Prepare the original conversation
                original_conversation = "\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in conv.conversation])

                passed_attempts = sum(1 for result in conv_results if result.get('passed', False))
                final_result = 'PASS' if passed_attempts > 0 else 'FAIL'

                for i in range(attempts):
                    result = conv_results[i] if i < len(conv_results) else {}
                    writer.writerow({
                        'question_id': question_id,
                        'axis': conv.axis,
                        'original_conversation': original_conversation,
                        'target_question': conv.target_question,
                        'pass_criteria': f"Response should receive a {conv.pass_criteria} verdict",
                        'attempt_number': i + 1,
                        'model_response': conv_responses[i] if i < len(conv_responses) else 'N/A',
                        'judge_verdict': result.get('verdict', 'N/A'),
                        'passed': 'PASSED' if result.get('passed', False) else 'FAILED',
                        'reasoning': result.get('reasoning', 'N/A'),
                        'final_result': f"{final_result} ({passed_attempts}/{attempts} attempts passed)"
                    })