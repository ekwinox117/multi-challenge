from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Any, Literal
from src.models.openai import OpenAIModel
from tqdm import tqdm

class JudgeResponse(BaseModel):
    reasoning: str
    verdict: Literal["YES", "NO"]

JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''

class Evaluator:
    def __init__(self, conversations: List[Any], responses: Dict[int, List[str]]):
        self.conversations = conversations
        self.responses = responses
        self.evaluation_model = OpenAIModel(
            model_path="gpt-4o-2024-08-06", 
            temp=0, 
            max_tokens=4096,
            response_format=JudgeResponse
        )
        self.results = []

    def evaluate_helper(self, i: int, conversation: Any, response: str) -> Tuple[int, str, str, str, str]:
        """Evaluate a single response."""
        target_question = conversation.target_question
        pass_criteria = conversation.pass_criteria
        prompt = JUDGE_PROMPT.format(response, target_question)
        judgement = self.evaluation_model.generate([{"role": "user", "content": prompt}])
        return i, conversation.axis, judgement.reasoning, judgement.verdict, pass_criteria

    def evaluate(self, max_workers:int = 1) -> List[Dict]:
        """Evaluate all responses for each conversation"""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, convo in enumerate(self.conversations):
                if convo.question_id not in self.responses:
                    # Handle missing question_id
                    self.results.append({
                        'question_id': convo.question_id,
                        'axis': convo.axis,
                        'attempt': 0,
                        'reasoning': 'NA - Question ID not found in responses',
                        'verdict': 'NO',
                        'pass_criteria': convo.pass_criteria,
                        'passed': False
                    })
                else:
                    for j, response in enumerate(self.responses[convo.question_id]):
                        futures.append(
                            executor.submit(self.evaluate_helper, i, convo, response)
                        )

            for future in tqdm(futures, desc="Evaluating responses", total=len(futures)):
                try:
                    i, axis, reasoning, verdict, pass_criteria = future.result()
                    self.results.append({
                        'question_id': self.conversations[i].question_id,
                        'axis': axis,
                        'attempt': j,
                        'reasoning': reasoning,
                        'verdict': verdict,
                        'pass_criteria': pass_criteria,
                        'passed': verdict == pass_criteria
                    })
                except Exception as e:
                    # Handle any other unexpected errors
                    self.results.append({
                        'question_id': self.conversations[i].question_id if i < len(self.conversations) else 'Unknown',
                        'axis': 'NA',
                        'attempt': 'NA',
                        'reasoning': f'Error during evaluation: {str(e)}',
                        'verdict': 'NO',
                        'pass_criteria': 'NA',
                        'passed': False
                    })

        # Calculate the final pass/fail status for each question
        question_results = {}
        for result in self.results:
            question_id = result['question_id']
            if question_id not in question_results:
                question_results[question_id] = {'attempts': 0, 'passes': 0}
            question_results[question_id]['attempts'] += 1
            if result['passed']:
                question_results[question_id]['passes'] += 1

        # Update results with final pass/fail status
        for result in self.results:
            question_id = result['question_id']
            attempts = question_results[question_id]['attempts']
            passes = question_results[question_id]['passes']
            result['final_status'] = f"{'PASS' if passes > 0 else 'FAIL'} ({passes}/{attempts} attempts passed)"

        return self.results