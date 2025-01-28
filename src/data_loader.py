import json
from typing import List, Dict
from src.conversation import Conversation
from src.models.base import ModelProvider
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataLoader:
    def __init__(self, input_file: str, response_file: str = None):
        self.input_file = input_file
        self.response_file = response_file
        self.conversations: List[Conversation] = []
        self.responses: Dict[int, List[str]] = {}  # Modified to store list of responses

    def load_data(self):
        """Loads input data and creates Conversation objects."""
        with open(self.input_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                conversation = Conversation(
                    question_id=data['QUESTION_ID'],
                    axis=data['AXIS'],
                    conversation=data['CONVERSATION'],
                    target_question=data['TARGET_QUESTION'],
                    pass_criteria=data['PASS_CRITERIA']
                )
                self.conversations.append(conversation)

    def load_responses(self, response_file):
        """Loads model responses from the provided file."""
        if response_file:
            with open(response_file, 'r') as f:
                self.responses = {
                    item['QUESTION_ID']: item['RESPONSE']  # Note: 'RESPONSE', not 'RESPONSES'
                    for item in (json.loads(line) for line in f)
                }
        return self.responses

    def generate_responses(self, model_provider: ModelProvider, attempts: int = 1, max_workers: int = 1) -> Dict[int, List[str]]:
        """Generate k responses for each conversation using the provided model provider in parallel."""

        def generate_conversation_responses(conversation):
            responses = []
            for _ in range(attempts):
                try:
                    response = model_provider.generate(conversation.conversation)
                    responses.append(response)
                except Exception as e:
                    print(f"Error generating response for question_id {conversation.question_id}: {str(e)}. Exception saved as response.")
                    responses.append(f"Error generating response for question_id {conversation.question_id}: {str(e)}.\n FAIL THIS QUESTION")
            return conversation.question_id, responses

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(generate_conversation_responses, conversation)
                for conversation in self.conversations
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating responses"):
                try:
                    question_id, responses = future.result()
                    self.responses[question_id] = responses
                except Exception as e:
                    print(f"Error processing future: {str(e)}")

        return self.responses

    def get_conversations(self) -> List[Conversation]:
        """Returns the list of Conversation objects."""
        return self.conversations
    
    def get_responses(self) -> Dict[int, List[str]]:
        """Returns the dictionary of responses."""
        return self.responses