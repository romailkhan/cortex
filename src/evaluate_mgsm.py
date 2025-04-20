import json
from typing import List, Dict

from Cortex import Cortex
from CortexSingle import CortexSingle

def load_test_data(file_path: str) -> List[Dict]:
    """Load test cases from JSONL file."""
    test_cases = []
    with open(file_path, 'r') as f:
        for line in f:
            test_cases.append(json.loads(line.strip()))
    return test_cases



def evaluate_single_agent():
    cortex = CortexSingle()
    test_cases = load_test_data('src/data/mgsm/test.jsonl')
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        expected_answer = test_case['answer_number']
        
        response = cortex.process_query(question)
        
        try:
            predicted_answer_str = response['agents']['reasoning']['final_answer']
            
            predicted_answer = float(predicted_answer_str)
            expected_answer = float(expected_answer)
            
            is_correct = abs(predicted_answer - expected_answer) < 1e-6
            if is_correct:
                correct += 1
                
            print("--------------------------------")
            print(f"Question {i}: {question}")
            print("--------------------------------")
            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {is_correct}")
            print("--------------------------------")
            
        except (KeyError, ValueError, TypeError) as e:
            print("--------------------------------")
            print(f"Question {i}: {question}")
            print("--------------------------------")
            print(f"Expected: {expected_answer}")
            print(f"Error processing prediction: {e}")
            print(f"Raw Response: {response}")
            print("--------------------------------")
            continue
            
    accuracy = correct / total if total > 0 else 0
    print("\n--- Single Agent Evaluation Summary ---")
    print(f"Final Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")
    print("--------------------------------------")


def evaluate_mgsm():
    cortex = Cortex()
    
    test_cases = load_test_data('src/data/mgsm/test.jsonl')
    
    correct = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case['question']
        expected_answer = test_case['answer_number']
        
        response = cortex.process_query(question)
        
        print("--------------------------------")
        print(question)
        print("--------------------------------")
        print(expected_answer)
        print("--------------------------------")
        print(response)
        print("--------------------------------")
        try:
            predicted_answer = float(response['agents']['language']['analysis']['final_response'])
            if abs(predicted_answer - expected_answer) < 1e-6:
                correct += 1
                
            print(f"Question {i}:")
            print(f"Expected: {expected_answer}")
            print(f"Predicted: {predicted_answer}")
            print(f"Correct: {predicted_answer == expected_answer}\n")
            
        except (KeyError, ValueError) as e:
            print(f"Error processing question {i}: {e}\n")
            continue
    
    accuracy = correct / total
    print(f"\nFinal Accuracy: {accuracy:.2%}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")

if __name__ == "__main__":
    # evaluate_single_agent()
    evaluate_mgsm()