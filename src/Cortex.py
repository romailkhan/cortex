from typing import Dict, Any
import json
from datetime import datetime
from agents.Perception import Perception
from agents.Emotion import Emotion
from agents.Reasoning import Reasoning
from agents.Language import Language
from agents.Feedback import Feedback

class Cortex:
    def __init__(self):
        # Initialize all agents
        self.perception_agent = Perception()
        self.emotion_agent = Emotion()
        self.reasoning_agent = Reasoning()
        self.language_agent = Language()
        self.feedback_agent = Feedback()
        self.reset_state()

    def reset_state(self):
        self.current_state = {
            "timestamp": datetime.now().isoformat(),
            "location": "Minneapolis, MN",
            "initial_query": None,
            "agents": {}
        }

    def perception(self, query: str) -> Dict:
        self.current_state["initial_query"] = query
        perception_output = self.perception_agent.analyze(query)
        self.current_state["agents"]["perception"] = perception_output
        return self.current_state

    def emotion(self, state: Dict) -> Dict:
        emotion_output = self.emotion_agent.analyze(
            json.dumps(state, indent=2)
        )
        state["agents"]["emotion"] = emotion_output
        return state

    def reasoning(self, state: Dict) -> Dict:
        reasoning_output = self.reasoning_agent.analyze(
            json.dumps(state, indent=2)
        )
        state["agents"]["reasoning"] = reasoning_output
        return state

    def language(self, state: Dict) -> Dict:
        language_output = self.language_agent.analyze(
            json.dumps(state, indent=2)
        )
        state["agents"]["language"] = language_output
        return state

    def feedback(self, state: Dict) -> Dict:
        feedback_output = self.feedback_agent.analyze(
            json.dumps(state, indent=2)
        )
        state["agents"]["feedback"] = feedback_output
        return state

    def process_query(self, query: str) -> Dict:
        self.reset_state()
        
        # Process through all agents sequentially
        state = self.perception(query)
        state = self.emotion(state)
        state = self.reasoning(state)
        state = self.language(state)
        state = self.feedback(state)

        self.save_state("cortex_output.json")
        return state

    def save_state(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.current_state, f, indent=2)