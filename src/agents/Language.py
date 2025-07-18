from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import json

class Language:
    def __init__(self):
        self.llm = ChatGroq(
            model_name=os.getenv("MODEL"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=os.getenv("TEMPERATURE"),
            max_tokens=os.getenv("MAX_TOKENS")
        )
        
        self.system_template = """
        You are a {topic} Language Specialist. Your role is to analyze language usage 
        and communication effectiveness. Use the provided memories to help inform your analysis.

        Background: You are an expert at understanding language patterns, meaning, and communication
        effectiveness across different contexts.

        Your goal is to analyze language usage and meaning.

        Focus on:
        - Key meanings
        - Language patterns
        - Communication style
        - Overall effectiveness

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Language Specialist",
            "analysis": {{
                "semantic_interpretations": ["interpretation1", "interpretation2"],
                "stylistic_patterns": ["pattern1", "pattern2"],
                "final_response": "final response to the user's query"
            }}
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.
        
        Finally, you must produce a final response to the user's query. If it is a math problem, you must solve it and provide the answer. If it is a code problem, you must solve it and provide the answer. If it is a general question, you must answer it based on the information provided.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, memories: List[Dict], topic: str = "General") -> Dict:
        """
        Analyze the language usage in the input text, informed by memories.
        
        Args:
            input_text (str): The text to analyze
            memories (List[Dict]): Relevant past memories.
            topic (str): The specific topic area for analysis (defaults to "General")
            
        Returns:
            Dict: Analysis results in the specified JSON format
        """
        response = self.chain.invoke({
            "input": input_text,
            "memories": memories,
            "topic": topic
        })
        
        response_text = str(response.content)
        
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            json_str = response_text[start_idx:end_idx]
            
            json_str = json_str.strip()
            json_str = ' '.join(line.strip() for line in json_str.splitlines())
            
            result = json.loads(json_str)
            if "role" not in result or "analysis" not in result:
                raise ValueError("Missing required top-level fields")
                
            required_analysis_fields = ["semantic_interpretations", "stylistic_patterns"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Language Specialist",
                "analysis": {
                    "semantic_interpretations": ["Error in analysis"],
                    "stylistic_patterns": ["Unable to parse response"]
                }
            } 