from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import json

class Reasoning:
    def __init__(self):
        self.llm = ChatGroq(
            model_name=os.getenv("MODEL"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=os.getenv("TEMPERATURE"),
            max_tokens=os.getenv("MAX_TOKENS")
        )
        
        self.system_template = """
        You are a {topic} Reasoning Specialist. Your role is to apply logical analysis 
        and draw well-supported conclusions.

        Background: You excel at logical analysis, finding connections between ideas, and drawing
        evidence-based conclusions.

        Your goal is to apply logical analysis to the input.

        Focus on:
        - Cause-effect relationships
        - Logical connections
        - Evidence evaluation
        - Conclusions

        IMPORTANT: You must respond with ONLY valid JSON. Your entire response must be a single JSON object with exactly this structure:
        {{
            "role": "Reasoning Specialist",
            "analysis": {{
                "logical_connections": ["connection1", "connection2"],
                "conclusions": ["conclusion1", "conclusion2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
        }}
        Do not include any other text, thoughts, or explanations. The response must be pure JSON only.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, topic: str = "General") -> Dict:
        """
        Apply logical analysis to the input text.
        
        Args:
            input_text (str): The text to analyze
            topic (str): The specific topic area for analysis (defaults to "General")
            
        Returns:
            Dict: Analysis results in the specified JSON format
        """
        response = self.chain.invoke({
            "input": input_text,
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
                
            required_analysis_fields = ["logical_connections", "conclusions", "recommendations"]
            if not all(field in result["analysis"] for field in required_analysis_fields):
                raise ValueError("Missing required analysis fields")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            print(f"Attempted to parse: {json_str}")
            return {
                "role": "Reasoning Specialist",
                "analysis": {
                    "logical_connections": ["Error in analysis"],
                    "conclusions": ["Unable to parse response"],
                    "recommendations": ["Please try again"]
                }
            } 