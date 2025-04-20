from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import re

class Reasoning_Benchmark:
    def __init__(self):
        max_tokens = int(os.getenv("MAX_TOKENS", 4096))

        temperature = float(os.getenv("TEMPERATURE", 0.6))
        max_tokens = int(os.getenv("MAX_TOKENS", 128000))
        top_p = float(os.getenv("TOP_P", 0.95))

        self.llm = ChatGroq(
            model_name=os.getenv("MODEL"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={
                "top_p": top_p,
                "max_completion_tokens": max_tokens
            }
        )
        
        self.system_template = """
        You are a {topic} Reasoning Specialist. Your role is to apply logical analysis 
        and draw well-supported conclusions. Use the provided memories to help inform your analysis.

        Background: You excel at concise logical analysis, finding connections between ideas, and drawing
        evidence-based conclusions.

        Your goal is to apply logical analysis to the input and provide a final answer to the user's math problem.
        
        IMPORTANT: You must respond with ONLY the final answer to the user's query as a number without any units.
        IMPORTANT: DO not add any formatting to the response.
        IMPORTANT: Please reason step by step, and put your final answer within the following format:
        {{
            "final_answer": "final answer to the user's query"
        }}
        """

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            ("user", "{input}")
        ])

        self.chain = self.prompt | self.llm

    def analyze(self, input_text: str, memories: List[Dict], topic: str = "Expert Reasoning Specialist") -> Dict:
        """
        Apply logical analysis to the input text, informed by memories.
        
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
        
        # extract the final answer from the response. Return "final_answer" field with the value    
        final_answer_match = re.search(r'"final_answer":\s*"(.*?)"', response_text)
        if final_answer_match:
            # Return the desired dictionary format
            return {"final_answer": final_answer_match.group(1)}
        else:
            print(f"Error: No final answer found in the response: {response_text}")
            # Return an error dictionary maintaining the structure
            return {"final_answer": f"Error: Parsing failed. Response: {response_text}"}