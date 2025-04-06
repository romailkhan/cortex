from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from agents.Perception import Perception
from agents.Emotion import Emotion
from agents.Reasoning import Reasoning
from agents.Language import Language
from agents.Feedback import Feedback
from Cortex import Cortex

cortex = Cortex()

response = cortex.process_query("A farmer has 17 sheep. All but 9 run away. Then, 3 of the remaining sheep are sold. How many sheep does the farmer have left?")
print(response)

# # Initialize agents
# perception_agent = Perception()
# emotion_agent = Emotion()
# reasoning_agent = Reasoning()
# language_agent = Language()
# feedback_agent = Feedback()

# # Example usage
# text = "AI technology has rapidly evolved over the past decade, leading to breakthrough applications in healthcare and education."

# # Get analyses from each agent
# perception_result = perception_agent.analyze(text)
# emotion_result = emotion_agent.analyze(text)
# reasoning_result = reasoning_agent.analyze(text)
# language_result = language_agent.analyze(text)

# # Get feedback on all analyses
# combined_analyses = f"""
# Perception Analysis: {perception_result}
# Emotion Analysis: {emotion_result}
# Reasoning Analysis: {reasoning_result}
# Language Analysis: {language_result}
# """
# feedback_result = feedback_agent.analyze(combined_analyses)

# print("Perception Analysis:", perception_result)
# print("--------------------------------")
# print("Emotion Analysis:", emotion_result)
# print("--------------------------------")
# print("Reasoning Analysis:", reasoning_result)
# print("--------------------------------")
# print("Language Analysis:", language_result)
# print("--------------------------------")
# print("Feedback:", feedback_result)
