import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import gradio as gr
from pydantic import BaseModel

# Load environment variables
def load_env():
    load_dotenv(override=True)

# Read profile from PDF
def get_profile(pdf_path):
    reader = PdfReader(pdf_path)
    profile = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            profile += text
    return profile

# Read summary from text file
def get_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
        return f.read()

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

def main():
    load_env()
    name = "Supreet Arora"
    profile = get_profile("me/resume.pdf")
    summary = get_summary("me/summary.txt")
    gemini = OpenAI(
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."
    system_prompt += f"\n\n## Summary:\n{summary}\n\n##  Profile:\n{profile}\n\n"
    system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

    evaluator_system_prompt = f"You are an evaluator that decides whether a response to a question is acceptable. \
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality. \
The Agent is playing the role of {name} and is representing {name} on their website. \
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website. \
The Agent has been provided with context on {name} in the form of their summary and profile details. Here's the information:"
    evaluator_system_prompt += f"\n\n## Summary:\n{summary}\n\n## profile Profile:\n{profile}\n\n"
    evaluator_system_prompt += f"With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."

    def evaluator_user_prompt(reply, message, history):
        user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
        user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
        user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
        user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
        return user_prompt

    def evaluate(reply, message, history) -> Evaluation:
        messages = [{"role": "system", "content": evaluator_system_prompt}] + [{"role": "user", "content": evaluator_user_prompt(reply, message, history)}]
        response = gemini.beta.chat.completions.parse(model="gemini-2.0-flash", messages=messages, response_format=Evaluation)
        return response.choices[0].message.parsed

    def rerun(reply, message, history, feedback):
        updated_system_prompt = system_prompt + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
        updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
        updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
        messages = [{"role": "system", "content": updated_system_prompt}] + history + [{"role": "user", "content": message}]
        response = gemini.chat.completions.create(model="gemini-2.0-flash", messages=messages)
        return response.choices[0].message.content

    def chat(message, history):
        if "patent" in message:
            system = system_prompt + "\n\nEverything in your reply needs to be in pig latin -               it is mandatory that you respond only and entirely in pig latin"
        else:
            system = system_prompt
        messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
        response = gemini.chat.completions.create(model="gemini-2.0-flash", messages=messages)
        reply = response.choices[0].message.content
        evaluation = evaluate(reply, message, history)
        if evaluation.is_acceptable:
            print("Passed evaluation - returning reply")
        else:
            print("Failed evaluation - retrying")
            print(evaluation.feedback)
            reply = rerun(reply, message, history, evaluation.feedback)
        return reply

    gr.ChatInterface(chat, type="messages").launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
