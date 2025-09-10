import streamlit as st
import os
import requests

# --- Portfolio Data (can be edited) ---
ABOUT = """
Hi! I'm Alex Carter, an AI Engineer passionate about building intelligent systems that make a difference. 
I love working on NLP, computer vision, and generative AI projects. 
"""

SOCIALS = {
    "LinkedIn": "https://www.linkedin.com/in/alexcarter-ai",
    "GitHub": "https://github.com/alexcarter-ai",
    "Twitter": "https://twitter.com/alexcarter_ai"
}

SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "Streamlit", "PyTorch", "TensorFlow", "LangChain", "Prompt Engineering"
]

PROJECTS = [
    {
        "title": "AI-Powered Resume Analyzer",
        "description": "Built an NLP-based web app that analyzes resumes and provides feedback using LLMs.",
        "link": "https://github.com/alexcarter-ai/resume-analyzer"
    },
    {
        "title": "Image Captioning with Transformers",
        "description": "Developed a deep learning model to generate captions for images using Vision Transformers.",
        "link": "https://github.com/alexcarter-ai/image-captioning"
    },
    {
        "title": "Chatbot for Healthcare FAQs",
        "description": "Created a chatbot using LLMs to answer healthcare-related questions for a hospital website.",
        "link": "https://github.com/alexcarter-ai/healthcare-chatbot"
    }
]

EXPERIENCE = [
    {
        "role": "AI Engineer",
        "company": "DeepVision Labs",
        "years": "2022 - Present",
        "desc": "Leading a team to develop computer vision solutions for retail analytics. Built scalable ML pipelines and deployed models to production."
    },
    {
        "role": "Machine Learning Researcher",
        "company": "AI Innovations Inc.",
        "years": "2020 - 2022",
        "desc": "Researched and implemented NLP models for document understanding and summarization. Published 2 papers in top AI conferences."
    }
]

CONTACT = {
    "Email": "alex.carter.ai@gmail.com",
    "Phone": "+1 555-123-4567",
    "Location": "San Francisco, CA"
}

# --- Helper for Chatbot Context ---
def get_portfolio_context():
    context = f"""
ABOUT ME:
{ABOUT}

SOCIALS:
""" + "\n".join([f"{k}: {v}" for k, v in SOCIALS.items()]) + """

SKILLS:
""" + ", ".join(SKILLS) + """

PROJECTS:
""" + "\n".join([f"{p['title']}: {p['description']} (Repo: {p['link']})" for p in PROJECTS]) + """

EXPERIENCE:
""" + "\n".join([f"{e['role']} at {e['company']} ({e['years']}): {e['desc']}" for e in EXPERIENCE]) + """

CONTACT:
""" + "\n".join([f"{k}: {v}" for k, v in CONTACT.items()])
    return context

# --- Streamlit UI ---
st.set_page_config(page_title="Alex Carter - AI Portfolio", page_icon="🤖", layout="wide")

st.title("🤖 Alex Carter - AI Portfolio")

# About Section
st.header("About Me")
st.write(ABOUT)

# Socials
st.subheader("Socials")
cols = st.columns(len(SOCIALS))
for i, (name, link) in enumerate(SOCIALS.items()):
    with cols[i]:
        st.markdown(f"[{name}]({link})")

# Skills
st.header("Skills")
st.write(", ".join(SKILLS))

# Projects
st.header("AI Projects")
for proj in PROJECTS:
    st.subheader(proj["title"])
    st.write(proj["description"])
    st.markdown(f"[GitHub Repo]({proj['link']})")

# Experience
st.header("Work Experience")
for exp in EXPERIENCE:
    st.subheader(f"{exp['role']} at {exp['company']} ({exp['years']})")
    st.write(exp["desc"])

# Contact
st.header("Contact")
for k, v in CONTACT.items():
    st.write(f"**{k}:** {v}")

st.markdown("---")

# --- Chatbot Section ---
st.header("💬 Ask Me Anything (Chatbot)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

from groq import Groq

def call_groq_api(messages, groq_api_key):
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_completion_tokens=512,
        top_p=1,
        # stream=False,
        # stop=None
    )
    # The response structure is similar to OpenAI's, so we extract the content accordingly
    return completion.choices[0].message.content

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask me about my career, skills, or projects:")
        submitted = st.form_submit_button("Send")
        if submitted and user_input:
            # Build context for the LLM
            system_prompt = (
                "You are a helpful AI assistant that answers questions about Alex Carter's AI portfolio, "
                "career, skills, projects, and experience. Use the following information to answer:\n"
                + get_portfolio_context()
            )
            messages = [{"role": "system", "content": system_prompt}]
            for msg in st.session_state.chat_history:
                messages.append(msg)
            messages.append({"role": "user", "content": user_input})

            try:
                with st.spinner("Thinking..."):
                    response = call_groq_api(messages, groq_api_key)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AlexBot:** {msg['content']}")
else:
    st.info("Please enter your Groq API key to use the chatbot.")
