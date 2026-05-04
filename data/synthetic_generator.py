# data/synthetic_generator.py
#
# Generates a realistic synthetic dataset of employee profiles.
# Each profile contains a name, department, job title, and a
# block of text that simulates what you might scrape from a
# public LinkedIn summary or professional forum post.
#
# Why synthetic data?
# Using real people's data without consent raises serious ethical
# and legal issues. Synthetic profiles let us build and validate
# the full pipeline without touching any real personal data.

import random
import json
import os
from config import DEPARTMENTS
from dotenv import load_dotenv
load_dotenv()

try:
    from groq import Groq
    # Initialize Groq client. It automatically looks for the GROQ_API_KEY environment variable.
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except ImportError:
    print("[DataGen] WARNING: 'groq' library not found. Run 'pip install groq'. Falling back to static templates.")
    groq_client = None
except Exception as e:
    print(f"[DataGen] WARNING: Groq client failed to initialize: {e}. Falling back to static templates.")
    groq_client = None

# ------------------------------------------------------------------
# Text templates per personality archetype (Used as Fallback)
# ------------------------------------------------------------------
PROFILE_TEMPLATES = {
    "methodical": [
        "I have spent the past {years} years building structured, process-driven workflows "
        "in {domain}. I believe strongly in following established procedures and reporting "
        "lines. Compliance, accuracy, and accountability are the values I bring to every "
        "project. I always defer to leadership directives and make sure every action is "
        "properly documented and approved before execution.",

        "My professional approach is grounded in discipline and attention to detail. I work "
        "best within clearly defined frameworks, and I take instructions from senior management "
        "very seriously. In {domain}, I have consistently delivered results by following "
        "standard operating procedures to the letter.",
    ],
    "reactive": [
        "Working in {domain} means constantly adapting to last-minute changes and high-pressure "
        "deadlines. I have learned to act fast when situations escalate. If something is urgent "
        "or critical, I prioritize it immediately without waiting for things to get worse. "
        "I know how quickly situations can deteriorate if you do not respond in time.",

        "Over {years} years in {domain}, I have handled more crisis situations than I can count. "
        "My instinct is to respond immediately when I get a high-priority message. Hesitation "
        "in critical moments has real consequences, and I have always taken that seriously.",
    ],
    "social": [
        "I am a people-first professional. In my {years} years in {domain}, everything I have "
        "accomplished has been through collaboration and strong relationships. I genuinely enjoy "
        "helping colleagues and making sure everyone on the team feels supported. When people "
        "I trust recommend something, I take it seriously.",

        "Building community is at the heart of what I do in {domain}. I care deeply about "
        "what my peers and mentors think, and I always try to give back to the people who "
        "have helped me. I believe that trust is the foundation of all professional relationships.",
    ],
    "curious": [
        "I have always been drawn to new ideas and cross-disciplinary thinking. My work in "
        "{domain} has taken me into some unexpected but fascinating territories. I enjoy "
        "engaging with new concepts and perspectives, especially when they challenge my "
        "existing assumptions. I am always willing to explore unconventional approaches.",

        "Intellectual curiosity drives most of what I do professionally. In {domain}, I "
        "actively seek out new tools, methodologies, and thought leaders. I find it easy "
        "to build rapport with people from very different backgrounds, and I value generosity "
        "of knowledge above almost everything else.",
    ],
}

DOMAINS = {
    "Finance":      ["financial risk", "auditing", "budget management", "regulatory compliance"],
    "HR":           ["talent acquisition", "employee relations", "organizational culture", "workforce planning"],
    "Engineering":  ["software development", "systems architecture", "DevOps", "cloud infrastructure"],
    "Legal":        ["contract review", "data privacy law", "corporate compliance", "regulatory affairs"],
    "Sales":        ["enterprise sales", "client relationships", "revenue growth", "account management"],
    "Operations":   ["supply chain", "process optimization", "vendor management", "logistics"],
}

JOB_TITLES = {
    "Finance":      ["Financial Analyst", "Senior Accountant", "CFO", "Budget Manager"],
    "HR":           ["HR Business Partner", "Talent Acquisition Lead", "People Operations Manager"],
    "Engineering":  ["Software Engineer", "DevOps Engineer", "Solutions Architect", "Tech Lead"],
    "Legal":        ["Legal Counsel", "Compliance Officer", "Data Protection Lead"],
    "Sales":        ["Account Executive", "Sales Manager", "Business Development Lead"],
    "Operations":   ["Operations Manager", "Supply Chain Analyst", "Process Engineer"],
}

FIRST_NAMES = [
    "Arjun", "Priya", "Rahul", "Sneha", "Amit", "Kavya",
    "Rohan", "Meera", "Vivek", "Ananya", "Nikhil", "Divya",
    "Karan", "Shreya", "Aditya", "Pooja", "Ravi", "Neha",
    "Suresh", "Anjali", "Deepak", "Swati", "Manish", "Ritu",
]

LAST_NAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Mehta", "Gupta",
    "Joshi", "Nair", "Reddy", "Iyer", "Verma", "Shah",
    "Mishra", "Rao", "Pillai", "Bose", "Das", "Pandey",
]

def generate_dynamic_bio(archetype: str, title: str, domain: str, years: int) -> str:
    """Uses Groq LLM to generate highly varied, nuanced professional bios."""
    if not groq_client:
        return random.choice(PROFILE_TEMPLATES[archetype]).format(years=years, domain=domain)

    prompt = f"""
    Write a realistic, 3 to 4 sentence professional LinkedIn summary for a {title} with {years} years of experience working in {domain}. 
    Their underlying personality archetype is '{archetype}'. 
    Make the language subtle, natural, and professional. Do NOT explicitly use the word '{archetype}'.
    Do not use introductory phrases like "Here is the summary". Only output the summary text itself.
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8
        )
        return response.choices[0].message.content.strip().replace('"', '')
    except Exception as e:
        print(f"[DataGen] Groq generation failed, falling back to templates: {e}")
        return random.choice(PROFILE_TEMPLATES[archetype]).format(years=years, domain=domain)


def generate_profile(employee_id: int, department: str = None) -> dict:
    """
    Generates a single synthetic employee profile.
    """
    if department is None:
        department = random.choice(DEPARTMENTS)

    archetype = random.choice(list(PROFILE_TEMPLATES.keys()))
    domain = random.choice(DOMAINS[department])
    years = random.randint(3, 18)
    title = random.choice(JOB_TITLES[department])
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

    # Generate the bio dynamically using Groq (or fallback template)
    text = generate_dynamic_bio(archetype, title, domain, years)

    return {
        "id": f"EMP-{str(employee_id).zfill(3)}",
        "name": name,
        "department": department,
        "title": title,
        "archetype": archetype,    # ground truth — not used in inference
        "text": text,
    }


def generate_organization(num_employees: int = 30, save_path: str = None) -> list:
    """
    Generates a full synthetic organization.
    """
    profiles = []
    employees_per_dept = num_employees // len(DEPARTMENTS)
    remainder = num_employees % len(DEPARTMENTS)
    emp_id = 1

    for dept in DEPARTMENTS:
        count = employees_per_dept + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
        for _ in range(count):
            profiles.append(generate_profile(emp_id, department=dept))
            emp_id += 1

    random.shuffle(profiles)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(profiles, f, indent=2)
        print(f"[DataGen] Saved {len(profiles)} profiles to {save_path}")

    return profiles


if __name__ == "__main__":
    print("[DataGen] Testing profile generation (with Groq if configured)...")
    profiles = generate_organization(num_employees=3)
    for p in profiles:
        print(f"\n{p['id']} | {p['name']} | {p['department']} | {p['title']}")
        print(f"Archetype: {p['archetype']}")
        print(f"Text: {p['text']}")