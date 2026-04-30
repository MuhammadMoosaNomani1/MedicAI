from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
session = requests.Session()

# -----------------------------
# 🧠 STRICT MEDICAL PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are a medical assistant for basic symptom checking.

RULES:
- Only common conditions (cold, flu, fever, headache, infection, dehydration, allergy)
- NEVER mention cancer, tumor, stroke unless emergency keyword is given
- Keep answers extremely short
- No explanations inside causes

FORMAT (STRICT):

Causes:
- 1
- 2
- 3
Symptoms:
- 1
- 2
- 3
Risk: Low / Medium / High
Emergency: Yes / No
Advice:
- short bullet 1
- short bullet 2
Doctor:
2 lines only
"""

# -----------------------------
# ⚡ SMART MODEL SELECTOR (OPTIMIZED)
# -----------------------------
def choose_model(text):
    length = len(text)

    # 🔥 FAST DEFAULT (best for your i3)
    if length < 120:
        return "phi"

    # 🔥 SLIGHTLY HARD CASES
    return "mistral"


# -----------------------------
# ⚡ AI CALL FUNCTION
# -----------------------------
def ask_model(prompt, model_name):
    try:
        response = session.post(
            OLLAMA_URL,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,

                "options": {
                    # 🔥 VERY IMPORTANT FOR CONSISTENCY
                    "temperature": 0.0,

                    # 🔥 LIMIT OUTPUT (prevents lag + hallucination)
                    "num_predict": 60,
                    "num_ctx": 768,
                    "num_thread": 2,

                    "top_k": 10,
                    "top_p": 0.7,
                    "repeat_penalty": 1.15
                },

                "keep_alive": "30m"
            },
            timeout=(2, 60)
        )

        response.raise_for_status()
        return response.json().get("response", "").strip()

    except requests.exceptions.Timeout:
        return "System busy. Try again."
    except Exception:
        return "AI server not responding."


# -----------------------------
# 🏠 HOME
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# 🧠 PROCESS
# -----------------------------
@app.route("/process", methods=["POST"])
def process():
    symptoms = request.form.get("text", "").strip().lower()

    if not symptoms:
        return jsonify({"result": "Please enter symptoms"})

    # 🚨 emergency bypass
    emergency_keywords = (
        "chest pain", "can't breathe", "shortness of breath",
        "fainting", "unconscious", "stroke", "severe bleeding"
    )

    if any(k in symptoms for k in emergency_keywords):
        return jsonify({
            "result": "🚨 EMERGENCY: Seek medical help immediately."
        })

    # 🔥 trim input for speed
    symptoms = symptoms[:120]

    model = choose_model(symptoms)

    prompt = SYSTEM_PROMPT + f"\nSymptoms: {symptoms}\n"

    result = ask_model(prompt, model)

    return jsonify({"result": result})


# -----------------------------
# 🚀 RUN
# -----------------------------
if __name__ == "__main__":
    app.run(
        debug=False,
        threaded=True,
        use_reloader=False
    )