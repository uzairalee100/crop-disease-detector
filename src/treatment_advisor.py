import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads .env file automatically

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

def format_disease_name(raw_name: str) -> tuple:
    """
    Convert raw class name to readable format.
    Example: Apple___Black_rot → ('Apple', 'Black Rot')
    """
    parts  = raw_name.replace('___', '|').replace('_', ' ').split('|')
    crop    = parts[0].strip() if len(parts) > 0 else 'Unknown'
    disease = parts[1].strip() if len(parts) > 1 else 'Unknown'
    return crop, disease

def get_treatment_english(crop: str, disease: str) -> dict:
    """Get treatment advice in English from Groq."""
    prompt = f"""
You are an expert agricultural scientist.
A farmer's {crop} crop has been diagnosed with: {disease}

Provide a structured treatment report with EXACTLY these sections:

DISEASE OVERVIEW:
(2-3 sentences about what this disease is)

SYMPTOMS TO LOOK FOR:
(3-4 bullet points of symptoms)

IMMEDIATE TREATMENT STEPS:
(4-5 specific actionable steps the farmer should do TODAY)

RECOMMENDED PRODUCTS:
(2-3 specific fungicides or pesticides with dosage)

PREVENTION FOR NEXT SEASON:
(3-4 prevention tips)

SEVERITY LEVEL:
(One word only: Low / Medium / High / Critical)

Keep language simple and practical for farmers.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )
    advice = response.choices[0].message.content

    # Extract severity from response
    severity = "Medium"
    for line in advice.split('\n'):
        if 'SEVERITY LEVEL:' in line:
            line_lower = line.lower()
            if 'critical' in line_lower: severity = "Critical"
            elif 'high'    in line_lower: severity = "High"
            elif 'low'     in line_lower: severity = "Low"
            else:                         severity = "Medium"

    return {"advice": advice, "severity": severity}

def get_treatment_urdu(crop: str, disease: str) -> dict:
    """Get treatment advice in Urdu from Groq."""
    prompt = f"""
آپ ایک ماہر زرعی سائنسدان ہیں۔
ایک کسان کی {crop} فصل میں {disease} بیماری تشخیص ہوئی ہے۔

براہ کرم اردو میں مندرجہ ذیل معلومات فراہم کریں:

بیماری کا تعارف:
(2-3 جملے)

علامات:
(3-4 نکات)

فوری علاج:
(4-5 عملی اقدامات)

تجویز کردہ ادویات:
(2-3 مخصوص کیڑے مار ادویات)

اگلے موسم کے لیے احتیاط:
(3-4 نکات)

سادہ اور عملی زبان استعمال کریں جو کسان آسانی سے سمجھ سکے۔
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.3
    )
    return {"advice": response.choices[0].message.content,
            "severity": "Medium"}

def get_advice(disease_class: str, language: str = "English") -> dict:
    """
    Main function — takes raw disease class name,
    returns formatted advice dict.
    """
    crop, disease = format_disease_name(disease_class)

    # Handle healthy crops
    if disease.lower() == 'healthy':
        if language == "Urdu":
            advice_text = f"مبارک ہو! آپ کی {crop} فصل بالکل صحت مند ہے۔ کوئی بیماری نہیں پائی گئی۔ باقاعدہ نگرانی جاری رکھیں۔"
        else:
            advice_text = f"Great news! Your {crop} crop appears healthy. No disease detected. Continue regular monitoring and maintain good agricultural practices."
        return {
            "crop":     crop,
            "disease":  "Healthy",
            "advice":   advice_text,
            "severity": "None",
            "language": language
        }

    # Get treatment advice
    if language == "Urdu":
        result = get_treatment_urdu(crop, disease)
    else:
        result = get_treatment_english(crop, disease)

    return {
        "crop":     crop,
        "disease":  disease,
        "advice":   result["advice"],
        "severity": result["severity"],
        "language": language
    }

if __name__ == "__main__":
    print("Testing Treatment Advisor...")
    print("=" * 50)

    # Test English
    result = get_advice("Apple___Black_rot", "English")
    print(f"Crop:     {result['crop']}")
    print(f"Disease:  {result['disease']}")
    print(f"Severity: {result['severity']}")
    print(f"\nAdvice:\n{result['advice']}")
    print("=" * 50)
