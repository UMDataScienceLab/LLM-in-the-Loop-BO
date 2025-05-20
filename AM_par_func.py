import numpy as np
from openai import OpenAI
import json

def _sample_one_candidate_AM(args):
    history_variant_str, target_score = args
    prompt = f"""
    The following are past evaluations of the stringing percentage and their corresponding Nozzle Temperature and Z Hop values:    
    {history_variant_str}

    You are allowed to adjust **only five slicing parameters**:
    1. **Nozzle Temperature**: Range 220–260°C (step: 1°C)
    2. **Z Hop Height**: Range 0.1–1.0 mm (step: 0.1 mm)
    3. **Coasting Volume**:	0.02–0.1 mm³ (step: 0.01 mm³)
    4. **Retraction Distance**: 1.0–10.0 mm (step: 1 mm)
    5. **Outer Wall Wipe Distance**: 0.0–1.0 mm (step: 0.1 mm)
    
    These slicing settings are fixed:
    - Retraction Speed = 60 mm/s
    - Travel Speed = 178 mm/s
    - Fan Speed = 60 %
    
    Other slicing settings are set to be the software's default values.
        
    Recommend a new ([Nozzle Temperature (°C), Z Hop Height (mm), Coasting Volume (mm³), Retraction Distance (mm), Outer Wall Wipe Distance (mm)) that can achieve the stringing percentage of {target_score}.
        
    **Instructions:**
    - Return only one 5D vector: `[Nozzle Temperature (°C), Z Hop Height (mm), Coasting Volume (mm³), Retraction Distance (mm), Outer Wall Wipe Distance (mm)]`
    - Ensure the values respect the allowed ranges and increments.
    - Respond with strictly valid **JSON format**.
    - Do **not** include any explanations, comments, or extra text. Do not include the word jason.
    """

    client = OpenAI()
    
    while True:
        try:
            messages = [
            {"role": "system", "content": "You are an AI assistant that helps me optimizing the 3D manufacturing process by controlling parameters."},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50
            ).choices[0].message.content.strip()
            print(response)
            extracted_value = json.loads(response)
            if isinstance(extracted_value, list) and len(extracted_value) == 5:
                extracted_value = [np.float64(v) for v in extracted_value]
                return tuple(extracted_value)

        except (ValueError, json.JSONDecodeError):
            continue


def _predict_llm_score_AM(args):
    x, history_variant_str = args
    prompt = f"""
    The following are past evaluations of the stringing percentage and the corresponding Nozzle Temperature and Z hop.    
    {history_variant_str}
    You are allowed to adjust **only five slicing parameters**:
    1. **Nozzle Temperature**: Range 220–260°C (step: 1°C)
    2. **Z Hop Height**: Range 0.1–1.0 mm (step: 0.1 mm)
    3. **Coasting Volume**:	0.02–0.1 mm³ (step: 0.01 mm³)
    4. **Retraction Distance**: 1.0–10.0 mm (step: 1 mm)
    5. **Outer Wall Wipe Distance**: 0.0–1.0 mm (step: 0.1 mm)
    
    All other slicing settings are fixed:
    - Retraction Speed = 60 mm/s
    - Travel Speed = 178 mm/s
    - Fan Speed = 60 %
    
    Predict the stringing percentage at ([Nozzle Temperature, Z Hop Height, Coasting Volume, Retraction Distance, Outer Wall Wipe Distance) = {x}.
    
    The stringing percentage needs to be a single value between 0 to 100. 
    Return only a single numerical value. Do not include any explanations, labels, formatting, percentage symbol, or extra text. 
    The response must be strictly a valid floating-point number.
    """

    client = OpenAI()
    while True:
        try:
            messages = [
            {"role": "system", "content": "You are an AI assistant that helps me optimizing the 3D manufacturing process by controlling parameters."},
            {"role": "user", "content": prompt}
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50
            ).choices[0].message.content.strip()
            print(response)
            return float(response), tuple(x)
        except ValueError:
            continue
