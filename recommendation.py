# recommendation.py

def generate_recommendation(result, disease_info):

    plant = result["plant"]
    disease = result["disease"]
    confidence = result["confidence"]

    # Healthy Plant
    if disease.lower() == "healthy":

        return f"""
The {plant} plant appears healthy.

Confidence: {confidence:.2f}%

Recommendations:

• Continue regular monitoring.
• Maintain proper irrigation.
• Apply balanced fertilizers.
• Keep the field free from weeds.
• Inspect leaves every few days for early signs of disease.
"""

    # Diseased Plant
    recommendation = f"""
Disease Detected: {disease}

Confidence: {confidence:.2f}%

Description:
{disease_info.get("description", "No description available.")}

Recommended Actions:

"""

    # Organic Treatments
    organic = disease_info.get("organic", [])

    if organic:
        recommendation += "\nOrganic Treatment:\n"
        for item in organic:
            recommendation += f"• {item}\n"

    # Chemical Treatments
    chemical = disease_info.get("chemical", [])

    if chemical:
        recommendation += "\nChemical Treatment:\n"
        for item in chemical:
            recommendation += f"• {item}\n"

    # Prevention
    prevention = disease_info.get("prevention", [])

    if prevention:
        recommendation += "\nPrevention:\n"
        for item in prevention:
            recommendation += f"• {item}\n"

    recommendation += """
General Advice:

• Remove infected leaves immediately.
• Avoid excessive watering.
• Improve air circulation.
• Monitor nearby plants for similar symptoms.
"""

    return recommendation