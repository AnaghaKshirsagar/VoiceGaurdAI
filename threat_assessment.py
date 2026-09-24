def calculate_threat_assessment(
    fake_score,
    fraud_risk_level,
    fraud_points
):
    """
    Combine deepfake detection and fraud-analysis signals
    into a single Voice Guard threat assessment.
    """

    threat_points = 0

    # Deepfake signal
    if fake_score >= 0.70:
        threat_points += 5
    elif fake_score >= 0.50:
        threat_points += 3

    # Fraud signal
    if fraud_risk_level == "HIGH":
        threat_points += 5
    elif fraud_risk_level == "MEDIUM":
        threat_points += 3
    else:
        threat_points += 0

    # Additional fraud intensity
    if fraud_points >= 8:
        threat_points += 2
    elif fraud_points >= 4:
        threat_points += 1

    # Final assessment
    if threat_points >= 8:
        assessment = "HIGH RISK"
    elif threat_points >= 4:
        assessment = "MEDIUM RISK"
    else:
        assessment = "LOW RISK"

    return {
        "assessment": assessment,
        "threat_points": threat_points
    }