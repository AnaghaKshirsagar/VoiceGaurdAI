import re


FRAUD_PATTERNS = {
    "financial_request": [
        "send money",
        "transfer money",
        "transfer",
        "payment",
        "pay me",
        "send me",
        "bank account",
        "account number",
        "upi",
        "upi id",
        "money",
        "cash",
    ],

    "credential_request": [
        "otp",
        "one time password",
        "password",
        "pin",
        "cvv",
        "verification code",
        "security code",
    ],

    "urgency": [
        "urgent",
        "urgently",
        "immediately",
        "right now",
        "quickly",
        "hurry",
        "as soon as possible",
        "don't wait",
    ],

    "impersonation": [
        "this is your son",
        "this is your daughter",
        "this is your friend",
        "this is your brother",
        "this is your sister",
        "i'm stuck",
        "i am stuck",
        "don't tell anyone",
        "keep this secret",
    ],
}


CATEGORY_WEIGHTS = {
    "financial_request": 3,
    "credential_request": 4,
    "urgency": 2,
    "impersonation": 3,
}


def normalize_text(text):
    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def analyze_fraud_signals(text):

    normalized_text = normalize_text(text)

    detected_signals = []

    total_points = 0

    for category, patterns in FRAUD_PATTERNS.items():

        matches = []

        for pattern in patterns:

            if pattern in normalized_text:
                matches.append(pattern)

        if matches:

            detected_signals.append({
                "category": category,
                "matches": matches,
                "points": CATEGORY_WEIGHTS[category]
            })

            total_points += CATEGORY_WEIGHTS[category]

    if total_points >= 8:
        risk_level = "HIGH"

    elif total_points >= 4:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    return {
        "risk_level": risk_level,
        "risk_points": total_points,
        "signals": detected_signals
    }