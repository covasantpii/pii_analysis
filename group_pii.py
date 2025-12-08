import json
from collections import defaultdict
import re

INPUT_FILE = "data/output/Vamp0000006645_ocr_PII.json"  # CHANGE THIS
OUTPUT_STRUCTURED = "data/output/Grouped_PII.json"


def extract_last4(card):
    digits = re.sub(r"[^\d]", "", card)
    return digits[-4:] if len(digits) >= 4 else None


def group_records(pii_records):
    persons = defaultdict(lambda: {
        "name": None,
        "dob": None,
        "ssn": None,
        "dl_number": None,
        "phone": None,
        "email": None,
        "credit_card": None,
        "card_last4": None,
        "auth_code": None,
        "transaction_date": None,
        "other_dates": []
    })

    active_person = None

    for rec in pii_records:
        ptype = rec["pii_type"]
        val = rec["text"]

        # ---------------------------
        # When PERSON appears, switch context
        # ---------------------------
        if ptype == "PERSON":
            active_person = val
            persons[val]["name"] = val
            continue

        if active_person is None:
            # No associated person → skip or tag as GLOBAL
            active_person = "UNKNOWN"

        entry = persons[active_person]

        if ptype == "DOB":
            entry["dob"] = val

        elif ptype == "SSN":
            entry["ssn"] = val

        elif ptype == "DLN":
            entry["dl_number"] = val

        elif ptype == "EMAIL":
            entry["email"] = val

        elif ptype == "PHONE":
            entry["phone"] = val

        elif ptype == "AUTH_CODE":
            entry["auth_code"] = val

        elif ptype == "CREDIT_CARD":
            entry["credit_card"] = val
            entry["card_last4"] = extract_last4(val)

        elif ptype == "CARD_LAST4":
            entry["card_last4"] = extract_last4(val)

        elif ptype == "DATE":
            if entry["transaction_date"] is None:
                entry["transaction_date"] = val
            else:
                entry["other_dates"].append(val)

    return list(persons.values())


def main():
    pii_data = json.loads(open(INPUT_FILE, "r").read())
    grouped = group_records(pii_data)

    with open(OUTPUT_STRUCTURED, "w") as f:
        json.dump(grouped, f, indent=2)

    print("\n✔ Grouped PII written to", OUTPUT_STRUCTURED)


if __name__ == "__main__":
    main()
