"""Generate synthetic Italian PA requests dataset for the PoC."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from app.utils.config import settings
from app.utils.constants import CATEGORIES, CATEGORY_TO_OFFICE


RANDOM_SEED = 42

CATEGORY_WEIGHTS = {
    "Anagrafe": 0.24,
    "Tributi": 0.22,
    "Edilizia": 0.14,
    "Mobilita": 0.16,
    "Ambiente": 0.15,
    "Servizi Sociali": 0.09,
}

PRIORITY_DISTRIBUTION = {
    "low": 0.35,
    "medium": 0.4,
    "high": 0.2,
    "urgent": 0.05,
}

STATUS_CHOICES = ["resolved", "resolved", "resolved", "in_review"]


REQUEST_TEMPLATES: Dict[str, List[str]] = {
    "Anagrafe": [
        "Buongiorno, devo rinnovare la carta d'identita scaduta il mese scorso. Potete indicarmi i documenti necessari?",
        "Richiedo certificato di residenza storico per pratica universitaria. Quali sono tempi e costi?",
        "Ho bisogno della carta d'identita elettronica per mio figlio minorenne, come prenoto?",
        "Segnalo errore nei miei dati anagrafici presenti nel certificato scaricato online.",
    ],
    "Tributi": [
        "Chiedo chiarimenti sul calcolo IMU per seconda casa ereditata nel 2025.",
        "Ho ricevuto un avviso TARI ma l'immobile risulta non occupato da mesi.",
        "Domando rateizzazione del debito relativo a tributi comunali arretrati.",
        "Desidero sapere se ho diritto a esenzione parziale TARI per nucleo numeroso.",
    ],
    "Edilizia": [
        "Vorrei informazioni su SCIA per ristrutturazione interna senza modifiche strutturali.",
        "Richiedo stato della pratica permesso di costruire presentata a gennaio.",
        "Ho necessita di autorizzazione paesaggistica per installazione pergolato in giardino.",
        "Segnalo cantiere confinante con rumori oltre orario consentito e possibile abuso edilizio.",
    ],
    "Mobilita": [
        "Richiedo pass residenti per zona ZTL, targa nuova dopo cambio veicolo.",
        "Segnalo disservizio nella linea autobus 7 con ritardi costanti in fascia mattutina.",
        "Domando autorizzazione temporanea di sosta per trasloco in via centrale.",
        "Vorrei chiarimenti su rinnovo contrassegno parcheggio disabili in scadenza.",
    ],
    "Ambiente": [
        "Segnalo mancata raccolta rifiuti organici da due settimane nel mio quartiere.",
        "Richiedo ritiro ingombranti a domicilio per vecchi mobili e materasso.",
        "Denuncio presenza di rifiuti abbandonati vicino al parco comunale.",
        "Vorrei indicazioni per compostaggio domestico e eventuali riduzioni tariffarie.",
    ],
    "Servizi Sociali": [
        "Richiedo informazioni per contributo affitto rivolto a nuclei con ISEE basso.",
        "Domando accesso al servizio assistenza domiciliare per genitore non autosufficiente.",
        "Necessito di appuntamento con assistente sociale per situazione familiare urgente.",
        "Chiedo chiarimenti su bonus alimentare e documentazione da presentare.",
    ],
}

RESPONSE_TEMPLATES = [
    "Gentile cittadino/a, la Sua richiesta e stata presa in carico dall'ufficio competente. A seguito delle verifiche preliminari, Le confermiamo che ricevera riscontro formale entro {days} giorni lavorativi.",
    "Si comunica che l'istanza e stata registrata con protocollo interno e inoltrata al settore responsabile. Qualora fossero necessari ulteriori documenti, sara contattato/a tramite i recapiti indicati.",
    "In riferimento alla Sua segnalazione, l'Amministrazione ha avviato le opportune verifiche tecniche. L'esito dell'istruttoria Le sara trasmesso con comunicazione ufficiale.",
    "La informiamo che, sulla base degli elementi forniti, la pratica puo procedere in via ordinaria. Restiamo a disposizione per eventuali integrazioni documentali.",
]


def pick_weighted(distribution: Dict[str, float]) -> str:
    """Sample a label based on configured probabilities."""
    labels = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(labels, weights=weights, k=1)[0]


def inject_noise(text: str) -> str:
    """Create noisy variants to simulate real citizen input quality."""
    variants = [
        text,
        text.lower(),
        text.replace(" ", "  "),
        text.replace("?", ""),
        text + " grazie",
        text[: max(20, len(text) // 2)],
        "URGENTE " + text,
    ]
    return random.choice(variants)


def build_request_text(category: str) -> str:
    """Build request text with optional ambiguity and varying lengths."""
    base = random.choice(REQUEST_TEMPLATES[category])
    addendum_pool = [
        "Preciso che ho gia contattato il numero verde senza esito.",
        "Allego eventuale documentazione disponibile in formato PDF.",
        "La situazione sta causando disagio al mio nucleo familiare.",
        "Resto disponibile per sopralluogo o integrazione dati.",
        "Segnalo che il problema persiste da oltre 30 giorni.",
    ]

    if random.random() < 0.25:
        base = f"{base} {random.choice(addendum_pool)} {random.choice(addendum_pool)}"
    elif random.random() < 0.15:
        base = random.choice(["Aiuto", "Info", "Urgente", "Richiesta supporto", "Permesso?"])

    if random.random() < 0.12:
        # Ambiguous text crossing category boundaries.
        base += " Non so se devo rivolgermi a tributi o anagrafe, chiedo supporto per individuare l'ufficio corretto."

    if random.random() < 0.2:
        base = inject_noise(base)

    return base


def build_operator_response(priority: str) -> str:
    """Create formal but varied PA operator responses."""
    base = random.choice(RESPONSE_TEMPLATES)
    days = {"low": 10, "medium": 7, "high": 4, "urgent": 2}[priority]
    return base.format(days=days)


def generate_rows(num_rows: int) -> List[Dict[str, str]]:
    """Generate synthetic request records."""
    random.seed(RANDOM_SEED)
    rows: List[Dict[str, str]] = []
    start_date = datetime(2025, 1, 1)

    for idx in range(num_rows):
        category = pick_weighted(CATEGORY_WEIGHTS)
        office = CATEGORY_TO_OFFICE[category]
        priority = pick_weighted(PRIORITY_DISTRIBUTION)
        status = random.choice(STATUS_CHOICES)

        created_at = start_date + timedelta(days=random.randint(0, 480), hours=random.randint(8, 18))
        resolution_days = {"low": random.randint(6, 20), "medium": random.randint(4, 12), "high": random.randint(2, 7), "urgent": random.randint(1, 3)}[priority]
        resolved_at = created_at + timedelta(days=resolution_days)

        row = {
            "request_id": f"REQ-{idx + 1:05d}",
            "citizen_request_text": build_request_text(category),
            "category": category,
            "office": office,
            "priority": priority,
            "status": status,
            "operator_response": build_operator_response(priority),
            "created_at": created_at.isoformat(),
            "resolved_at": resolved_at.isoformat() if status == "resolved" else "",
        }
        rows.append(row)

    duplicate_count = max(20, int(num_rows * 0.05))
    for i in range(duplicate_count):
        duplicate = rows[random.randint(0, len(rows) - 1)].copy()
        duplicate["request_id"] = f"REQ-DUP-{i + 1:04d}"
        rows.append(duplicate)

    return rows


def main() -> None:
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic PA requests dataset.")
    parser.add_argument("--output", type=str, default=str(settings.dataset_path), help="Output CSV path")
    parser.add_argument("--rows", type=int, default=1500, help="Base number of rows before duplicates")
    args = parser.parse_args()

    rows = generate_rows(args.rows)
    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path} | rows={len(df)}")


if __name__ == "__main__":
    main()
