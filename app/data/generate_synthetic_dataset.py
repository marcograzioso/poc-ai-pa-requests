"""Generate synthetic Italian PA requests dataset for the PoC."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from app.utils.config import settings
from app.utils.constants import CATEGORIES, CATEGORY_TO_OFFICE


RANDOM_SEED = 42

CATEGORY_WEIGHTS = {
    "Anagrafe": 0.22,
    "Tributi": 0.21,
    "Edilizia": 0.14,
    "Mobilita": 0.15,
    "Ambiente": 0.14,
    "Servizi Sociali": 0.09,
    "Non Pertinente": 0.05,
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
        "Devo richiedere il cambio di residenza per trasferimento in altro quartiere, quali passaggi devo seguire?",
        "Chiedo appuntamento per rilascio certificato di stato di famiglia in bollo per uso notarile.",
        "Non riesco a ottenere lo SPID perche i miei dati anagrafici risultano non allineati.",
        "Richiedo estratto di nascita plurilingue per pratica estera e vorrei sapere se serve delega.",
        "Ho smarrito la carta d'identita e devo ottenere un duplicato con urgenza per viaggio imminente.",
    ],
    "Tributi": [
        "Chiedo chiarimenti sul calcolo IMU per seconda casa ereditata nel 2025.",
        "Ho ricevuto un avviso TARI ma l'immobile risulta non occupato da mesi.",
        "Domando rateizzazione del debito relativo a tributi comunali arretrati.",
        "Desidero sapere se ho diritto a esenzione parziale TARI per nucleo numeroso.",
        "Richiedo rettifica dell'avviso di pagamento per possibile duplicazione di importo.",
        "Vorrei attivare addebito automatico per tributi comunali e conoscere le modalita.",
        "Ho venduto un immobile e chiedo aggiornamento posizione tributaria dalla data del rogito.",
        "Segnalo mancata registrazione del pagamento TARI effettuato tramite pagoPA.",
        "Richiedo informazione su ravvedimento operoso per versamento IMU tardivo.",
    ],
    "Edilizia": [
        "Vorrei informazioni su SCIA per ristrutturazione interna senza modifiche strutturali.",
        "Richiedo stato della pratica permesso di costruire presentata a gennaio.",
        "Ho necessita di autorizzazione paesaggistica per installazione pergolato in giardino.",
        "Segnalo cantiere confinante con rumori oltre orario consentito e possibile abuso edilizio.",
        "Chiedo elenco completo documenti per CILA relativa a manutenzione straordinaria appartamento.",
        "Vorrei fissare un appuntamento tecnico per verifica conformita prima di presentare SCIA.",
        "Richiedo accesso agli atti edilizi dell'immobile acquistato per verifica precedenti autorizzazioni.",
        "Segnalo difformita tra progetto autorizzato e opere in corso nel cantiere vicino.",
        "Domando tempi medi di rilascio agibilita dopo fine lavori e collaudo impianti.",
    ],
    "Mobilita": [
        "Richiedo pass residenti per zona ZTL, targa nuova dopo cambio veicolo.",
        "Segnalo disservizio nella linea autobus 7 con ritardi costanti in fascia mattutina.",
        "Domando autorizzazione temporanea di sosta per trasloco in via centrale.",
        "Vorrei chiarimenti su rinnovo contrassegno parcheggio disabili in scadenza.",
        "Segnalo segnaletica orizzontale cancellata in prossimita della scuola primaria.",
        "Richiedo installazione stallo disabili in prossimita della mia abitazione.",
        "Vorrei sapere come presentare ricorso per sanzione ZTL notificata erroneamente.",
        "Domando aggiornamenti sul piano viabilita durante lavori stradali in centro storico.",
        "Chiedo autorizzazione per accesso occasionale in area pedonale per assistenza familiare.",
    ],
    "Ambiente": [
        "Segnalo mancata raccolta rifiuti organici da due settimane nel mio quartiere.",
        "Richiedo ritiro ingombranti a domicilio per vecchi mobili e materasso.",
        "Denuncio presenza di rifiuti abbandonati vicino al parco comunale.",
        "Vorrei indicazioni per compostaggio domestico e eventuali riduzioni tariffarie.",
        "Segnalo cassonetto danneggiato che impedisce il corretto conferimento dei rifiuti.",
        "Chiedo intervento di pulizia straordinaria per area verde con vetri e rifiuti sparsi.",
        "Vorrei informazioni su calendario raccolta porta a porta per nuova utenza domestica.",
        "Segnalo odori persistenti da impianto vicino e richiedo verifica ambientale.",
        "Domando modalita di conferimento RAEE per elettrodomestici non funzionanti.",
    ],
    "Servizi Sociali": [
        "Richiedo informazioni per contributo affitto rivolto a nuclei con ISEE basso.",
        "Domando accesso al servizio assistenza domiciliare per genitore non autosufficiente.",
        "Necessito di appuntamento con assistente sociale per situazione familiare urgente.",
        "Chiedo chiarimenti su bonus alimentare e documentazione da presentare.",
        "Richiedo valutazione per inserimento in graduatoria alloggio ERP per nucleo monogenitoriale.",
        "Vorrei sapere come attivare supporto educativo domiciliare per minore con disabilita.",
        "Domando accesso al contributo spese scolastiche per famiglia con ISEE sotto soglia.",
        "Chiedo indicazioni per servizio trasporto sociale verso strutture sanitarie.",
        "Segnalo situazione di fragilita economica improvvisa e necessito colloquio urgente.",
    ],
    "Non Pertinente": [
        "Volevo sapere i numeri vincenti del lotto di questa sera.",
        "Offro consulenza commerciale per aumentare vendite, contattatemi subito.",
        "Perche il mio smartphone si scarica velocemente?",
        "Messaggio promozionale: click qui per ricevere buoni regalo.",
        "Salve, questo non riguarda il comune ma un problema con il mio modem.",
        "Cerco consigli su dieta personalizzata per perdere peso prima dell'estate.",
        "Vendita online di prodotti cosmetici, scrivetemi per offerte esclusive.",
        "La mia stampante non si collega al wifi, potete aiutarmi?",
        "Vorrei prenotare una vacanza last minute, avete suggerimenti?",
        "Questo messaggio e una catena promozionale e non riguarda servizi pubblici.",
    ],
}

RESPONSE_SOLUTIONS: Dict[str, List[str]] = {
    "Anagrafe": [
        "Gentile cittadino/a, per completare la pratica anagrafica e sufficiente prenotare appuntamento tramite portale comunale o URP. Porti documento di identita valido, tessera sanitaria e modulo allegato. In caso di urgenza certificata, e disponibile slot prioritario entro {days} giorni lavorativi.",
        "La Sua richiesta e stata lavorata: abbiamo verificato i dati e aperto rettifica anagrafica con codice pratica dedicato. Ricevera conferma via email con istruzioni per ritiro o download del certificato aggiornato entro {days} giorni lavorativi.",
        "Per il rilascio richiesto abbiamo predisposto istruttoria semplificata. Le chiediamo solo integrazione fotografica conforme e pagamento dei diritti tramite pagoPA; a pagamento acquisito, il documento sara emesso entro {days} giorni lavorativi.",
    ],
    "Tributi": [
        "In merito alla posizione tributaria, l'ufficio ha avviato ricalcolo puntuale sulla base dei dati catastali comunicati. Se emergera eccedenza, verra emesso provvedimento di sgravio o rimborso; in caso di debito residuo Le invieremo piano rateizzato entro {days} giorni lavorativi.",
        "Abbiamo registrato la richiesta e aperto verifica su pagamento/avviso contestato. Le suggeriamo di allegare ricevuta pagoPA e documento immobile per accelerare istruttoria; riscontro tecnico previsto entro {days} giorni lavorativi.",
        "Per la Sua istanza e stata avviata procedura di regolarizzazione guidata con eventuale ravvedimento. Ricevera prospetto importi aggiornato e istruzioni operative per il versamento entro {days} giorni lavorativi.",
    ],
    "Edilizia": [
        "La pratica edilizia puo proseguire con iter ordinario: abbiamo inserito controllo preliminare documentale e Le invieremo check-list personalizzata (elaborati tecnici, asseverazioni, diritti) entro {days} giorni lavorativi.",
        "Per la segnalazione su cantiere, e stato pianificato sopralluogo con nucleo tecnico comunale. Gli esiti e le eventuali prescrizioni verranno notificati formalmente entro {days} giorni lavorativi.",
        "In riferimento alla richiesta autorizzativa, l'ufficio ha aperto istruttoria e calendarizzato verifica conformita urbanistica. Se non emergeranno criticita, Le trasmetteremo provvedimento o richiesta integrazione entro {days} giorni lavorativi.",
    ],
    "Mobilita": [
        "Per la Sua richiesta di mobilita, abbiamo attivato il procedimento amministrativo e predisposto verifica dei requisiti (residenza, targa, titoli autorizzativi). L'esito con eventuale rilascio permesso sara disponibile entro {days} giorni lavorativi.",
        "La segnalazione su viabilita/trasporto e stata inoltrata al gestore competente con livello di priorita operativo. E previsto aggiornamento intervento o risposta motivata entro {days} giorni lavorativi.",
        "Per l'autorizzazione temporanea richiesta Le invieremo modulo precompilato e istruzioni per pagamento diritti. A documentazione completa, il nulla osta sara emesso entro {days} giorni lavorativi.",
    ],
    "Ambiente": [
        "Abbiamo aperto ordine di servizio per il gestore igiene urbana con intervento programmato nell'area segnalata. Ricevera conferma di esecuzione o motivazione tecnica entro {days} giorni lavorativi.",
        "Per la richiesta di ritiro/conferimento, Le invieremo calendario disponibile e modalita operative. Una volta confermata la prenotazione, l'intervento sara effettuato nella prima finestra utile, con riscontro entro {days} giorni lavorativi.",
        "La segnalazione ambientale e stata trasmessa al nucleo di controllo territoriale per verifica puntuale. Eventuali azioni correttive e relative tempistiche Le saranno comunicate entro {days} giorni lavorativi.",
    ],
    "Servizi Sociali": [
        "La Sua istanza sociale e stata presa in carico da assistente di riferimento. Entro {days} giorni lavorativi ricevera convocazione per colloquio e indicazione della documentazione necessaria alla valutazione.",
        "Per il beneficio richiesto e stata avviata verifica requisiti ISEE e composizione nucleo. All'esito istruttorio Le comunicheremo ammissione, eventuale graduatoria e tempi di erogazione entro {days} giorni lavorativi.",
        "Abbiamo classificato la pratica con priorita di tutela e attivato coordinamento con servizi territoriali. Le verra trasmesso piano di supporto iniziale o richiesta integrazione entro {days} giorni lavorativi.",
    ],
}


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


def build_operator_response(priority: str, category: str) -> str:
    """Create formal but varied PA operator responses."""
    if category == "Non Pertinente":
        return (
            "Gentile cittadino/a, la comunicazione ricevuta non risulta pertinente ai servizi "
            "erogati da questo Ente oppure presenta caratteristiche riconducibili a contenuti non "
            "istituzionali. La invitiamo a inoltrare una richiesta attinente ai procedimenti comunali."
        )

    base = random.choice(RESPONSE_SOLUTIONS.get(category, RESPONSE_SOLUTIONS["Anagrafe"]))
    days = {"low": 10, "medium": 7, "high": 4, "urgent": 2}[priority]
    return base.format(days=days)


def ensure_non_pertinente_presence(rows: List[Dict[str, str]], start_index: int) -> List[Dict[str, str]]:
    """Guarantee that dataset always contains non-pertinent/spam examples."""
    current_count = sum(1 for row in rows if row.get("category") == "Non Pertinente")
    min_required = max(10, int(len(rows) * 0.02))
    missing = max(0, min_required - current_count)
    if missing == 0:
        return rows

    now = datetime(2026, 1, 1, 9, 0)
    for i in range(missing):
        created_at = now + timedelta(days=i)
        rows.append(
            {
                "request_id": f"REQ-NP-{start_index + i + 1:04d}",
                "citizen_request_text": random.choice(REQUEST_TEMPLATES["Non Pertinente"]),
                "category": "Non Pertinente",
                "office": CATEGORY_TO_OFFICE["Non Pertinente"],
                "priority": "low",
                "status": "resolved",
                "operator_response": build_operator_response("low", "Non Pertinente"),
                "created_at": created_at.isoformat(),
                "resolved_at": (created_at + timedelta(days=2)).isoformat(),
            }
        )

    return rows


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
            "operator_response": build_operator_response(priority, category),
            "created_at": created_at.isoformat(),
            "resolved_at": resolved_at.isoformat() if status == "resolved" else "",
        }
        rows.append(row)

    duplicate_count = max(20, int(num_rows * 0.05))
    for i in range(duplicate_count):
        duplicate = rows[random.randint(0, len(rows) - 1)].copy()
        duplicate["request_id"] = f"REQ-DUP-{i + 1:04d}"
        rows.append(duplicate)

    rows = ensure_non_pertinente_presence(rows, start_index=len(rows))

    return rows


def main() -> None:
    """CLI entrypoint for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic PA requests dataset.")
    parser.add_argument("--output", type=str, default=str(settings.dataset_path), help="Output CSV path")
    parser.add_argument("--rows", type=int, default=5000, help="Base number of rows before duplicates")
    args = parser.parse_args()

    rows = generate_rows(args.rows)
    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated: {output_path} | rows={len(df)}")


if __name__ == "__main__":
    main()
