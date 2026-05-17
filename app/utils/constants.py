"""Domain constants used across the platform."""

from __future__ import annotations


CATEGORIES = [
    "Anagrafe",
    "Tributi",
    "Edilizia",
    "Mobilita",
    "Ambiente",
    "Servizi Sociali",
]

PRIORITY_LEVELS = ["low", "medium", "high", "urgent"]

CATEGORY_TO_OFFICE = {
    "Anagrafe": "Ufficio Anagrafe e Stato Civile",
    "Tributi": "Ufficio Tributi",
    "Edilizia": "Sportello Unico Edilizia",
    "Mobilita": "Ufficio Mobilita e Viabilita",
    "Ambiente": "Ufficio Ambiente e Decoro Urbano",
    "Servizi Sociali": "Ufficio Servizi Sociali",
}
