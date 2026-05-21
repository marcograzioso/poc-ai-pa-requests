"""Prompt templates for governance-aware generation."""

from __future__ import annotations


RESPONSE_DRAFT_PROMPT = """
Sei un assistente per operatori della Pubblica Amministrazione italiana.
Obiettivo: generare una bozza di risposta formale per l'operatore.

Regole vincolanti:
1) Usa esclusivamente le informazioni fornite nel contesto, inerenti alla richiesta del cittadino.
2) Non inventare norme, dati o procedure non presenti.
3) Mantieni tono istituzionale, chiaro e cortese.
4) Indica esplicitamente che il testo e una bozza da validare da parte dell'operatore.
5) Non scrivere mai un numero di giorni di attesa o scadenza anche se menzionato nei casi simili ma inserisci un placeholder [TEMPO_STIMATO].

Richiesta cittadino:
{new_request}

Casi simili e risposte storiche:
{retrieved_context}

Produci una risposta in italiano, max 220 parole.
""".strip()


SUMMARIZATION_PROMPT = """
Riassumi in massimo 5 punti chiave la richiesta seguente, in italiano formale:
{request_text}
""".strip()


EXPLANATION_PROMPT = """
Spiega in linguaggio semplice per un operatore PA:
- categoria prevista: {category}
- priorita prevista: {priority}
- confidenza categoria: {category_confidence:.2f}
- confidenza priorita: {priority_confidence:.2f}

Evidenzia eventuali limiti della previsione e suggerisci verifica umana.
""".strip()
