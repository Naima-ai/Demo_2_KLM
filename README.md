
# KLM
Knowledge Models (Patient, Pathology, Research) and Meta Model Assembler for Nephrology AI Demo

## Pipeline 

Meta Model Assembler receives knowledge from three specialised KLMs, fuses it into a single ranked, evidence-weighted clinical response and delivers it to the doctor via a structured SLM output.

```
                      META MODEL ASSEMBLER
Patient KLM  ──┐
Pathology KLM ─┼──▶  1. Entity Alignment
Research KLM  ─┘          ↓
                      2. Evidence-Weighted Attention
                          ↓     
                      3. Fusion
                          ↓
                      4. Conflict Detection
                          ↓
                      5. SLM  →  Structured Clinical Response
```

| File | Purpose |
|---|---|
| `meta_model.py` | Full pipeline — entity alignment, attention, fusion, conflict detection, SLM adapters |
| `demo.py` | Interactive terminal demo with live streaming output |
| `query.py`             | One-shot query script for OpenClaw / programmatic integration   |
| `patient_triples.json` | Patient KLM — 142 triples (EHR, genetics, symptoms) |
| `pathology_triples.json` | Pathology KLM — 173 triples (disease knowledge, treatments) |
| `research_triples.json` | Research KLM — 140 triples (trial evidence, biomarkers) |
---

## Setup

Create and activate the venv with 

```
python3 -m venv env
source env/bin/activate
```
 
Install the necessary packages
```
pip install -r requirements.txt
```

## Usage [DEMO]
 
### Interactive demo (demo.py)
 
Live shell for iterative queries. Streams tokens as they arrive when Ollama is running.
 
```bash
python demo.py
```
 
Type `exit` or press Ctrl+C to quit.

### One-shot query (query.py)
 
Single call, returns answer and exits. Designed for OpenClaw tool integration.
 
```
# Plain text answer
python query.py "What treatment options are there for this patient?"
 
# Full JSON output
python query.py "What treatment options are there for this patient?" --json
 
# Control number of ranked triples fed to the prompt
python query.py "What treatment options are there for this patient?" --top-n 15
```

**JSON output shape** (for `--json`):
 
```json
{
  "query": "...",
  "answer": "CLINICAL ANSWER\n...",
  "slm": "ollama | template (no LLM)",
  "conflicts": [],
  "cross_klm_entities": [
    { "cui": "C0007134", "names": ["renal cell carcinoma"], "klms": ["patient_klm", "pathology_klm"] }
  ],
  "top_triples": [
    { "rank": 1, "klm": "pathology_klm", "head": "...", "relation": "...", "tail": "...",
      "confidence": 0.97, "evidence": "I", "annotation": "..." }
  ],
  "pipeline_ms": 90,
  "slm_ms": 1,
  "total_ms": 130
}
```

## SLM 
### Option 1: Ollama
Install
```
# Windows
winget install Ollama.Ollama
# Mac / Linux        
curl -fsSL https://ollama.com/install.sh | sh  
```
Pull a model (use your preferred model)
```
ollama pull llama3.2:1b   # fastest on CPU

```
### Option 2: Template fallback 
If Ollama is not running the demo falls back automatically to a built-in rule-based synthesiser. 
Output is fully structured and clinically correct, produced in ~15ms with no LLM.

## Example queries
 
```
What treatment options are there for this patient?
What symptoms does the patient have?
What are the symptoms of RCC?
What genetic predispositions does the patient have?
What are the risk factors for the patient?
Risk factors for CKD progression
VHL mutation significance for treatment
```

---
## Evidence weights
| Level | Weight | Source type |
|---|---|---|
| I | 1.0 | RCTs, meta-analyses |
| II | 0.8 | Cohort studies |
| III | 0.5 | Expert consensus |
| IV | 0.3 | Case reports |
| V | 0.1 | Preclinical |

