# Virtue Foundation IDP Agent - Implementation Plan

## Goal Description
Build an Intelligent Document Parsing (IDP) agentic AI system for the Virtue Foundation to extract, verify, and synthesize medical facility capabilities from unstructured data. The goal is to identify infrastructure gaps, map medical expertise, and highlight "medical deserts" to reduce patient wait times and optimize healthcare resource allocation.

## Phaser-wise Execution Strategy

### Phase 1: Foundation & Infrastructure Setup
**Objective**: Establish the project skeleton, set up data ingestion pipelines, and define data models.
- **Repository Setup**: Create `README.md`, set up Python environment, and define project structure (src, data, notebooks, tests).
- **Data Modeling**: Translate the provided Schema Documentation (Organization, Facility, NGO, Medical Specialties, Facts) into robust Pydantic schemas.
- **Data Ingestion**: Implement loaders for parsing raw, unstructured medical notes and facility reports.
- **Mock Data Generation**: Create a small, representative dataset (if not fully provided yet) for initial testing.

### Phase 2: Core IDP Agent Engine (Unstructured Extraction)
**Objective**: Build the AI engine capable of extracting structured facts from unstructured text.
- **LLM Integration**: Set up connection to an LLM provider (e.g., via LangChain or LlamaIndex) suitable for complex extraction tasks.
- **Extraction Pipeline**: Develop prompts and parsing logic to reliably populate the Pydantic schemas (ngos, facilities, phone numbers, specialties, procedure, equipment, capability, etc.).
- **Anomaly Detection**: Implement reasoning steps to flag suspicious or incomplete claims about hospital capabilities.

### Phase 3: Intelligent Synthesis & Storage
**Objective**: Combine extractions into a searchable, relational format.
- **Database/Vector Store**: Integrate a lightweight database (e.g., LanceDB, SQLite, or Databricks SQL if using Databricks platform) to store both the structured schemas and vectorized representations of descriptions.
- **Synthesis Agent**: Create an agent that cross-references extracted facts to provide a unified view of regional capabilities.

### Phase 4: Planning System & User Interface
**Objective**: Build an accessible, intuitive UI for non-technical NGO planners.
- **Web App**: Develop a Streamlit application.
- **Chat Interface**: Implement a natural language query interface allowing users to ask "Where are the nearest trauma centers with CT scanners?"
- **Map Visualization**: Integrate map components (e.g., Folium/PyDeck) to plot facilities and visually demonstrate "medical deserts".

### Phase 5: Advanced Features & Refinement (Stretch Goals)
**Objective**: Add transparency and operational ready features.
- **Citations & Tracing**: Implement system to trace back answers to specific rows/chunks of the source data. Consider integrating MLflow for tracking agent steps.
- **Testing & Evaluation**: Build an evaluation suite focusing on extraction accuracy and anomaly detection on edge cases.

## Verification Plan

### Automated Tests
- Unit tests for Pydantic schema validation.
- End-to-end tests passing known unstructured text and asserting extracted outputs match expected JSONs.
- Vector store retrieval tests.

### Manual Verification
- Testing the Streamlit UI interactively with complex, multi-hop questions.
- Visually inspecting the map for accurate geocoding and filtering.
- Reviewing citation links to ensure they trace accurately to the source text.
