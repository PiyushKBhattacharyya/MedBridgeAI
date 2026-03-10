# Virtue Foundation IDP Agent

By 2030, the world will face a shortage of over 10 million healthcare workers. The Virtue Foundation faces the real-world challenge of skilled doctors being disconnected from the hospitals and communities that urgently need them.

This project aims to build an **Intelligent Document Parsing (IDP) Agent**—an agentic AI intelligence layer for healthcare. This system extracts and verifies medical facility capabilities from messy, unstructured data, reasons over it to find where care exists and where it is missing, and identifies infrastructure gaps ("medical deserts").

## Core Features
1. **Unstructured Feature Extraction**: Processes free-form text to identify specific medical data (procedures, equipment, capabilities) mapping to a detailed schema.
2. **Intelligent Synthesis**: Combines unstructured insights with structured facility schemas for a comprehensive view.
3. **Planning System**: An accessible UI designed for non-technical NGO planners using natural language.

## Project Structure (Planned)
* `data/`: Raw and processed data.
* `src/`: Core application source code.
  * `schema/`: Pydantic models for extraction.
  * `extraction/`: IDP pipelines and LLM agents.
  * `synthesis/`: RAG/Database integration.
  * `ui/`: Streamlit app providing the planning interface.
* `notebooks/`: Exploration and prototyping.
* `tests/`: Automated test suite.

## Development Phases
- **Phase 1**: Foundation & Infrastructure Setup (Data models, schemas)
- **Phase 2**: Core IDP Agent Engine (Unstructured extraction pipeline)
- **Phase 3**: Intelligent Synthesis & Storage (Database, RAG)
- **Phase 4**: Planning System & User Interface (Streamlit, NLP chat)
- **Phase 5**: Advanced Features & Refinement (Citations, Map Visualization)

## Setup Instructions

*(Coming soon...)*
