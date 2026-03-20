import lancedb
import os
import pandas as pd
import onnxruntime as ort
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from src.schema.models import Facility, NGO, DocumentExtraction

load_dotenv()

class MedBridgeStore:
    def __init__(self, db_path: str = "data/medbridge.lancedb"):
        """
        Initializes the LanceDB store and the embedding function.
        """
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        
        # Initialize Local FastEmbed Embeddings with Multi-Hardware support
        # Use CPU for embeddings for stability (it's fast enough for BGE-small)
        providers = ["CPUExecutionProvider"]
        
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            providers=providers
        )
        # MedBridgeStore initialization successful
        print(f"MedBridgeStore initialized: {providers[0] if providers else 'CPU'} prioritized.")
        
        self.facility_table_name = "facilities"
        self.ngo_table_name = "ngos"

    def _prepare_facility_data(self, facilities: List[Facility], source_doc: str = "Unknown"):
        """
        Converts Pydantic Facility objects to a list of dicts with embeddings and source info.
        Uses chunked batch embedding for real-time progress logging.
        """
        texts = []
        for f in facilities:
            embedding_text = f"{f.name} {f.description or ''} "
            embedding_text += " ".join(f.procedure or []) + " "
            embedding_text += " ".join(f.equipment or []) + " "
            embedding_text += " ".join(f.capability or [])
            texts.append(embedding_text)
        
        batch_size = 50
        all_vectors = []
        print(f"Generating embeddings for {len(texts)} facilities in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            vectors = self.embeddings.embed_documents(chunk)
            all_vectors.extend(vectors)
            print(f"\tEmbedded {min(i + batch_size, len(texts))}/{len(texts)} facilities...")
        
        data = []
        for i, f in enumerate(facilities):
            d = f.model_dump()
            d['source_doc'] = source_doc
            d['vector'] = all_vectors[i]
            data.append(d)
        return data

    def _prepare_ngo_data(self, ngos: List[NGO], source_doc: str = "Unknown"):
        """
        Converts Pydantic NGO objects to a list of dicts with embeddings and source info.
        Uses chunked batch embedding for real-time progress logging.
        """
        texts = []
        for n in ngos:
            embedding_text = f"{n.name} {n.organizationDescription or n.missionStatement or ''} "
            embedding_text += " ".join(n.countries or [])
            texts.append(embedding_text)
            
        batch_size = 50
        all_vectors = []
        print(f"Generating embeddings for {len(texts)} NGOs in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            vectors = self.embeddings.embed_documents(chunk)
            all_vectors.extend(vectors)
            print(f"\tEmbedded {min(i + batch_size, len(texts))}/{len(texts)} NGOs...")
        
        data = []
        for i, n in enumerate(ngos):
            d = n.model_dump()
            d['source_doc'] = source_doc
            d['vector'] = all_vectors[i]
            data.append(d)
        return data

    def add_extractions(self, extraction: DocumentExtraction, source_doc: str = "Unknown"):
        """
        Adds extracted facilities and NGOs to the database with source tracking.
        """
        if extraction.facilities:
            facility_data = self._prepare_facility_data(extraction.facilities, source_doc)
            self._upsert_table(self.facility_table_name, facility_data)
            
        if extraction.ngos:
            ngo_data = self._prepare_ngo_data(extraction.ngos, source_doc)
            self._upsert_table(self.ngo_table_name, ngo_data)

    def _upsert_table(self, table_name: str, data: List[dict], mode: str = "append"):
        """
        Creates or updates a table with data. 
        Handles schema mismatches by recreating the table if necessary.
        """
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            try:
                table.add(data, mode=mode)
            except Exception as e:
                print(f"Schema mismatch in {table_name}, recreating table: {e}")
                self.db.drop_table(table_name)
                self.db.create_table(table_name, data=data)
        else:
            self.db.create_table(table_name, data=data)

    def search_facilities(self, query: str, limit: int = 5):
        """
        Performs a vector search for facilities. 
        If query is empty, returns the first 'limit' rows in natural order.
        """
        if self.facility_table_name not in self.db.table_names():
            return pd.DataFrame()
            
        table = self.db.open_table(self.facility_table_name)
        
        # Natural order for empty queries
        if not query.strip():
            return table.to_pandas().head(limit)
            
        vector = self.embeddings.embed_query(query)
        return table.search(vector).limit(limit).to_pandas()

    def search_ngos(self, query: str, limit: int = 5):
        """
        Performs a vector search for NGOs.
        If query is empty, returns the first 'limit' rows in natural order.
        """
        if self.ngo_table_name not in self.db.table_names():
            return pd.DataFrame()
            
        table = self.db.open_table(self.ngo_table_name)

        # Natural order for empty queries
        if not query.strip():
            return table.to_pandas().head(limit)

        vector = self.embeddings.embed_query(query)
        return table.search(vector).limit(limit).to_pandas()


    def get_all_facilities(self):
        """
        Retrieves all facility records for CSV export.
        """
        if self.facility_table_name not in self.db.table_names():
            return pd.DataFrame()
        return self.db.open_table(self.facility_table_name).to_pandas()

    def get_all_ngos(self):
        """
        Retrieves all NGO records for CSV export.
        """
        if self.ngo_table_name not in self.db.table_names():
            return pd.DataFrame()
        return self.db.open_table(self.ngo_table_name).to_pandas()

