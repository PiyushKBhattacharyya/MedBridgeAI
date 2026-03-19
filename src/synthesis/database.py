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
        available_providers = ort.get_available_providers()
        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        if "DmlExecutionProvider" in available_providers:
            providers.append("DmlExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            providers=providers
        )
        # MedBridgeStore initialization successful
        print(f"MedBridgeStore initialized: {providers[0] if providers else 'CPU'} prioritized.")
        
        self.facility_table_name = "facilities"
        self.ngo_table_name = "ngos"

    def _prepare_facility_data(self, facilities: List[Facility]):
        """
        Converts Pydantic Facility objects to a list of dicts with embeddings.
        """
        data = []
        for f in facilities:
            d = f.model_dump()
            
            # Create a combined text for embedding
            embedding_text = f"{f.name} {f.description or ''} "
            embedding_text += " ".join(f.procedure or []) + " "
            embedding_text += " ".join(f.equipment or []) + " "
            embedding_text += " ".join(f.capability or [])
            
            d['vector'] = self.embeddings.embed_query(embedding_text)
            data.append(d)
        return data

    def _prepare_ngo_data(self, ngos: List[NGO]):
        """
        Converts Pydantic NGO objects to a list of dicts with embeddings.
        """
        data = []
        for n in ngos:
            d = n.model_dump()
            
            # Create a combined text for embedding
            embedding_text = f"{n.name} {n.organizationDescription or n.missionStatement or ''} "
            embedding_text += " ".join(n.countries or [])
            
            d['vector'] = self.embeddings.embed_query(embedding_text)
            data.append(d)
        return data

    def add_extractions(self, extraction: DocumentExtraction):
        """
        Adds extracted facilities and NGOs to the database.
        """
        if extraction.facilities:
            facility_data = self._prepare_facility_data(extraction.facilities)
            self._upsert_table(self.facility_table_name, facility_data)
            
        if extraction.ngos:
            ngo_data = self._prepare_ngo_data(extraction.ngos)
            self._upsert_table(self.ngo_table_name, ngo_data)

    def _upsert_table(self, table_name: str, data: List[dict]):
        """
        Creates or updates a table with data. 
        For simplicity in MVP, we append or overwrite.
        """
        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            table.add(data)
        else:
            self.db.create_table(table_name, data=data)

    def search_facilities(self, query: str, limit: int = 5):
        """
        Performs a vector search for facilities.
        """
        if self.facility_table_name not in self.db.table_names():
            return pd.DataFrame()
            
        table = self.db.open_table(self.facility_table_name)
        vector = self.embeddings.embed_query(query)
        return table.search(vector).limit(limit).to_pandas()

    def search_ngos(self, query: str, limit: int = 5):
        """
        Performs a vector search for NGOs.
        """
        if self.ngo_table_name not in self.db.table_names():
            return pd.DataFrame()
            
        table = self.db.open_table(self.ngo_table_name)
        vector = self.embeddings.embed_query(query)
        return table.search(vector).limit(limit).to_pandas()
