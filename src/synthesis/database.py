import lancedb
import os
import sys
import pandas as pd
import pyarrow as pa
import onnxruntime as ort
from typing import List, Optional, Any
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schema.models import Facility, NGO, DocumentExtraction

def get_facility_schema():
    """Defines the fixed schema for the facilities table."""
    return pa.schema([
        ("name", pa.string()),
        ("phone_numbers", pa.list_(pa.string())),
        ("email", pa.string()),
        ("websites", pa.list_(pa.string())),
        ("yearEstablished", pa.int64()),
        ("acceptsVolunteers", pa.bool_()),
        ("facebookLink", pa.string()),
        ("twitterLink", pa.string()),
        ("linkedinLink", pa.string()),
        ("instagramLink", pa.string()),
        ("logo", pa.string()),
        ("source_doc", pa.string()),
        ("address_line1", pa.string()),
        ("address_line2", pa.string()),
        ("address_line3", pa.string()),
        ("address_city", pa.string()),
        ("address_stateOrRegion", pa.string()),
        ("address_zipOrPostcode", pa.string()),
        ("address_country", pa.string()),
        ("address_countryCode", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("facilityTypeId", pa.string()),
        ("operatorTypeId", pa.string()),
        ("affiliationTypeIds", pa.list_(pa.string())),
        ("description", pa.string()),
        ("area", pa.int64()),
        ("numberDoctors", pa.int64()),
        ("capacity", pa.int64()),
        ("procedure", pa.list_(pa.string())),
        ("equipment", pa.list_(pa.string())),
        ("capability", pa.list_(pa.string())),
        ("specialties", pa.list_(pa.string())),
        ("vector", pa.list_(pa.float32(), 384)),
    ])

def get_ngo_schema():
    """Defines the fixed schema for the NGOs table."""
    return pa.schema([
        ("name", pa.string()),
        ("phone_numbers", pa.list_(pa.string())),
        ("email", pa.string()),
        ("websites", pa.list_(pa.string())),
        ("yearEstablished", pa.int64()),
        ("acceptsVolunteers", pa.bool_()),
        ("facebookLink", pa.string()),
        ("twitterLink", pa.string()),
        ("linkedinLink", pa.string()),
        ("instagramLink", pa.string()),
        ("logo", pa.string()),
        ("source_doc", pa.string()),
        ("address_line1", pa.string()),
        ("address_line2", pa.string()),
        ("address_line3", pa.string()),
        ("address_city", pa.string()),
        ("address_stateOrRegion", pa.string()),
        ("address_zipOrPostcode", pa.string()),
        ("address_country", pa.string()),
        ("address_countryCode", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("countries", pa.list_(pa.string())),
        ("missionStatement", pa.string()),
        ("missionStatementLink", pa.string()),
        ("organizationDescription", pa.string()),
        ("vector", pa.list_(pa.float32(), 384)),
    ])

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
            real_source = f.source_doc or source_doc
            d = self._sanitize_dict(d)
            d['source_doc'] = real_source
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
            real_source = n.source_doc or source_doc
            d = self._sanitize_dict(d)
            d['source_doc'] = real_source
            d['vector'] = all_vectors[i]
            data.append(d)
        return data

    def _sanitize_dict(self, d: dict):
        """
        Only cleans list fields to ensure they are at least empty lists,
        allowing numeric/boolean fields to stay None (null).
        The explicit PyArrow schema handles the 'None' inference logic.
        """
        clean_d = {}
        list_fields = {
            "phone_numbers", "websites", "affiliationTypeIds", 
            "procedure", "equipment", "capability", "specialties", "countries"
        }
        
        for k, v in d.items():
            if v is not None and v != [] and v != "":
                clean_d[k] = v
            else:
                if k in list_fields:
                    clean_d[k] = []
                else:
                    # Keep as None/null for the schema to handle
                    clean_d[k] = None
        return clean_d

    def add_extractions(self, extraction: DocumentExtraction, source_doc: str = "Unknown"):
        """
        Adds extracted facilities and NGOs to the database with source tracking.
        """
        if extraction.facilities:
            self.add_facilities(extraction.facilities, source_doc)
            
        if extraction.ngos:
            self.add_ngos(extraction.ngos, source_doc)

    def add_facilities(self, facilities: List[Facility], source_doc: str = "CSV Seed"):
        """Adds a list of Facility objects to the database."""
        if not facilities: return
        data = self._prepare_facility_data(facilities, source_doc)
        self._upsert_table(self.facility_table_name, data)

    def add_ngos(self, ngos: List[NGO], source_doc: str = "CSV Seed"):
        """Adds a list of NGO objects to the database."""
        if not ngos: return
        data = self._prepare_ngo_data(ngos, source_doc)
        self._upsert_table(self.ngo_table_name, data)

    def _upsert_table(self, table_name: str, data: List[dict], mode: str = "append"):
        """
        Creates or updates a table with data. 
        Handles schema mismatches by recreating the table if necessary.
        Ensures data is a list of dicts.
        """
        if not data:
            return

        if table_name in self.db.table_names():
            table = self.db.open_table(table_name)
            try:
                table.add(data, mode=mode)
            except Exception as e:
                print(f"Schema mismatch in {table_name}: {e}. Recreating table...")
                self.db.drop_table(table_name)
                schema = get_facility_schema() if table_name == self.facility_table_name else get_ngo_schema()
                self.db.create_table(table_name, data=data, schema=schema)
        else:
            schema = get_facility_schema() if table_name == self.facility_table_name else get_ngo_schema()
            self.db.create_table(table_name, data=data, schema=schema)

    def search_facilities(self, query: str, limit: int = 5):
        """
        Performs a vector search for facilities. 
        If query is empty, returns the first 'limit' rows in natural order.
        """
        if self.facility_table_name not in self.db.table_names():
            return pd.DataFrame()
            
        table = self.db.open_table(self.facility_table_name)
        
        vector = self.embeddings.embed_query(query)
        results_df = table.search(vector).limit(20).to_pandas()
        
        # Hybrid Reranking: Boost results containing query keywords
        keywords = [k.lower() for k in query.split() if len(k) > 2]
        if keywords and not results_df.empty:
            def calculate_boost(row):
                text = f"{row['name']} {row.get('description', '')} {' '.join(row.get('capability', []))}".lower()
                score = 0
                for kw in keywords:
                    if kw in text: score += 1
                return score
            
            results_df['boost_score'] = results_df.apply(calculate_boost, axis=1)
            results_df = results_df.sort_values(by=['boost_score', '_distance'], ascending=[False, True])
            
        return results_df.head(limit)

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

    def clear_database(self):
        """
        Drops the facility and NGO tables to allow for a fresh seed.
        """
        for table in [self.facility_table_name, self.ngo_table_name]:
            if table in self.db.table_names():
                self.db.drop_table(table)
        print("Database cleared.")

