import pyarrow as pa
from src.schema.models import Facility, NGO

def test():
    try:
        from lancedb.pydantic import pydantic_to_schema
        schema = pydantic_to_schema(Facility)
        print("Schema successfully generated from Facility")
    except ImportError:
        print("ImportError: pydantic_to_schema not found")

test()
