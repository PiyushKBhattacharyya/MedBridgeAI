import lancedb
db = lancedb.connect("data/medbridge.lancedb")
if "facilities" in db.table_names():
    table = db.open_table("facilities")
    schema = table.schema
    for i in range(len(schema)):
        print(f"Field: {schema.names[i]} - Type: {schema.types[i]}")
else:
    print("Table facilities not found")
