import sys, rag
if len(sys.argv) < 2:
    sys.exit("Usage: python ingest.py SIET2026")
print(f"Done. {rag.ingest(sys.argv[1])} chunks stored.")
