"""Thin wrapper over FAISS index + SQLite metadata."""
import faiss
import sqlite3
from pathlib import Path
import numpy as np
from typing import List, Any, Tuple

class VectorStore:
    def __init__(self, index_path: Path, db_path: Path, dim: int):
        self.index_path = index_path
        self.db_path = db_path
        self.dim = dim
        self._load_index()
        self._init_db()

    def _load_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS chunks
                       (id TEXT PRIMARY KEY, file TEXT, unit TEXT,
                        text TEXT, mtime REAL, current INTEGER, faiss_idx INTEGER)""")
        self.conn.commit()

    def upsert(self, chunk_ids: List[str], vectors: Any, metadata_rows: List[Tuple]):
        vectors = np.asarray(vectors, dtype='float32')
        
        # Get current FAISS index size to assign new indices
        current_size = self.index.ntotal
        
        # Add vectors to FAISS
        self.index.add(vectors)
        
        # Update metadata with FAISS indices
        cur = self.conn.cursor()
        for i, row in enumerate(metadata_rows):
            # Add FAISS index to metadata: (id, file, unit, text, mtime, current, faiss_idx)
            extended_row = row + (current_size + i,)
            cur.execute("REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)", extended_row)
        
        self.conn.commit()
        faiss.write_index(self.index, str(self.index_path))

    def query(self, vector: Any, k: int = 8) -> List[dict]:
        vec = np.asarray(vector, dtype='float32').reshape(1, -1)
        dists, idxs = self.index.search(vec, k)
        
        # Map FAISS indices back to chunk data via SQLite
        results = []
        cur = self.conn.cursor()
        for i, (dist, idx) in enumerate(zip(dists[0], idxs[0])):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            # Get chunk by FAISS index
            cur.execute("SELECT id, file, unit, text FROM chunks WHERE faiss_idx = ? AND current = 1", (int(idx),))
            row = cur.fetchone()
            if row:
                results.append({
                    'id': row[0],
                    'file': row[1], 
                    'unit': row[2],
                    'text': row[3],
                    'distance': float(dist)
                })
        return results
