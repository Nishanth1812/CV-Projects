import os
import psycopg2  # type: ignore
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from contextlib import contextmanager
import numpy as np
from pgvector.psycopg2 import register_vector #type: ignore 

load_dotenv(override=True)

DATABASE_URL=os.getenv("DATABASE_URL")

_pool=None 


# Initialising the database pool
def init_pool():
    
    global _pool
    if _pool is None:
        _pool=SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=DATABASE_URL
        )
    
    return _pool 

@contextmanager
def get_db_conn():
    pool=init_pool()
    conn=pool.getconn()
    
    try:
        try:
            register_vector(conn)
        except Exception:
            pass
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
    
    
def init_database():
    
    with get_db_conn() as conn:
        with conn.cursor() as cur:
            
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;") 
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='users' AND column_name='password_hash'
                    ) THEN
                        ALTER TABLE users ADD COLUMN password_hash TEXT;
                    END IF;
                END $$;
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    embedding VECTOR(128) NOT NULL,
                    model_name TEXT DEFAULT 'Facenet',
                    model_version TEXT DEFAULT 'deepface',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
                )
            """)
            
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE c.relname = 'idx_embeddings_cosine' AND n.nspname = 'public'
                    ) THEN
                        CREATE INDEX idx_embeddings_cosine
                        ON face_embeddings USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                    END IF;
                END $$;
            """)
            
            cur.execute("""
                SELECT data_type, udt_name FROM information_schema.columns
                WHERE table_name='face_embeddings' AND column_name='embedding'
            """)
            
            row=cur.fetchone()
            if row and (row[0]=='bytea' or row[1]== 'bytea'):
                cur.execute("ALTER TABLE face_embeddings ADD COLUMN IF NOT EXISTS embedding_vec VECTOR(128)")
                
                cur.execute("SELECT username, embedding FROM face_embeddings WHERE embedding IS NOT NULL")
                
                for uname,emb_bytes in cur.fetchall():
                    if emb_bytes:
                        try:
                            arr=np.frombuffer(emb_bytes,dtype=np.float32)
                            
                            arr=arr/(np.linalg.norm(arr))
                            
                            cur.execute("UPDATE face_embeddings SET embedding_vec = %s WHERE username = %s",
                                (arr.tolist(), uname)
                                )
                            
                        except Exception:
                            pass 
                        
                cur.execute("ALTER TABLE face_embeddings DROP COLUMN embedding")
                cur.execute("ALTER TABLE face_embeddings RENAME COLUMN embedding_vec TO embedding")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_cosine
                    ON face_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """) 
                


"""Database Operations"""

# User helper functions

def load_users():
    try:
        with get_db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT username, email, password_hash FROM users")
                return {r['username']: {'email': r['email'], 'password': r['password_hash']} for r in cur.fetchall()} 
    except Exception as e:
        print(f"Error while loading users: {e}")
        return {} 
    

def save_user(username,email,password_hash=None):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (username, email, password_hash)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (username) DO UPDATE SET
                        email = EXCLUDED.email,
                        password_hash = COALESCE(EXCLUDED.password_hash, users.password_hash),
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (username, email, password_hash))
                return cur.fetchone()[0]
    
    except Exception as e:
        print(f"Error while saving the user: {e}")
        return None 
    
def get_user(username):
    try:
        with get_db_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT username, email, password_hash FROM users WHERE username = %s", (username,)) 
                r=cur.fetchone()
                
                return {'username': r['username'], 'email': r['email'], 'password': r['password_hash']} if r else None 
    except Exception as e:
        print(f"Error while getting user data: {e}")
        return None 
    
def user_exists(username):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE username = %s LIMIT 1", (username,))
                return cur.fetchone() is not None 
    
    except Exception as e:
        print(f"Error with the database: {e}")
        return False 
    
def delete_user(username):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users WHERE username = %s RETURNING id", (username,))
                return cur.fetchone() is not None 
    except Exception as e:
        print(f"Couldnt delete user: {e}")
        return False
    
    
# Face embedding helper function

def _normalize_vector(arr):
    arr=np.asarray(arr,dtype=np.float32)
    nrm=np.linalg.norm(arr)
    return arr/nrm 

def save_embedding(username,embedding):
    try:
        embedding=_normalize_vector(embedding)
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings (username, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (username) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, (username, embedding.tolist()))
                return cur.fetchone()[0]
    except Exception as e:
        print(f"Error while saving embedding: {e}")
        return None
    
def load_embedding(username):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT embedding FROM face_embeddings WHERE username = %s", (username,))
                r=cur.fetchone()
                return np.array(r[0], dtype=np.float32) if r else None 
    
    except Exception as e:
        print(f"Couldn't load embeddings: {e}")
        return None 
    

def del_embedding(username):
    try:
        with get_db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM face_embeddings WHERE username = %s RETURNING id", (username,))
                return cur.fetchone() is not None 
    except Exception as e:
        print(f"Couldn't delete the embedding: {e}")
        return False
    

try:
    init_database()
except Exception as e:
    print(f"Couldn't init database {e}")
    