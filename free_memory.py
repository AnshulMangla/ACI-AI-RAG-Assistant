# free_memory.py

import gc

def free_up_memory():
    # Delete known large variables if you're running in an interactive session
    globals_to_clear = ["text", "chunks", "embeddings", "index", "model"]
    for var in globals_to_clear:
        if var in globals():
            del globals()[var]

    # Collect garbage
    gc.collect()
    print("✅ Memory cleared.")

if __name__ == "__main__":
    free_up_memory()
