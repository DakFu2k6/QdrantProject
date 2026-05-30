import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncManager:
    def __init__(self):
        self.last_synced_at: Optional[datetime] = None
    
    def sync_jobs(self):
        logger.info("Starting incremental sync...")
        
        # TODO: Implement actual sync logic
        # 1. Read last_synced_at from state
        # 2. Query database for jobs where updated_at > last_synced_at
        # 3. Normalize and embed jobs
        # 4. Upsert into Qdrant
        # 5. For inactive jobs, mark status or delete
        # 6. Update last_synced_at
        
        logger.info("Sync completed")

if __name__ == "__main__":
    manager = SyncManager()
    manager.sync_jobs()
