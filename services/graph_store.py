"""
Graph Store Service for Neo4j knowledge graph.

Stores extracted entities and relationships in Neo4j.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import partial
from typing import List

from neo4j import GraphDatabase

from core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=2)


class GraphStoreService:
    """
    Service for storing entities and relationships in Neo4j.
    """

    def __init__(self):
        """Initialize Neo4j driver."""
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.driver.verify_connectivity()

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, partial(func, *args, **kwargs)
        )

    async def add_graph_documents(
        self,
        entities: List[dict],
        relationships: List[dict],
        document_id: str,
        source_file: str | None = None,
    ) -> dict:
        """
        Add extracted graph data to Neo4j with PENDING status for review.
        """
        nodes_created = 0
        rels_created = 0
        created_at = datetime.now(timezone.utc).isoformat()

        # Create entities with PENDING status
        for entity in entities:
            props = entity.get("properties", {})
            props["source_document_id"] = document_id
            props["status"] = "PENDING"
            props["created_at"] = created_at
            if source_file:
                props["source_file"] = source_file

            await self._run_sync(
                self.driver.execute_query,
                f"""
                MERGE (n:{entity['label']} {{name: $name}})
                SET n += $properties
                SET n.updated_at = datetime()
                """,
                name=entity["name"],
                properties=props,
                database_="neo4j",
            )
            nodes_created += 1

        # Create relationships with PENDING status
        for rel in relationships:
            rel_props = rel.get("properties", {})
            rel_props["status"] = "PENDING"
            rel_props["created_at"] = created_at
            rel_props["source_document_id"] = document_id
            if source_file:
                rel_props["source_file"] = source_file

            await self._run_sync(
                self.driver.execute_query,
                f"""
                MATCH (a:{rel['from_label']} {{name: $from_name}})
                MATCH (b:{rel['to_label']} {{name: $to_name}})
                MERGE (a)-[r:{rel['type']}]->(b)
                SET r += $properties
                SET r.updated_at = datetime()
                """,
                from_name=rel["from_name"],
                to_name=rel["to_name"],
                properties=rel_props,
                database_="neo4j",
            )
            rels_created += 1

        return {
            "nodes_created": nodes_created,
            "relationships_created": rels_created,
        }


# Singleton instance
_graph_store_service: GraphStoreService | None = None


def get_graph_store_service() -> GraphStoreService:
    """Get or create graph store service singleton."""
    global _graph_store_service
    if _graph_store_service is None:
        _graph_store_service = GraphStoreService()
    return _graph_store_service
