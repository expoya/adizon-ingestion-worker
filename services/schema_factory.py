"""
Schema Factory Service for Dynamic Ontology-Based Pydantic Models.

This module provides multi-tenant support by dynamically generating Pydantic models
based on YAML ontology configurations.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, create_model


class OntologyConfig(BaseModel):
    """Parsed ontology configuration from YAML."""

    domain_name: str
    description: str
    node_types: list[dict[str, str]]
    relationship_types: list[dict[str, str]]


class SchemaFactory:
    """
    Factory for creating dynamic Pydantic models based on YAML ontology configuration.
    """

    def __init__(self, config_path: str | Path):
        """Initialize the SchemaFactory with a path to the ontology YAML file."""
        self.config_path = Path(config_path)
        self._config: OntologyConfig | None = None
        self._models: dict[str, type[BaseModel]] | None = None
        self._node_type_literal: type | None = None
        self._relationship_type_literal: type | None = None

    def load_config(self) -> OntologyConfig:
        """Load and parse the YAML ontology configuration."""
        if self._config is not None:
            return self._config

        if not self.config_path.exists():
            raise FileNotFoundError(f"Ontology config not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        self._config = OntologyConfig(**raw_config)
        return self._config

    def _build_type_literals(self) -> None:
        """Build Literal types for node and relationship types."""
        config = self.load_config()

        node_type_names = tuple(nt["name"] for nt in config.node_types)
        self._node_type_literal = Literal[node_type_names]  # type: ignore

        rel_type_names = tuple(rt["name"] for rt in config.relationship_types)
        self._relationship_type_literal = Literal[rel_type_names]  # type: ignore

    def get_node_types(self) -> list[str]:
        """Get list of allowed node type names."""
        config = self.load_config()
        return [nt["name"] for nt in config.node_types]

    def get_relationship_types(self) -> list[str]:
        """Get list of allowed relationship type names."""
        config = self.load_config()
        return [rt["name"] for rt in config.relationship_types]

    def get_dynamic_models(self) -> dict[str, type[BaseModel]]:
        """Create dynamic Pydantic models based on the ontology configuration."""
        if self._models is not None:
            return self._models

        self._build_type_literals()

        DynamicNode = create_model(
            "DynamicNode",
            type=(self._node_type_literal, Field(..., description="The type of the node")),
            name=(str, Field(..., description="The unique name/identifier of the node")),
            properties=(
                dict[str, Any],
                Field(default_factory=dict, description="Additional properties of the node")
            ),
        )

        DynamicRelationship = create_model(
            "DynamicRelationship",
            type=(
                self._relationship_type_literal,
                Field(..., description="The type of the relationship")
            ),
            source_name=(str, Field(..., description="Name of the source node")),
            source_type=(
                self._node_type_literal,
                Field(..., description="Type of the source node")
            ),
            target_name=(str, Field(..., description="Name of the target node")),
            target_type=(
                self._node_type_literal,
                Field(..., description="Type of the target node")
            ),
            properties=(
                dict[str, Any],
                Field(default_factory=dict, description="Additional properties of the relationship")
            ),
        )

        ExtractionResult = create_model(
            "ExtractionResult",
            nodes=(
                list[DynamicNode],
                Field(default_factory=list, description="List of extracted nodes")
            ),
            relationships=(
                list[DynamicRelationship],
                Field(default_factory=list, description="List of extracted relationships")
            ),
        )

        self._models = {
            "DynamicNode": DynamicNode,
            "DynamicRelationship": DynamicRelationship,
            "ExtractionResult": ExtractionResult,
        }

        return self._models

    def get_system_instruction(self) -> str:
        """Generate a system prompt instruction string based on the ontology."""
        config = self.load_config()

        lines = [
            f"# Domain: {config.domain_name}",
            f"{config.description}",
            "",
            "## Available Node Types",
            "Extract entities using ONLY these node types:",
            "",
        ]

        for nt in config.node_types:
            lines.append(f"- **{nt['name']}**: {nt['description']}")

        lines.extend([
            "",
            "## Available Relationship Types",
            "Connect nodes using ONLY these relationship types:",
            "",
        ])

        for rt in config.relationship_types:
            lines.append(f"- **{rt['name']}**: {rt['description']}")

        lines.extend([
            "",
            "## Extraction Rules",
            "1. Only use the node types and relationship types defined above.",
            "2. Each node must have a unique, descriptive name.",
            "3. Relationships must connect nodes of appropriate types.",
            "4. Include relevant properties when available in the source text.",
            "5. Do not invent information not present in the source.",
        ])

        return "\n".join(lines)

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON schema for the ExtractionResult model."""
        models = self.get_dynamic_models()
        return models["ExtractionResult"].model_json_schema()


@lru_cache(maxsize=1)
def get_schema_factory() -> SchemaFactory:
    """Get a cached SchemaFactory instance using the configured ontology path."""
    from core.config import get_settings

    settings = get_settings()
    return SchemaFactory(settings.ontology_path)


def clear_schema_factory_cache() -> None:
    """Clear the schema factory cache."""
    get_schema_factory.cache_clear()
