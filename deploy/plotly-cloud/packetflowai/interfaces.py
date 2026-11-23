"""Protocol interfaces for later architecture phases."""

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from .domain import FeedbackRecord, LocalPrediction


@runtime_checkable
class TaxonomyMapper(Protocol):
    def normalize(self, native_label: Any) -> str: ...


@runtime_checkable
class MitreMapper(Protocol):
    def map_techniques(self, attack_family: str, evidence: Mapping[str, Any]) -> tuple[str, ...]: ...


@runtime_checkable
class EvidenceSanitizer(Protocol):
    def sanitize(self, evidence: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class ReadOnlyReasoningTool(Protocol):
    @property
    def name(self) -> str: ...

    def query(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


@runtime_checkable
class ActiveLearningSelector(Protocol):
    def select(self, predictions: Iterable[LocalPrediction]) -> list[str]: ...


@runtime_checkable
class ModelRegistry(Protocol):
    def list_models(self) -> list[Mapping[str, Any]]: ...

    def active_model(self) -> Mapping[str, Any]: ...


@runtime_checkable
class FeedbackStore(Protocol):
    def add(self, record: FeedbackRecord) -> None: ...
