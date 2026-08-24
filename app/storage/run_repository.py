"""Repository contract for immutable completed-run snapshots."""

from typing import Any, Dict, List, Optional, Protocol, Tuple


class RunRepository(Protocol):
    def save(self, snapshot: Dict[str, Any]) -> None: ...

    def get(self, run_id: str) -> Optional[Dict[str, Any]]: ...

    def list_by_session(self, session_id: str) -> List[Dict[str, Any]]: ...

    def list(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None,
        group_by_session: bool = False,
    ) -> Tuple[List[Dict[str, Any]], int]: ...
