"""How the runner talks to the tool server: read the record, move the world.

The runner needs four things the agent must never have: the call stream for a conversation,
a checkpoint at each turn boundary, a branch of a world for a fork, and a restore plus a
plant for the planted channel. They live on the control face, on its own port, and this is
the client for it.

The split is the guarantee. An agent handed only the tool URL cannot read the record of what
it did, cannot put a world back the way it was before it spent the money, and cannot write
into a field and then trigger itself. None of that rests on a secret.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agentred.mcp.recorder import RecordedCall

CONTROL_TIMEOUT_SECONDS = 30.0


class ControlError(RuntimeError):
    """The tool server could not be reached, or refused.

    Terminal for the conversation it concerns. A run that cannot read the call stream cannot
    say what the agent did, and continuing would produce a transcript whose emptiness looks
    like an agent that did nothing.
    """


@runtime_checkable
class ArenaControl(Protocol):
    """What the runner may ask of the tool server.

    An interface so the drivers can be tested against a server in the same process, with no
    socket, while production talks to one over HTTP.
    """

    def health(self) -> dict[str, Any]:
        """What the server is, and which agents it serves tools for."""
        ...

    def calls(self, run: str, session: str) -> tuple[RecordedCall, ...]:
        """Every call one session made in one run, in sequence."""
        ...

    def checkpoint(self, session: str) -> int:
        """Save the session's world at a turn boundary, and return the turn count."""
        ...

    def branch(self, source: str, session: str, at_turn: int | None = None) -> None:
        """Give a new session the source's world as it stood after `at_turn` turns."""
        ...

    def restore(self, session: str) -> None:
        """Put a session's world back to the seeded baseline."""
        ...

    def plant(
        self, session: str, *, collection: str, record_id: str, field_name: str, payload: str
    ) -> str:
        """Write attacker-controlled text into a field, and return what it replaced."""
        ...

    def subjects(
        self, session: str, *, collection: str, kinds: tuple[str, ...]
    ) -> tuple[dict[str, str], ...]:
        """Who a collection is about, one entry per record, read from the world."""
        ...


class HttpxArenaControl:
    """The real client, over HTTP, against one tool server's control face.

    Attributes:
        base_url: Origin of the control face. Comes from the registry entry for the target,
            never from a caller, for the same reason no driver accepts a chat URL.
        timeout: Seconds to wait. Short: these are local operations on data structures, and
            one that takes thirty seconds is a server in trouble rather than a slow answer.
    """

    def __init__(self, base_url: str, timeout: float = CONTROL_TIMEOUT_SECONDS) -> None:
        """Build a client.

        Args:
            base_url: Origin of the control face, with no path.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Send one control request and decode it.

        Raises:
            ControlError: If the server is unreachable, answers with a non-200 status, or
                returns something that is not a JSON object.
        """
        import httpx

        url = f"{self.base_url}{path}"
        try:
            response = httpx.request(method, url, timeout=self.timeout, **kwargs)
        except httpx.HTTPError as error:
            raise ControlError(
                f"the tool server at {self.base_url} is unreachable: {error}"
            ) from error
        if response.status_code != 200:
            raise ControlError(
                f"the tool server answered {method} {path} with HTTP "
                f"{response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
        except ValueError as error:
            raise ControlError(f"the tool server answered {path} with a non-JSON body") from error
        if not isinstance(body, dict):
            raise ControlError(
                f"the tool server answered {path} with a {type(body).__name__}, expected an object"
            )
        return body

    def health(self) -> dict[str, Any]:
        """Ask the server what it is. See `ArenaControl.health`."""
        return self._request("GET", "/health")

    def calls(self, run: str, session: str) -> tuple[RecordedCall, ...]:
        """Read the recorded stream for one conversation. See `ArenaControl.calls`."""
        body = self._request("GET", f"/calls/{run}/{session}")
        entries = body.get("calls")
        if not isinstance(entries, list):
            raise ControlError(f"the tool server answered /calls/{run}/{session} with no calls")
        try:
            return tuple(RecordedCall.from_payload(entry) for entry in entries)
        except ValueError as error:
            raise ControlError(f"the tool server returned an unreadable record: {error}") from error

    def checkpoint(self, session: str) -> int:
        """Checkpoint one world. See `ArenaControl.checkpoint`."""
        body = self._request("POST", f"/sessions/{session}/checkpoint")
        return int(body.get("turns", 0))

    def branch(self, source: str, session: str, at_turn: int | None = None) -> None:
        """Branch one world. See `ArenaControl.branch`."""
        self._request(
            "POST",
            "/sessions/branch",
            json={"source": source, "session": session, "at_turn": at_turn},
        )

    def restore(self, session: str) -> None:
        """Restore one world to the baseline. See `ArenaControl.restore`."""
        self._request("POST", f"/sessions/{session}/restore")

    def plant(
        self, session: str, *, collection: str, record_id: str, field_name: str, payload: str
    ) -> str:
        """Plant a payload. See `ArenaControl.plant`."""
        body = self._request(
            "POST",
            "/plant",
            json={
                "session": session,
                "collection": collection,
                "record_id": record_id,
                "field_name": field_name,
                "payload": payload,
            },
        )
        return str(body.get("replaced", ""))

    def subjects(
        self, session: str, *, collection: str, kinds: tuple[str, ...]
    ) -> tuple[dict[str, str], ...]:
        """Read a cohort from the world. See `ArenaControl.subjects`."""
        body = self._request(
            "GET",
            f"/subjects/{session}",
            params=[("collection", collection), *(("kind", kind) for kind in kinds)],
        )
        found = body.get("subjects")
        if not isinstance(found, list):
            raise ControlError(
                f"the tool server answered /subjects/{session} with no cohort. An attempt "
                f"whose cohort cannot be read is not an attempt with an empty one."
            )
        return tuple(
            {str(k): str(v) for k, v in entry.items()} for entry in found if isinstance(entry, dict)
        )
