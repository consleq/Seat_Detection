"""Keep the Firestore view independent of short-lived local detections."""

from dataclasses import dataclass

from live_tracking import match_boxes


STABLE_SECONDS = 3.0
MISSING_SECONDS = 15.0


def boxes_close(first, second):
    """Allow a few pixels of jitter, with a larger tolerance for larger boxes."""
    x1, y1, x2, y2 = first
    x_margin = max(4, (x2 - x1) * 0.05)
    y_margin = max(4, (y2 - y1) * 0.05)
    return all(
        abs(old - new) <= margin
        for old, new, margin in zip(first, second, (x_margin, y_margin, x_margin, y_margin))
    )


@dataclass
class StableSeat:
    local_id: str
    last_seen: float
    box: tuple | None = None
    status: bool | str = "checking"
    candidate: tuple | None = None
    candidate_since: float = 0.0

    def observe(self, seat, now):
        self.last_seen = now
        if self.box is not None and seat.state.occupied is not None:
            self.status = seat.state.occupied

        if not seat.visible or (self.box is not None and boxes_close(self.box, seat.box)):
            self.candidate = None
            return
        if self.candidate is None or not boxes_close(self.candidate, seat.box):
            self.candidate = seat.box
            self.candidate_since = now
        if now - self.candidate_since >= STABLE_SECONDS:
            self.box = seat.box
            self.candidate = None
            if seat.state.occupied is not None:
                self.status = seat.state.occupied


class StableSeats:
    def __init__(self):
        self.seats = {}

    def update(self, active, now):
        evidence = {
            seat.id: seat for seat, detected in active if seat.visible or detected
        }

        # A local seat can expire before its cloud counterpart. Reuse the cloud
        # ID if a newly detected chair overlaps that chair during the grace period.
        for seat_id, stable in list(self.seats.items()):
            if stable.local_id not in evidence and (
                stable.box is None or now - stable.last_seen >= MISSING_SECONDS
            ):
                del self.seats[seat_id]
        tracked = {stable.local_id for stable in self.seats.values()}
        missing = [stable for stable in self.seats.values() if stable.local_id not in evidence]
        new = [seat for seat in evidence.values() if seat.visible and seat.id not in tracked]
        matches = match_boxes([stable.box for stable in missing], [seat.box for seat in new], 0.5)
        for index, seat in enumerate(new):
            if index in matches:
                stable = missing[matches[index]]
                stable.local_id = seat.id
                stable.candidate = None
            else:
                self.seats[seat.id] = StableSeat(seat.id, now)

        for stable in self.seats.values():
            seat = evidence.get(stable.local_id)
            if seat is None:
                stable.candidate = None
            else:
                stable.observe(seat, now)

    def snapshot(self):
        seats = []
        for seat_id, stable in self.seats.items():
            if stable.box is not None:
                x1, y1, x2, y2 = stable.box
                seats.append({
                    "id": seat_id,
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "status": stable.status,
                })
        return seats
