"""Associate detections across frames for the live camera view."""

from collections import deque
from dataclasses import dataclass, field
from math import hypot

from pose_seat_detection import SeatState, compute_iou


CHAIR_GRACE_SECONDS = 3.0
MOTION_WINDOW_SECONDS = 0.5


def match_boxes(previous, current, minimum_iou):
    """Match each detection to at most one previous box, highest overlap first."""
    pairs = []
    for previous_index, old_box in enumerate(previous):
        for current_index, new_box in enumerate(current):
            iou = compute_iou(old_box, new_box)
            if iou >= minimum_iou:
                pairs.append((iou, previous_index, current_index))

    matches = {}
    used = set()
    for _, previous_index, current_index in sorted(pairs, reverse=True):
        if previous_index not in used and current_index not in matches:
            matches[current_index] = previous_index
            used.add(previous_index)
    return matches


@dataclass
class LiveSeat:
    id: str
    box: tuple
    last_evidence: float
    visible: bool = True
    state: SeatState = field(default_factory=SeatState)

    def as_seat(self):
        x1, y1, x2, y2 = self.box
        return {
            "id": self.id,
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        }


class LiveSeats:
    def __init__(self):
        self.seats = []
        self.next_id = 1

    def update_boxes(self, boxes, now, width, height):
        current = []
        for box in boxes:
            x1, y1, x2, y2 = map(round, box)
            clipped = (
                max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
            )
            if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
                current.append(clipped)

        matches = match_boxes([seat.box for seat in self.seats], current, 0.35)
        for seat in self.seats:
            seat.visible = False
        for index, box in enumerate(current):
            if index in matches:
                seat = self.seats[matches[index]]
                seat.box = box
                seat.last_evidence = now
                seat.visible = True
            else:
                self.seats.append(LiveSeat(f"seat_{self.next_id}", box, now))
                self.next_id += 1

    def update_occupancy(self, detections, now, occupy_seconds, empty_seconds):
        active = []
        for seat, detected in zip(self.seats, detections):
            if detected:
                seat.last_evidence = now
            if seat.visible or detected:
                seat.state.update(detected, now, occupy_seconds, empty_seconds)
            else:
                # A hidden chair without a matched sitter gives no evidence of vacancy.
                seat.state.candidate = None
            if now - seat.last_evidence < CHAIR_GRACE_SECONDS:
                active.append((seat, detected))
        self.seats = [seat for seat, _ in active]
        return active


class PersonMotion:
    def __init__(self, threshold):
        self.threshold = threshold
        self.boxes = []
        self.histories = []

    def update(self, boxes, now):
        matches = match_boxes(self.boxes, boxes, 0.2)
        histories = []
        moving = []
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            history = self.histories[matches[index]] if index in matches else deque()
            x, y, height = (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1
            history.append((now, x, y, height))
            while len(history) > 1 and now - history[1][0] >= MOTION_WINDOW_SECONDS:
                history.popleft()
            previous_time, previous_x, previous_y, previous_height = history[0]
            moving.append(
                MOTION_WINDOW_SECONDS / 2 <= now - previous_time <= MOTION_WINDOW_SECONDS * 2
                and hypot(x - previous_x, y - previous_y)
                >= self.threshold * (height + previous_height) / 2
            )
            histories.append(history)
        self.boxes = boxes
        self.histories = histories
        return moving
