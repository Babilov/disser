from dataclasses import dataclass

@dataclass
class Roi:
    cords: list  # [[x1, y1], ...]
    index: int

    def to_dict(self):
        return {"cords": self.cords, "index": self.index}
