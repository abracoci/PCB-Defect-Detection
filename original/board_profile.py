from dataclasses import dataclass
import numpy as np


@dataclass
class BoardProfile:
    width: int
    height: int
    margin: int = 150

    @classmethod
    def from_holes(cls, holes, margin=150):
        """Estimate reference dimensions from four mounting holes."""
        holes = np.asarray(holes, dtype=np.float32)

        if holes.shape != (4, 2):
            raise ValueError("Expected four mounting holes.")

        tl, tr, br, bl = holes

        width = np.mean([
            np.linalg.norm(tr - tl),
            np.linalg.norm(br - bl)
        ])

        height = np.mean([
            np.linalg.norm(bl - tl),
            np.linalg.norm(br - tr)
        ])

        if width <= 0 or height <= 0:
            raise ValueError("Invalid board dimensions.")

        return cls(
            width=int(round(width)),
            height=int(round(height)),
            margin=margin
        )

    def template(self):
        m = self.margin
        return np.float32([
            [m, m],
            [m + self.width, m],
            [m + self.width, m + self.height],
            [m, m + self.height]
        ])
