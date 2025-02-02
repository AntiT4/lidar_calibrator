from dataclasses import dataclass

@dataclass
class CalibrateResult:
    theta: float
    mu: float
    sigma: float
    c: float
    relocated_value: float

    def __str__(self):
        return f"theta={self.theta:.3f} => mu={self.mu:.6f}, sigma={self.sigma:.6f}, c={self.c:.4f} => relocated_value={self.relocated_value:.4f}"