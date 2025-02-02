import numpy as np
from .calibrate_result import CalibrateResult


def folded_normal_sample(mu: float, sigma: float, size: int = 1) -> np.ndarray:
    """
    Folded Normal Distribution에서 난수를 생성하는 함수 (scipy.stats 없이 구현).

    Args:
        mu (float): 평균값 (mu)
        sigma (float): 표준편차 (sigma)
        size (int): 생성할 샘플 개수 (기본값: 1)

    Returns:
        np.ndarray: Folded Normal 분포에서 샘플링된 값들
    """
    samples = np.random.normal(loc=mu, scale=sigma, size=size)  # 정규 분포에서 샘플 생성
    return np.abs(samples)  # 음수 값을 제거 (Folded Normal 분포 적용)


class LidarCalibrator(object):
    def __init__(self):
        # Polynomial coefficients for mu(theta) [3rd degree]
        self.coeff_mu = [1.02435095e-02, 1.18158352e-03, -1.63488892e-06, -1.44258931e-07]
        # Polynomial coefficients for sigma(theta) [3rd degree]
        self.coeff_sigma = [6.84580438e-03, 1.04360732e-03, -1.14643768e-05, -1.02891733e-08]

    def poly_eval(self, theta, coeff: np.ndarray) -> float:
        """Evaluate a polynomial at given theta."""
        return sum(c * (theta ** i) for i, c in enumerate(coeff))

    def calibrate(self, theta: np.ndarray) -> CalibrateResult:
        """Perform the same calibration operation for each theta in an array or a single value."""
        theta = np.asarray(theta)  # Ensure theta is a numpy array

        # Apply the calibration function element-wise
        def single_calibrate(t):
            if t >= 75.0:
                return CalibrateResult(t, 0.0, 1.0, 0.0, 0.0)

            mu_val = self.poly_eval(t, self.coeff_mu)
            sigma_val = max(self.poly_eval(t, self.coeff_sigma), 1e-8)
            c_param = mu_val / sigma_val
            reloc_val = folded_normal_sample(mu_val,sigma_val, size=1)[0]

            return CalibrateResult(t, mu_val, sigma_val, c_param, reloc_val.item())

        # Vectorize the function so it works on both scalars and arrays
        vectorized_calibrate = np.vectorize(single_calibrate, otypes=[object])

        return vectorized_calibrate(theta)


# 기존 main 함수는 유지하되, LidarRefitter 클래스를 활용하도록 수정
def main():
    calibrator = LidarCalibrator()

    print("=== Standalone Program ===\nUsing the previously fitted polynomial coefficients.\n")

    while True:
        inp = input("\nEnter theta (or 'q' to quit): ").strip().lower()

        if inp in ['q', 'quit', 'exit']:
            print("Quit.")
            break

        try:
            theta_user = float(inp)
        except ValueError:
            print("Invalid float. Try again.")
            continue

        result = calibrator.calibrate(theta_user)

        print(result)


if __name__ == '__main__':
    main()