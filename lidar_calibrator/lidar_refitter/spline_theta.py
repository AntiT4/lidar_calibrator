import numpy as np
from .calibrate_result import CalibrateResult


def bspline_basis(i, k, t, x):
    """
    Compute the B-spline basis function value for index i, degree k, knots t, at position x.
    """
    if k == 0:
        return np.where((t[i] <= x) & (x < t[i + 1]), 1.0, 0.0)
    else:
        left = 0.0
        if (t[i + k] - t[i]) != 0:
            left = ((x - t[i]) / (t[i + k] - t[i])) * bspline_basis(i, k - 1, t, x)

        right = 0.0
        if (t[i + k + 1] - t[i + 1]) != 0:
            right = ((t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1])) * bspline_basis(i + 1, k - 1, t, x)

        return left + right


def bspline_evaluate(t, c, k, x):
    """
    Evaluate a B-spline at given x using the de Boor algorithm.

    Args:
        t (np.ndarray): Knots array
        c (np.ndarray): Coefficients array
        k (int): Degree of the spline
        x (float or np.ndarray): Points at which to evaluate

    Returns:
        float or np.ndarray: Evaluated B-spline values at x
    """
    n = len(c) - 1
    spline_value = np.zeros_like(x, dtype=np.float64)

    for i in range(n + 1):
        spline_value += c[i] * bspline_basis(i, k, t, x)

    return spline_value


def folded_normal_sample(mean, std, size, rng):
    """
    Folded Normal Distribution에서 샘플을 생성하는 함수.

    Args:
        mean (float): 평균 (mu)
        std (float): 표준 편차 (sigma)
        size (int): 생성할 샘플 개수
        rng (numpy.random.Generator, optional): 난수 값

    Returns:
        np.ndarray: Folded Normal 분포에서 샘플링된 값
    """
    # 정규 분포에서 샘플링 후 절댓값 적용
    samples = np.abs(rng.normal(loc=mean, scale=std, size=size))

    return samples if size > 1 else samples[0]


# Material properties dictionary including mean, standard deviation parameters, and dropout spline coefficients
material_dict = {
    "Aluminium": {
        "mean_params": [0.0564, 0.2014, 0.0, 50.0, 21.9503, 0.0003, 0.0109],
        "std_params":  [0.0620, 0.3197, 0.0, 47.8725, 20.9216, 0.0240, 5.6573],
        "spline_knots": np.array([1.0, 6.6, 7.0, 7.2, 7.3, 8.0, 8.7, 9.4, 12.1, 23.2, 34.3, 45.4, 56.5, 67.6, 89.7]),
        "spline_coefs": np.array([-0.03669465, 0.14803687, -0.22405445, 0.15701279, 0.28178591, 0.38203843, 0.09171097, 0.33909473, 0.36862915, 0.66832097, 0.53964095, 0.73963473, 0.679984, 0.93491434, 0.94019927, 1.05936161, 0.97708951]),
        "spline_degree": 3
    },
    "SUS": {
        "mean_params": [0.1020, 1.1358, -0.0810, 13.7612, 23.7153, -0.0375, 9.1649],
        "std_params":  [0.0692, 0.9601, -0.0601, 9.6128, 28.1161, -0.0179, 6.9438],
        "spline_knots": np.array([1.0, 6.6, 8.0, 8.7, 9.1, 9.4, .8, 12.1, 17.7, 23.2, 34.3, 39.9, 45.4, 56.5, 67.6, 89.7]),
        "spline_coefs": np.array([-0.01205687, 0.05382891, -0.11811259, 0.16537451, 0.16514144, 0.24220003, 0.29965838, 0.26438822, 0.41168756, 0.30525519, 0.19341307, 0.35222604, 0.23225088, 0.58998089, 0.75704803, 1.0594766, 0.99483187, 0.99746057]),
        "spline_degree": 3
    }
}

material_ind = ["Aluminium", "SUS"]


# Lambert + Specular + Gaussian weighted sum model
def lambert_specular_gauss(theta_deg, wL, p, wS, n, theta_peak, amp_peak, width_peak):
    theta_rad = np.deg2rad(theta_deg)
    cos_val = np.cos(theta_rad)
    lambert_term = np.where(cos_val > 0, cos_val**p, 0.0)
    specular_term = np.where(cos_val > 0, cos_val**n, 0.0)
    delta_peak = amp_peak * np.exp(-((theta_deg - theta_peak)**2)/(2*(width_peak**2)))
    return wL * lambert_term + wS * specular_term + delta_peak


# Retrieve material model based on material_id
def get_material_model(material_id):
    if material_id not in material_dict:
        raise ValueError(f"Unknown material_id={material_id}")
    mat_info = material_dict[material_id]
    t = mat_info["spline_knots"]
    c = mat_info["spline_coefs"]
    k = mat_info["spline_degree"]
    # NumPy 기반 B-Spline 대체
    drop_spline = lambda theta: bspline_evaluate(t, c, k, theta)

    mean_p = mat_info["mean_params"]
    std_p = mat_info["std_params"]

    def mean_func(theta_deg):
        return lambert_specular_gauss(theta_deg, *mean_p)

    def std_func(theta_deg):
        return lambert_specular_gauss(theta_deg, *std_p)
    return drop_spline, mean_func, std_func


class LidarCalibrator(object):
    def __init__(self):
        super().__init__()

    def poly_eval(self, theta, coeff: np.ndarray) -> float:
        """Evaluate a polynomial at given theta."""
        return sum(c * (theta ** i) for i, c in enumerate(coeff))

    def calibrate(self, theta: np.ndarray, material_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Perform the same calibration operation for each theta in an array or a single value."""
        theta = np.asarray(theta)  # Ensure theta is a numpy array
        rng = np.random.default_rng()

        # Apply the calibration function element-wise
        def single_calibrate(t, m_id):
            drop_spline, mean_func, std_func = get_material_model(material_id[m_id])

            drop_raw = drop_spline(t)
            drop_prob = np.clip(drop_raw, 0.0, 1.0)
            rnd_val = rng.random()
            is_lost = (rnd_val < drop_prob)
            shift_val = None
            if not is_lost:
                m = mean_func(t)
                s = std_func(t)
                if s <= 0:
                    s = 1e-9
                shift_val = folded_normal_sample(m, s, 1, rng)

            return shift_val, is_lost

        # ✅ 두 개의 벡터를 따로 반환하도록 설정
        vectorized_calibrate = np.vectorize(single_calibrate, otypes=[float, bool])

        # ✅ 개별 벡터로 분리하여 반환
        shift_vals, lost_flags = vectorized_calibrate(theta, material_id)

        return shift_vals, lost_flags


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