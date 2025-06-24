# New file: airship/physics.py
"""
Physical model calculation module - includes calculations for mass, inertia and aerodynamic characteristics
Reference: Development of an Aerodynamic Model and Control Law Design for a High Altitude Airship

"""

# pylint: disable=invalid-name
# cspell:ignore ndarray
import numpy as np




# === Calculate Added Mass/Inertia ===
def calculate_added_mass_inertia(a1, a2, b, rho_air_):
    """
    Calculate the added mass and added inertia matrices based on the
    geometric parameters of a double-ellipsoid model.

    Reference Equations: Eq. 42 - 51 from the provided image.

    Args:
        a1 (float): Semi-major axis 1.
        a2 (float): Semi-major axis 2.
        b (float): Semi-minor axis.
        rho_air_ (float): Local air density.

    Returns:
        tuple: A tuple containing two NumPy arrays: (M_prime, I0_prime)
               M_prime (np.ndarray): Added mass matrix (3x3).
               I0_prime (np.ndarray): Added inertia matrix (3x3).
               k1 (float): Added mass factor 1.
               k2 (float): Added mass factor 2.
               k3 (float): Added mass factor 3.

    Raises:
        ValueError: If geometric parameters are invalid.
    """

    if b <= 0:
        raise ValueError("Semi-minor axis b must be greater than 0")

    # Calculate mean semi-major axis 'a'
    a = (a1 + a2) / 2.0


    # Check for prolate spheroid assumption a >= b
    # Note: If b > a (oblate), eccentricity and related formulas differ.
    # Assuming a >= b as consistent with the provided formulas.
    if b > a:
        print(
            f"Warning: The current formula is applicable for prolate spheroids (a >= b), "
            f"but the input is a={a:.3f}, b={b:.3f} (oblate spheroid). "
            f"The result may be inaccurate."
        )
        # Different formulas or source check needed for oblate case.
        # Continue calculation (result may be incorrect)
        # raise ValueError("Current formula only for a >= b case")

    # Calculate Volume V - Eq. 43
    V = (2.0 / 3.0) * np.pi * (a1 + a2) * b**2
    # V = (4.0 / 3.0) * np.pi * a * b**2  # Use equivalent formula with mean value a

    # Calculate mass of displaced air
    m_air = rho_air_ * V

    # Handle special case: Sphere
    tolerance = 1e-9  # Define a small tolerance
    if abs(a - b) < tolerance:
        # For a sphere, a = b, e = 0
        # k1_ = k2_ = k3_ = 0.5 (standard hydrodynamic result)
        k1_ = 0.5
        k2_ = 0.5
        k3_ = 0.5
    else:
        # Calculate eccentricity e - Eq. 44
        # Ensure a^2 > 0 and 1 - (b^2 / a^2) >= 0

        term_inside_sqrt = 1.0 - (b**2 / a)
        if (1.0 - (b**2 / (a))) < 0:
            # This should not happen theoretically when a >= b, unless there are numerical errors
            print(f"Warning: Eccentricity calculation issue " f"(term = {term_inside_sqrt:.2e}). Setting e as 0.")
            _e = 0.0
            k1_ = k2_ = k3_ = 0.5  # Fallback to sphere case
        else:
            _e = np.sqrt(1.0 - (b**2 / a))  #!! eq. 44

            # Avoid e extremely close to 1 (to avoid division by zero in f)
            if abs(1.0 - _e) < tolerance:
                raise ValueError("Eccentricity e approaches 1 (b approaches 0), invalid geometry.")

            # Calculate intermediate parameters f, g, alpha_prime, beta_prime

            # f (Eq. 45)
            f = np.log((1.0 + _e) / (1.0 - _e))  #!! eq. 45

            # g (Eq. 46)
            # Avoid division by zero for e=0 (handled in sphere case)
            e_sq = _e**2
            e_cubed = _e**3
            if abs(e_cubed) < tolerance:
                # Theoretically e is non-zero, but numerically small
                raise ValueError("Eccentricity e cubed is close to zero, cannot calculate g.")
            _g = (1.0 - e_sq) / e_cubed  #!! eq. 46

            # alpha_ (Eq. 47)
            alpha_ = 2.0 * _g * (f / 2.0 - _e)  #!! eq. 47

            # beta_ (Eq. 48)
            if abs(e_sq) < tolerance:
                raise ValueError(
                    "The square of eccentricity e is close to zero, beta_prime cannot be calculated."
                )
            beta_ = (1.0 / e_sq) - (_g * f / 2.0)  #!! eq. 48

            # Calculate inertia factors k1, k2, k3
            # k1 (Eq. 49)
            denominator_k1 = 2.0 - alpha_  # denominator, numerator, fraction
            if abs(denominator_k1) < tolerance:
                raise ValueError("Small denominator in k1 calculation")
            k1_ = - alpha_ / (2.0 - alpha_)  #!! eq. 49

            # k2 (Eq. 50)
            denominator_k2 = 2.0 - beta_
            if abs(denominator_k2) < tolerance:
                raise ValueError("Small denominator in k2 calculation")
            k2_ = - beta_ / (2.0 - beta_)  #!! eq. 50

            # k3 (Eq. 51)
            a_sq = a**2
            b_sq = b**2
            term1_num_k3 = (b_sq - a_sq) * (alpha_ - beta_)
            term2_den_k3 = 2.0 * (b_sq - a_sq) + (b_sq + a_sq) * (beta_ - alpha_)

            if abs(term2_den_k3) < tolerance:
                # Check if sphere case has been handled (e=0 -> a=b -> b^2-a^2 = 0)
                # If a != b but denominator is zero, there may be another issue or resonance case.
                if abs(a - b) > tolerance:
                    raise ValueError("Small denominator in k3 calculation (non-sphere case).")
                else:  # If it's a sphere, numerator is also zero, limit should be 0.5
                    k3_ = 0.5
            else:
                k3_ = -(1.0 / 5.0) * term1_num_k3 / term2_den_k3  #!! eq. 51

    # Construct Added Mass Matrix M_prime - Eq. 42
    _M_prime = m_air * np.diag([k1_, k2_, k2_])  #!! eq. 42

    # Construct Added Inertia Matrix I0' - Eq. 42
    _I0_prime = m_air * np.diag([0.0, k3_, k3_])  #!! eq. 42

    return _M_prime, _I0_prime, k1_, k2_, k3_
