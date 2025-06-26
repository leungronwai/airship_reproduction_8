"""
aero_coefficients.py

"""



# pylint: disable=invalid-name
# pylint: disable=undefined-variable
# cspell:ignore coeffs ddelta eta_f Sh Sg Sf Cdcf dalpha arcsin coeff
import numpy as np


from src.config import parameters as params




# from added_mass_calculator import calculate_added_mass_inertia # Assumed available

# ==============================================================================
#   (Define Basic Geometric, Env, Aero Params)
# ==============================================================================


# === Geometric Parameters ===
airship_a1 = params.airship_a1  # [m] Front ellipsoid semi-major axis
airship_a2 = params.airship_a2  # [m] Rear ellipsoid semi-major axis
airship_b = params.airship_b  # [m] Semi-minor axis

Lh = airship_a1 + airship_a2  # [m] Total hull length - Assumption
L_ref = Lh  # [m] Reference length - Assumption
# Volume Center x-coordinate - Placeholder,
# Exact value depends on specific combination of double ellipsoids, using average a approximation or assuming origin at specific position
xcv = airship_a1 + (3 / 8) * (airship_a2 - airship_a1)  # [m] - Placeholder, assume origin or calc if needed Eq.22

# Calculate Volume
mean_a = (airship_a1 + airship_a2) / 2.0
V_airship = (4.0 / 3.0) * np.pi * mean_a * airship_b**2  # [m^3]

Sh = V_airship ** (2.0 / 3.0)  # [m^2] Hull reference area
Sf = 3656.0  # [m^2] Fin reference area
Sg = 202.0  # [m^2] Gondola reference area

# Lever Arms
lf1 = 117.5  # [m] x-dist origin to aero center fins
lf2 = 129.7  # [m] x-dist origin to geom center fins
lf3 = 18.3  # [m] y,z-dist origin to aero center fins (Used for Cl1)
lgx = 29.2  # [m] x-dist origin to aero center gondola
lgz = 40.0  # [m] z-dist origin to aero center gondola (Used for Cl2)

# === Environmental Parameters ===
rho_air_at_altitude = params.rho_air_at_altitude  # [kg/m^3] Air density @ ~20km - Placeholder

# === Basic Aero Coeffs & Derivatives from Table 2 ===
CDh0 = 0.025
CDf0 = 0.006
CDg0 = 0.01
CDch = 0.5
CDcf = 1.0
CDcg = 1.0
dCL_dalpha_f = 5.73  # (∂CL/∂α)f
dCL_ddelta_f = 1.24  # (∂CL/∂δ)f

# === Efficiency & Integral Factors from Table 2 ===
eta_f = 0.29
eta_k = 1.19
# Using integral factor values from Table 2
I1_table = 0.33
I3_table = -0.69
J1_table = 1.31
J2_table = 0.53
# Note: If calculation based on geometry (Eq. 82-85) is preferred, uncomment below


# Defaulting to table values
I1 = I1_table
I3 = I3_table
J1 = J1_table
J2 = J2_table





# ==============================================================================
#  Function to Calculate Higher-Order Aero Coeffs
# ==============================================================================


def get_aero_coefficients(k1=0, k2=0):
    """
    Calculate aerodynamic coefficients Cx1...Cn4.
    Uses the global basic parameters defined at the top of this file.

    Args:
        k1 (float): Added mass factor k1.
        k2 (float): Added mass factor k2.

    Returns:
        dict: Dictionary containing all calculated aerodynamic coefficients.
    """
    coeffs = {}



    # Use module-level defined parameters
    # Eq. 66-81
    coeffs["Cx1"] = -(CDh0 * Sh + CDf0 * Sf + CDg0 * Sg)
    coeffs["Cx2"] = (k2 - k1) * eta_k * I1 * Sh
    # Assume Cz4 is a typo or Cz4 depends on different derivatives
    # Calculate Cy4, Cz4 based on the most direct interpretation

    # coeffs['Cz4'] = 0.5 * dCL_ddelta_f * Sf * eta_f # Assume control surface efficiency is the same
    coeffs["Cz1"] = coeffs["Cx2"]
    coeffs["Cy1"] = coeffs["Cx2"]

    coeffs["Cy2"] = -0.5 * dCL_dalpha_f * Sf * eta_f
    coeffs["Cz2"] = coeffs["Cy2"]  #

    coeffs["Cy3"] = -(CDch * J1 * Sh + CDcf * Sf + CDcg * Sg)
    coeffs["Cy4"] = 0.5 * dCL_ddelta_f * Sf * eta_f
    coeffs["Cz4"] = coeffs["Cy4"]
    coeffs["Cz3"] = -(CDch * J1 * Sh + CDcf * Sf)

    coeffs["Cl1"] = dCL_ddelta_f * Sf * eta_f * lf3
    coeffs["Cl2"] = -CDcg * Sg * lgz
    coeffs["Cm1"] = -(k1 - k2) * eta_k * I3 * Sh * L_ref
    coeffs["Cm2"] = -0.5 * dCL_dalpha_f * Sf * eta_f * lf1
    coeffs["Cm3"] = -(CDch * J2 * Sh * L_ref + CDcf * Sf * lf2)
    coeffs["Cm4"] = 0.5 * dCL_ddelta_f * Sf * eta_f * lf1

    coeffs["Cn1"] = -coeffs["Cm1"]
    coeffs["Cn2"] = -coeffs["Cm2"]
    coeffs["Cn3"] = -coeffs["Cm3"]
    coeffs["Cn4"] = -coeffs["Cm4"]  # <-- Using Cm4

    return coeffs





# ==============================================================================
#  Main execution part (Example)
# ==============================================================================
if __name__ == "__main__":
    # This part only runs when aero_coefficients.py is directly executed, for testing
    print("--- Testing Aerodynamic Coefficient Calculation ---")
    try:
        # Use calculated k1, k2 to compute aerodynamic coefficients
        inertia_k1 = params.k1
        inertia_k2 = params.k2
        aero_coeffs_calculated = get_aero_coefficients(k1=inertia_k1, k2=inertia_k2)

        print("\nCalculated Aerodynamic Coefficients:")
        for coeff, value in aero_coeffs_calculated.items():
            print(f"  {coeff}: {value:.4f}")

    except ValueError as e:
        print(f"\nError during calculation: {e}")
