"""Physical constants.

Units used throughout unzipy: forces in pN, lengths in nm, energies in kT.
"""

KB = 0.01380649  # Boltzmann constant, pN nm / K


def kT(temperature_c: float = 25.0) -> float:
    """Return the thermal energy at a given temperature.

    Args:
        temperature_c (float): Temperature in degrees Celsius.

    Returns:
        float: ``k_B T`` in pN nm.
    """
    return KB * (temperature_c + 273.15)


KT25 = kT(25.0)  # ~4.116 pN nm
