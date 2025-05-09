from airship.physics import calculate_added_mass_inertia

def test_import():
    assert callable(calculate_added_mass_inertia)