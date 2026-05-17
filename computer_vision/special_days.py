# Benin special days (month, day)
BENIN_HOLIDAYS = {
    (1, 1),    # Nouvel An
    (1, 10),   # Fête du Vodoun
    (4, 1),    # Pâques (approximate)
    (5, 1),    # Fête du Travail
    (8, 1),    # Fête Nationale
    (8, 15),   # Assomption
    (11, 1),   # Toussaint
    (11, 15),  # Jour de la République
    (12, 25),  # Noël
}

# Moving holidays (need year-specific calculation)
MOVING_HOLIDAYS_2024 = {
    (3, 29),   # Tabaski 2024
}

def is_special_day(dt):
    """Check if datetime is a Benin special day."""
    return (dt.month, dt.day) in BENIN_HOLIDAYS