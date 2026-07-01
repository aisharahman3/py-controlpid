def ziegler_nichols(ku, tu, kind="classic"):
    """Closed-loop Z-N gains from ultimate gain Ku and period Tu."""
    if tu <= 0:
        raise ValueError("Tu must be > 0")
    table = {
        "P":  (0.5 * ku, 0.0, 0.0),
        "PI": (0.45 * ku, 0.54 * ku / tu, 0.0),
        "classic": (0.6 * ku, 1.2 * ku / tu, 0.075 * ku * tu),
    }
    kp, ki, kd = table[kind]
    return {"kp": kp, "ki": ki, "kd": kd}
