def get_diet_plan(risk_level, t):
    """
    Returns diet list based on predicted risk
    risk_level: low / normal / high
    """

    if risk_level == "low":
        return [
            t("diet_low_1"),
            t("diet_low_2"),
            t("diet_low_3"),
            t("diet_low_4"),
        ]

    if risk_level == "high":
        return [
            t("diet_high_1"),
            t("diet_high_2"),
            t("diet_high_3"),
            t("diet_high_4"),
            t("diet_high_5"),
        ]

    # normal
    return [
        t("diet_normal_1"),
        t("diet_normal_2"),
        t("diet_normal_3"),
    ]
