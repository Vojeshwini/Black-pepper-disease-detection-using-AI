"""
PepperAI Recommendation Engine
Pure Python rules engine -- no ML training needed.
Maps {disease, soil_type, rainfall, severity} -> pesticide, irrigation,
fertilizer, and soil handling recommendations.

Source: ICAR-IISR Black Pepper Extension Pamphlet (2015) + GAP for Black Pepper (2026)
"""

# ---------------------------------------------------------------------------
# 1. PESTICIDE / FUNGICIDE PLANS (keyed by disease class from the CNN)
# ---------------------------------------------------------------------------
PESTICIDE_PLANS = {
    "foot_rot": [
        {"product": "1% Bordeaux mixture", "method": "Drench soil + spray foliage",
         "timing": "At onset of monsoon (May-June), repeat every 20-25 days"},
        {"product": "Metalaxyl + Mancozeb (Ridomil MZ) 0.125%", "method": "Soil drench around collar, 5-10 L/vine",
         "timing": "Two rounds: May-June and August-September"},
        {"product": "Potassium phosphonate 0.3%", "method": "Soil drench + foliar spray",
         "timing": "May-June, repeat August-September, third round October if monsoon prolonged"},
        {"product": "Trichoderma harzianum (bio-control) 50g/vine", "method": "Soil application around base",
         "timing": "Onset of monsoon, second application August-September"},
    ],
    "slow_decline": [
        {"product": "Phorate 10G 30g/vine OR Carbofuran 3G 100g/vine", "method": "Soil application in basin",
         "timing": "May-June and September-October"},
        {"product": "Copper oxychloride 0.2% OR Potassium phosphonate 0.3%", "method": "Basin drench",
         "timing": "Alongside nematicide application"},
        {"product": "Pochonia chlamydosporia or Trichoderma harzianum 50g/vine", "method": "Soil application",
         "timing": "Twice a year: April-May and September-October"},
    ],
    "pollu_disease": [
        {"product": "Quinalphos 0.05%", "method": "Spray on spikes and berries, underside of leaves",
         "timing": "June-July and September-October (coincide with beetle emergence)"},
        {"product": "Neemgold 0.6% (neem-based, organic-compatible)", "method": "Foliar spray",
         "timing": "August, September, October at 21-day intervals"},
        {"product": "1% Bordeaux mixture", "method": "Spike/berry spray",
         "timing": "Follow-up 20 days after quinalphos"},
    ],
    "leaf_blight": [
        {"product": "Carbendazim + Mancozeb 0.1%", "method": "Foliar spray",
         "timing": "Repeat every 15 days until symptoms subside"},
        {"product": "1% Bordeaux mixture", "method": "Foliar spray, alternate with carbendazim",
         "timing": "During wet, humid spells"},
    ],
    "yellow_mottle_virus": [
        {"product": "Imidacloprid 0.5 ml/L OR Thiomethon 0.5 g/L", "method": "Spray to control aphid/mealybug vectors",
         "timing": "As soon as insect vectors are noticed"},
        {"product": "Use virus-free planting material", "method": "Source only from certified healthy mother vines",
         "timing": "At planting stage -- prevention, not cure"},
        {"product": "Roguing (remove & destroy infected vines)", "method": "Manual removal",
         "timing": "Immediately upon confirmed diagnosis, to prevent spread"},
    ],
    "healthy": [
        {"product": "No curative treatment needed", "method": "-", "timing": "-"},
        {"product": "1% Bordeaux mixture (prophylactic)", "method": "Foliar spray",
         "timing": "Once, pre-monsoon as prevention"},
    ],
}

# ---------------------------------------------------------------------------
# 2. IRRIGATION PLANS (keyed by rainfall band, mm/month)
# ---------------------------------------------------------------------------
def get_irrigation_plan(rainfall_mm_per_month, vine_age_years):
    if rainfall_mm_per_month > 250:
        band = "wet"
        plan = [
            {"method": "Drip, reduced frequency", "detail": "Every 4-5 days, 8-10 L/vine",
             "note": "Avoid waterlogging, ensure collar drainage"},
            {"method": "Skip irrigation on rain days", "detail": "-",
             "note": "Check soil moisture before each cycle"},
        ]
    elif rainfall_mm_per_month < 100:
        band = "dry"
        # summer irrigation quantity scales with vine age (from ICAR-IISR pamphlet)
        if vine_age_years >= 15:
            litres = 50
        elif vine_age_years >= 11:
            litres = 40
        else:
            litres = 30
        plan = [
            {"method": "Drip irrigation, increased frequency", "detail": f"Fortnightly, {litres} L/vine (age-adjusted)",
             "note": "Critical during Dec-Apr dry spell; irrigating in this window boosts productivity 90-100% vs unirrigated crop"},
            {"method": "Mulching", "detail": "Apply 10cm organic mulch",
             "note": "Reduces evapotranspiration loss by ~30%"},
        ]
    else:
        band = "normal"
        plan = [
            {"method": "Drip irrigation", "detail": "Every 3 days, 10-15 L/vine",
             "note": "Mulch to retain moisture"},
            {"method": "Basin irrigation (if no drip)", "detail": "Weekly, deep watering",
             "note": "Maintain basin 1-1.5m radius"},
        ]
    return band, plan

# ---------------------------------------------------------------------------
# 3. FERTILIZER PLANS (keyed by soil type) -- NPK g/vine/year, from ICAR-IISR Table 2
# ---------------------------------------------------------------------------
FERTILIZER_PLANS = {
    "laterite": [
        {"input": "FYM / compost", "quantity": "10 kg", "timing": "Pre-monsoon, mix into basin"},
        {"input": "Neem cake", "quantity": "1 kg", "timing": "Split into two applications"},
        {"input": "NPK 50:50:150 g (general) or 50:50:200 g (Panniyur/Kannur)", "quantity": "per vine/year",
         "timing": "Split May-June and August-September"},
    ],
    "loam": [
        {"input": "FYM / compost", "quantity": "8 kg", "timing": "Pre-monsoon"},
        {"input": "NPK 50:50:150 g", "quantity": "per vine/year", "timing": "Two split doses"},
    ],
    "alluvial": [
        {"input": "FYM / compost", "quantity": "8 kg", "timing": "Pre-monsoon"},
        {"input": "NPK 50:50:150 g", "quantity": "per vine/year", "timing": "Two split doses"},
        {"input": "Dolomite/lime", "quantity": "500 g", "timing": "If pH < 5.5, alternate years"},
    ],
    "clay": [
        {"input": "FYM / compost", "quantity": "12 kg", "timing": "Improves drainage, apply generously"},
        {"input": "Gypsum", "quantity": "500 g", "timing": "Improves soil structure"},
        {"input": "NPK 50:50:150 g", "quantity": "per vine/year", "timing": "Reduce N slightly, risk of waterlogging"},
    ],
    "sandy": [
        {"input": "FYM / compost", "quantity": "12 kg", "timing": "Higher dose -- low retention soil"},
        {"input": "NPK 140:55:270 g (Kozhikode-style, higher K)", "quantity": "per vine/year",
         "timing": "More frequent, smaller splits"},
        {"input": "Mulch", "quantity": "Continuous", "timing": "Critical for moisture + nutrient retention"},
    ],
}

# soil-test-based fine-tuning (from ICAR-IISR Table 2), for yield targets 3 t/ha vs 6 t/ha
SOIL_TEST_NPK = {
    "N": [(150, 50, 100), (250, 25, 80), (400, 10, 55), (float("inf"), 0, 20)],   # (upper_bound_kg_ha, dose_3t, dose_6t)
    "P": [(10, 40, 80), (30, 30, 70), (50, 10, 55), (float("inf"), 0, 30)],
    "K": [(110, 150, 310), (300, 125, 275), (500, 80, 250), (float("inf"), 35, 110)],
}

# ---------------------------------------------------------------------------
# 4. SOIL HANDLING NOTES
# ---------------------------------------------------------------------------
SOIL_NOTES = {
    "laterite": "Well-drained and naturally suited to black pepper. Maintain organic matter to buffer acidity; ideal pH 5.5-6.5.",
    "loam": "Balanced texture with good water and nutrient retention. Monitor drainage during peak monsoon to avoid Phytophthora conditions.",
    "alluvial": "Fertile but check pH -- alluvial soils can drift alkaline; correct with gypsum/organic matter if pH climbs above 7.",
    "clay": "Highest foot-rot risk due to poor drainage. Raise planting mounds, add sand/organic matter, prioritise fungicide drench schedule.",
    "sandy": "Drains fast but leaches nutrients quickly -- favour split fertilizer doses and heavy mulching to retain moisture.",
}

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def get_recommendation(disease, soil_type, rainfall_mm_per_month, vine_age_years=7, severity="medium"):
    """
    disease: one of the 6 CNN classes (foot_rot, healthy, leaf_blight,
             pollu_disease, slow_decline, yellow_mottle_virus)
    soil_type: one of laterite, loam, alluvial, clay, sandy
    rainfall_mm_per_month: float, from live weather or yield model input
    vine_age_years: int
    severity: none / low / medium / high (from disease model confidence, or user input)
    """
    disease = disease.lower()
    soil_type = soil_type.lower()

    if disease not in PESTICIDE_PLANS:
        raise ValueError(f"Unknown disease class: {disease}")
    if soil_type not in FERTILIZER_PLANS:
        raise ValueError(f"Unknown soil type: {soil_type}")

    irrigation_band, irrigation_plan = get_irrigation_plan(rainfall_mm_per_month, vine_age_years)

    result = {
        "disease": disease,
        "severity": severity,
        "pesticide_plan": PESTICIDE_PLANS[disease],
        "irrigation_band": irrigation_band,
        "irrigation_plan": irrigation_plan,
        "fertilizer_plan": FERTILIZER_PLANS[soil_type],
        "soil_notes": SOIL_NOTES[soil_type],
    }

    # add an urgency flag for the website UI
    high_risk_diseases = {"foot_rot", "slow_decline"}
    if disease in high_risk_diseases and severity in ("medium", "high"):
        result["urgency"] = "high"
        result["urgency_note"] = (
            f"{disease.replace('_',' ').title()} is one of black pepper's most destructive diseases. "
            "Act within the current spray window -- delayed treatment risks vine collapse."
        )
    elif disease == "healthy":
        result["urgency"] = "none"
        result["urgency_note"] = "No active disease detected. Maintain preventive schedule."
    else:
        result["urgency"] = "moderate"
        result["urgency_note"] = "Follow the recommended treatment schedule at the next spray window."

    return result


if __name__ == "__main__":
    # quick self-test
    import json
    example = get_recommendation(
        disease="foot_rot",
        soil_type="clay",
        rainfall_mm_per_month=310,
        vine_age_years=12,
        severity="high"
    )
    print(json.dumps(example, indent=2))
