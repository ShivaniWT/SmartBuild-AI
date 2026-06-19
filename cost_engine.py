import os
from io import BytesIO
from datetime import datetime
from pydoc import doc

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


st.write("✅ RUN CHECK - NEW VERSION - 19Jun 2024")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"C:\Users\shivani.thakur\Documents\SmartBuild"
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRAINED_DIR = os.path.join(DATA_DIR, "trained")

os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINED_DIR, exist_ok=True)

PAST_ESTIMATES_FILE = os.path.join(PROCESSED_DIR, "past_estimates.csv")

FORECAST_MODEL_PATH = os.path.join(TRAINED_DIR, "forecast_model_v1.pkl")
FORECAST_META_PATH = os.path.join(TRAINED_DIR, "forecast_model_v1_metadata.pkl")

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="SmartBuild AI — Cost Estimator",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #FFF6CC;
        color: #13294B;
    }

    [data-testid="stSidebar"] {
        background-color: #F4E7A1;
        border-right: 1px solid #D9C97A;
    }

    h1, h2, h3, h4, h5, h6, label, p, span, div {
        color: #13294B;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #13294B;
        margin-bottom: 4px;
    }

    .sub-title {
        font-size: 14px;
        color: #415A77;
        margin-bottom: 16px;
    }

    .section-header {
        font-size: 12px;
        font-weight: 800;
        color: #13294B;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        border-bottom: 2px solid #DFC86D;
        padding-bottom: 8px;
        margin-bottom: 14px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #E9DC94 !important;
        border-radius: 14px;
        padding: 6px;
        gap: 6px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 12px !important;
        padding: 10px 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] span {
        color: #13294B !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #13294B !important;
        border-radius: 12px !important;
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] div,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: #13294B !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 26px !important;
        width: 100% !important;
        min-height: 56px !important;
        font-size: 16px !important;
    }

    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }

    .stDownloadButton > button {
        background: #234A7D !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100% !important;
    }

    .stDownloadButton > button p,
    .stDownloadButton > button span,
    .stDownloadButton > button div {
        color: #FFFFFF !important;
    }

    [data-testid="stMetric"] {
        background: #FFFBE6;
        border: 1px solid #D9C97A;
        border-radius: 14px;
        padding: 10px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    input {
        background-color: #FFFBE6 !important;
        color: #13294B !important;
        border-radius: 10px !important;
    }

    .small-note {
        font-size: 12px;
        color: #415A77 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# GLOBALS / CONSTANTS
# ============================================================
TE_API_KEY = os.getenv("TE_API_KEY", "")
METALS_API_KEY = os.getenv("METALS_API_KEY", "")
FALLBACK_STEEL_RATE = 68.5

REGIONS = [
    "Mumbai MMR", "Thane", "Navi Mumbai", "Pune",
    "Delhi NCR", "Gurgaon NCR", "Bengaluru", "Hyderabad", "Nashik", "Nagpur"
]

STATE_OPTIONS = [
    "Maharashtra", "Delhi", "Haryana", "Karnataka",
    "Telangana", "Tamil Nadu", "Gujarat", "Uttar Pradesh"
]

CATEGORIES = [
    "Mid Income", "Higher Mid Income", "Premium Segment", "Luxury Segment"
]

STAGES = ["Concept", "SD", "DD", "GFC", "Tender"]

STRUCTURES = ["RCC Frame", "Steel Frame", "Load Bearing", "Prefab"]
FACADES = ["Plaster + Paint", "Texture Paint", "Partial Cladding", "Full Glass Facade"]
PARKING = ["Open", "Stilt", "Basement Parking", "MLCP"]
FLOORING = ["Vitrified Basic", "Vitrified Premium", "Italian Marble", "Wooden (Engineered)"]

SOIL_TYPES = [
    "Hard Rock", "Soft Rock", "Murrum",
    "Clay / Black Cotton", "Sandy / Loose Fill",
    "Waterlogged Fill", "Made-up Ground"
]

FLAT_TYPES = ["Studio", "1 RK", "1 BHK", "2 BHK", "2.5 BHK", "3 BHK", "3.5 BHK", "4 BHK", "Penthouse"]

DEFAULT_CARPET = {
    "Studio": 350,
    "1 RK": 400,
    "1 BHK": 450,
    "2 BHK": 650,
    "2.5 BHK": 800,
    "3 BHK": 950,
    "3.5 BHK": 1150,
    "4 BHK": 1400,
    "Penthouse": 2000
}

CITY_INDEX = {
    "Mumbai MMR": 1.45,
    "Thane": 1.20,
    "Navi Mumbai": 1.25,
    "Pune": 1.15,
    "Delhi NCR": 1.30,
    "Gurgaon NCR": 1.32,
    "Bengaluru": 1.20,
    "Hyderabad": 1.10,
    "Nashik": 0.85,
    "Nagpur": 0.80,
}

CAT_INDEX = {
    "Mid Income": 1.00,
    "Higher Mid Income": 1.25,
    "Premium Segment": 1.60,
    "Luxury Segment": 2.20
}

SOIL_MULT = {
    "Hard Rock": 1.00,
    "Soft Rock": 1.05,
    "Murrum": 1.03,
    "Clay / Black Cotton": 1.12,
    "Sandy / Loose Fill": 1.18,
    "Waterlogged Fill": 1.25,
    "Made-up Ground": 1.20
}

MICRO_MARKET_INDEX = {
    "Sewri": 1.08,
    "Worli": 1.18,
    "Lower Parel": 1.20,
    "Andheri West": 1.10,
    "BKC": 1.22,
    "Thane West": 1.05,
    "Navi Mumbai": 1.03,
    "Golf Course Road": 1.18,
    "Dwarka Expressway": 1.10,
    "Noida Expressway": 1.08,
    "Whitefield": 1.08,
    "Sarjapur": 1.06,
    "Hebbal": 1.07,
    "DEFAULT": 1.00,
}

PINCODE_FACTOR = {
    "400": 1.03,
    "401": 1.02,
    "411": 1.01,
    "560": 1.02,
    "122": 1.03,
    "110": 1.02,
    "500": 1.01,
}

BOQ_PCT = {
    "Earthwork & Excavation": 0.030,
    "Basement Works": 0.040,
    "Shell & Core Works": 0.420,
    "Finishing Works": 0.200,
    "MEP Works": 0.110,
    "External Works": 0.055,
    "Other Secondary Structures": 0.050,
    "Contingency & Overheads": 0.095,
}
# ============================================================
# EXCEL-ALIGNED DROPDOWNS
# ============================================================
SECTOR_OPTIONS = [
    "Residential",
    "Commercial",
    "Mixed Use",
    "Township" 
]
MAIN_CATEGORY_HEIGHT_OPTIONS = [
    "Low Rise",
    "Mid Rise",
    "High Rise"
]

SUB_CATEGORY_OPTIONS = [
    "Rehab",
    "Affordable",
    "Mid Premium",
    "Premium",
    "Luxury","Premium(Bareshell)",
     "Luxury(Bareshell)" 
]


PROJECT_STATUS_OPTIONS = [
    "Concept",
    "Ongoing",
    "Completed"
]

COUNTRY_OPTIONS = [
    "India"
]

REGION_OPTIONS = [
    "West",
    "North",
    "South",
    "East",
    "Central"
]

STAGE_OF_BUDGET_OPTIONS = [
    "CD",
    "SD",
    "DD",
    "GFC",
    "Tender"
]

STRUCTURE_OPTIONS = [
    "RCC",
    "Steel",
    "Composite"
]

SHUTTERING_OPTIONS = [
    "Conventional",
    "System",
    "Conventional+System",
    "Aluminium",
    "Jump Form",
    "Tunnel Form"
]

FACADE_SYSTEM_OPTIONS = [
    "System Windows",
    "Curtain Wall",
    "Semi Unitized",
    "Unitized",
    "Stone Cladding",
    "ACP Cladding",
    "Glass + Metal Combination"
]

WINDOW_CONFIGURATION_OPTIONS = [
    "SGU",
    "DGU",
    "Toughened Glass",
    "Laminated Glass"
]

WINDOW_TYPE_OPTIONS = [
    "Sliding",
    "Openable",
    "Casement",
    "Fixed",
    "Combination"
]

DOOR_OPTIONS = [
    "Laminated Door",
    "Flush Door",
    "Metal Door",
    "Engineered Door",
    "Fire Rated Door"
]

EXCAVATION_MODE_OPTIONS = [
    "Normal Excavation",
    "Controlled Blasting",
    "Diamond Cutting",
    "Mechanical Excavation",
    "Manual Excavation"
]

FLOORING_OPTIONS = [
    "Tile",
    "Marble",
    "Wooden Flooring",
    "Kota",
    "Granite",
    "Vitrified Tile",
    "Anti-skid Tile"
]
RISING_MAINS_OPTIONS = [
    "Cu. Cable",
    "Al. Cable"
]

PHE_SYSTEM_OPTIONS = [
    "Single Stack",
    "Double Stack",
    "Single Stack + Vent",
    "Double Stack + Vent"
]

PIPE_MATERIAL_OPTIONS = [
    "UPVC",
    "CPVC",
    "CI",
    "GI",
    "PPR",
    "DWC",
    "SWR"
]

FAPA_SYSTEM_OPTIONS = [
    "Addressable",
    "Conventional"
]

DG_OPTIONS = [
    "Emergency Load",
    "Total Load",
    "Not Applicable"
]

PARKING_OPTIONS = [
    "None",
    "Stack Car Parking",
    "Puzzle Car Parking",
    "Parking Tower"
]

AC_OPTIONS = [
    "Not Applicable",
    "Split AC",
    "VRV",
    "Centralized AC"
]
# ============================================================
# SESSION STATE
# ============================================================
if "result" not in st.session_state:
    st.session_state["result"] = None

if "project_data" not in st.session_state:
    st.session_state["project_data"] = None

# ============================================================
# HELPERS
# ============================================================
def auto_calibration(
    city,
    region,
    micro_market,
    total_floors,
    num_basement,
    num_podium,
    efficiency_ratio,
    quality_level,
    execution_type,
    total_bua,
    category=None
):
    """
    Rule-based smart auto calibration for project estimate.
    Returns an auto-calibration factor and detailed explanation.
    """

    auto_factor = 1.0
    factor_details = {}

    # --------------------------------------------------------
    # 1. Location factor
    # --------------------------------------------------------
    location_factor = 1.0

    premium_cities = ["Mumbai", "Mumbai MMR", "Bengaluru", "Delhi NCR", "Gurgaon NCR"]
    if city in premium_cities:
        location_factor *= 1.05

    if micro_market:
        premium_micro_markets = ["Worli", "BKC", "Lower Parel", "Sewri", "Golf Course Road"]
        if micro_market.strip() in premium_micro_markets:
            location_factor *= 1.03

    factor_details["location_factor"] = round(location_factor, 3)
    auto_factor *= location_factor

    # --------------------------------------------------------
    # 2. Height / floor factor
    # --------------------------------------------------------
    floor_factor = 1.0
    if total_floors > 35:
        floor_factor *= 1.08
    elif total_floors > 20:
        floor_factor *= 1.05
    elif total_floors > 10:
        floor_factor *= 1.02

    factor_details["floor_factor"] = round(floor_factor, 3)
    auto_factor *= floor_factor

    # --------------------------------------------------------
    # 3. Basement / podium complexity
    # --------------------------------------------------------
    basement_podium_factor = 1.0

    if num_basement >= 2:
        basement_podium_factor *= 1.06
    elif num_basement == 1:
        basement_podium_factor *= 1.03

    if num_podium >= 3:
        basement_podium_factor *= 1.04
    elif num_podium >= 1:
        basement_podium_factor *= 1.02

    factor_details["basement_podium_factor"] = round(basement_podium_factor, 3)
    auto_factor *= basement_podium_factor

    # --------------------------------------------------------
    # 4. Efficiency factor
    # --------------------------------------------------------
    efficiency_factor = 1.0
    if efficiency_ratio <= 58:
        efficiency_factor *= 1.08
    elif efficiency_ratio <= 62:
        efficiency_factor *= 1.05
    elif efficiency_ratio <= 68:
        efficiency_factor *= 1.02
    elif efficiency_ratio >= 78:
        efficiency_factor *= 0.97

    factor_details["efficiency_factor"] = round(efficiency_factor, 3)
    auto_factor *= efficiency_factor

    # --------------------------------------------------------
    # 5. Quality / category factor
    # --------------------------------------------------------
    quality_factor_map = {
        "Basic": 0.88,
        "Standard": 0.95,
        "Mid Premium": 1.00,
        "Premium": 1.08,
        "Luxury": 1.18
    }

    quality_factor = quality_factor_map.get(quality_level, 1.0)

    if category is not None:
        if category in ["Mid Income", "Affordable"]:
            quality_factor *= 0.95
        elif category in ["Premium Segment"]:
            quality_factor *= 1.03
        elif category in ["Luxury Segment"]:
            quality_factor *= 1.08

    factor_details["quality_factor"] = round(quality_factor, 3)
    auto_factor *= quality_factor

    # --------------------------------------------------------
    # 6. Execution factor
    # --------------------------------------------------------
    execution_factor_map = {
        "Standard": 1.00,
        "Optimized EPC": 0.92,
        "Value Engineered": 0.85
    }

    execution_factor = execution_factor_map.get(execution_type, 1.0)
    factor_details["execution_factor"] = round(execution_factor, 3)
    auto_factor *= execution_factor

    # --------------------------------------------------------
    # 7. Scale factor
    # --------------------------------------------------------
    scale_factor = 1.0
    if total_bua >= 800000:
        scale_factor *= 0.94
    elif total_bua >= 500000:
        scale_factor *= 0.96
    elif total_bua >= 250000:
        scale_factor *= 0.98

    factor_details["scale_factor"] = round(scale_factor, 3)
    auto_factor *= scale_factor

    # --------------------------------------------------------
    # 8. Safety cap
    # --------------------------------------------------------
    auto_factor = max(0.75, min(auto_factor, 1.25))
    factor_details["final_auto_factor"] = round(auto_factor, 3)

    return {
        "auto_factor": round(auto_factor, 3),
        "factor_details": factor_details,
        "summary": [
            f"Location Factor = {factor_details['location_factor']}×",
            f"Floor Factor = {factor_details['floor_factor']}×",
            f"Basement/Podium Factor = {factor_details['basement_podium_factor']}×",
            f"Efficiency Factor = {factor_details['efficiency_factor']}×",
            f"Quality Factor = {factor_details['quality_factor']}×",
            f"Execution Factor = {factor_details['execution_factor']}×",
            f"Scale Factor = {factor_details['scale_factor']}×",
            f"Final Auto Calibration Factor = {factor_details['final_auto_factor']}×"
        ]
    }
def get_location_factor(region, micro_market, pincode):
    mm_factor = MICRO_MARKET_INDEX.get(str(micro_market).strip(), MICRO_MARKET_INDEX["DEFAULT"])

    pincode_factor = 1.0
    pincode_str = str(pincode).strip()
    if len(pincode_str) >= 3:
        pincode_factor = PINCODE_FACTOR.get(pincode_str[:3], 1.0)

    final_factor = round((mm_factor * 0.70) + (pincode_factor * 0.30), 3)
    return final_factor


def convert_steel_market_to_inr_per_kg(raw_price, usd_inr=83.0):
    if raw_price is None:
        return None
    try:
        return round((float(raw_price) * usd_inr) / 1000, 2)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_steel_te():
    te_key = TE_API_KEY if TE_API_KEY else "guest:guest"
    url = f"https://api.tradingeconomics.com/markets?c={te_key}&f=json"

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        if not isinstance(data, list):
            return None

        for row in data:
            name = str(row.get("Name", "")).lower()
            ticker = str(row.get("Ticker", "")).lower()
            symbol = str(row.get("Symbol", "")).lower()
            last_val = row.get("Last", None)

            if ("steel" in name or "steel" in ticker or "steel" in symbol) and last_val is not None:
                return {
                    "provider": "TradingEconomics",
                    "symbol": row.get("Symbol") or row.get("Ticker") or "N/A",
                    "name": row.get("Name", "Steel"),
                    "raw_price": last_val,
                    "timestamp": row.get("LastUpdate") or row.get("Date") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        return None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_steel_metals(symbol="STEEL-RE"):
    if not METALS_API_KEY:
        return None

    url = f"https://metals-api.com/api/latest?access_key={METALS_API_KEY}&base=USD&symbols={symbol}"

    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return None

        data = r.json()
        rates = data.get("rates", {})
        if symbol not in rates:
            return None

        return {
            "provider": "MetalsAPI",
            "symbol": symbol,
            "name": "Steel Rebar Reference",
            "raw_price": rates[symbol],
            "timestamp": data.get("date", datetime.now().strftime("%Y-%m-%d"))
        }
    except Exception:
        return None


def get_auto_sidebar_rates(region, micro_market, pincode, usd_inr=83.0):
    steel = fetch_live_steel_te()
    if steel is None:
        steel = fetch_live_steel_metals("STEEL-RE")

    if steel:
        steel_inr_kg = convert_steel_market_to_inr_per_kg(steel.get("raw_price"), usd_inr=usd_inr)
        if steel_inr_kg is None:
            steel_inr_kg = FALLBACK_STEEL_RATE
        steel_display = f"₹{steel_inr_kg}/kg"
        steel_source = f"{steel.get('provider', 'Unknown')} ({steel.get('symbol', 'N/A')})"
        updated_at = steel.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        debug_reason = "Live API working or partially working"
    else:
        steel_inr_kg = FALLBACK_STEEL_RATE
        steel_display = f"₹{steel_inr_kg}/kg"
        steel_source = "Fallback benchmark"
        updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        debug_reason = "API key missing or no live response"

    location_factor = get_location_factor(region, micro_market, pincode)

    return {
        "Steel TMT": steel_display,
        "Cement OPC": "N/A",
        "Sand": "N/A",
        "Labour": "N/A",
        "steel_rate_inr_per_kg": steel_inr_kg,
        "steel_source": steel_source,
        "updated_at": updated_at,
        "location_factor": location_factor,
        "debug_reason": debug_reason
    }


@st.cache_resource
def load_forecast_model():
    if os.path.exists(FORECAST_MODEL_PATH) and os.path.exists(FORECAST_META_PATH):
        model = joblib.load(FORECAST_MODEL_PATH)
        metadata = joblib.load(FORECAST_META_PATH)
        return model, metadata
    return None, None


def build_forecast_input(
    city,
    category,
    stage,
    num_towers,
    num_basement,
    num_podium,
    num_typical_floors,
    total_floors,
    fth_tower_m,
    building_height_m,
    total_saleable_sqft,
    total_builtup_sqft,
    total_units,
    sidebar_rates,
    feature_cols
):
    # Height category
    if total_floors <= 10:
        height_category = "low_rise"
        main_category_height = "Low Rise"
    elif total_floors <= 25:
        height_category = "mid_rise"
        main_category_height = "Mid Rise"
    else:
        height_category = "high_rise"
        main_category_height = "High Rise"

    sector = "Residential"
    quality_grade = category
    stage_of_budget = stage

    city_tier_map = {
        "Mumbai": 1,
        "Mumbai MMR": 1,
        "Thane": 1,
        "Navi Mumbai": 1,
        "Pune": 1,
        "Delhi NCR": 1,
        "Gurgaon NCR": 1,
        "Bengaluru": 1,
        "Hyderabad": 1,
        "Nashik": 2,
        "Nagpur": 2
    }
    city_tier = city_tier_map.get(city, 1)

    live_steel = sidebar_rates.get("steel_rate_inr_per_kg", FALLBACK_STEEL_RATE)
    material_index = round(live_steel / FALLBACK_STEEL_RATE, 3) if FALLBACK_STEEL_RATE else 1.0
    labour_index = 1.0

    # Full input dictionary
    input_dict = {
        "city": city,
        "main_category_height": main_category_height,
        "sector": sector,
        "quality_grade": quality_grade,
        "stage_of_budget": stage_of_budget,
        "num_towers": num_towers,
        "num_basement": num_basement,
        "num_podium": num_podium,
        "num_typical_floors": num_typical_floors,
        "total_floors": total_floors,
        "fth_tower_m": fth_tower_m,
        "building_height_m": building_height_m,
        "total_saleable_sqft": total_saleable_sqft,
        "total_builtup_sqft": total_builtup_sqft,
        "total_units": total_units,
        "labour_index": labour_index,
        "material_index": material_index,
        "city_tier": city_tier,
        "height_category": height_category,
        # benchmark-model-only fields defaulted safely
        "cost_earth_works": 0,
        "cost_soil_retention": 0,
        "cost_shell_core": 0,
        "cost_finishing": 0,
        "cost_mep": 0,
        "cost_external_dev": 0,
        "cost_shell_core_psf": 0,
        "cost_finishing_psf": 0,
        "cost_mep_psf": 0,
        "cost_external_psf": 0,
    }

    df = pd.DataFrame([input_dict])

    # Force exact feature order expected by saved model
    if feature_cols:
        for col in feature_cols:
            if col not in df.columns:
                # Safe defaults
                if col in ["city", "main_category_height", "sector", "quality_grade", "stage_of_budget", "height_category"]:
                    df[col] = "Unknown"
                else:
                    df[col] = 0
        df = df.reindex(columns=feature_cols)

    return df
def build_boq_breakdown(total_cost_cr, total_sba, total_bua, category, num_basement):
    boq_breakdown = {}
    pct_total = sum(BOQ_PCT.values())

    for head, pct in BOQ_PCT.items():
        if head == "Basement Works" and num_basement == 0:
            continue

        adjusted_pct = pct
        if head == "Finishing Works":
            category_factor_map = {
                "Mid Income": 1.00,
                "Higher Mid Income": 1.10,
                "Premium Segment": 1.25,
                "Luxury Segment": 1.40,
                "Affordable": 1.00,
                "Mid Premium": 1.10,
                "Premium": 1.25,
                "Luxury": 1.40,
                "Rehab": 0.95
            }
            adjusted_pct = pct * category_factor_map.get(category, 1.0)

        amt_cr = round((adjusted_pct / pct_total) * total_cost_cr, 2)
        cost_sba = round((amt_cr * 1e7) / total_sba, 0) if total_sba else 0
        cost_bua = round((amt_cr * 1e7) / total_bua, 0) if total_bua else 0

        boq_breakdown[head] = {
            "amount_cr": amt_cr,
            "cost_per_sba": int(cost_sba),
            "cost_per_bua": int(cost_bua),
            "pct": round((amt_cr / total_cost_cr) * 100, 1) if total_cost_cr else 0
        }

    return boq_breakdown


def rule_based_estimate(region, category, soil_type, total_bua, total_sba, total_carpet, total_units, total_floors, num_basement, sidebar_rates, micro_market, pincode):
    city_idx = CITY_INDEX.get(region, 1.0)
    cat_idx = CAT_INDEX.get(category, 1.0)
    soil_mult = SOIL_MULT.get(soil_type, 1.0)
    location_factor = get_location_factor(region, micro_market, pincode)

    if total_floors > 30:
        floor_idx = 1.18
    elif total_floors > 15:
        floor_idx = 1.08
    else:
        floor_idx = 1.00

    base_psf = 3800
    live_steel = sidebar_rates.get("steel_rate_inr_per_kg", FALLBACK_STEEL_RATE)
    steel_index = (live_steel / FALLBACK_STEEL_RATE) if FALLBACK_STEEL_RATE else 1.0
    live_rate_index = round((0.70 * 1.0) + (0.30 * steel_index), 3)
    dynamic_base_psf = base_psf * live_rate_index

    total_cost_inr = (
        total_bua
        * dynamic_base_psf
        * city_idx
        * cat_idx
        * floor_idx
        * soil_mult
        * location_factor
    )
    total_cost_cr = round(total_cost_inr / 1e7, 2)

    boq_breakdown = build_boq_breakdown(
        total_cost_cr=total_cost_cr,
        total_sba=total_sba,
        total_bua=total_bua,
        category=category,
        num_basement=num_basement
    )

    return {
        "total_cost_cr": total_cost_cr,
        "range_low_cr": round(total_cost_cr * 0.90, 2),
        "range_high_cr": round(total_cost_cr * 1.10, 2),
        "cost_per_sba": int(total_cost_inr / total_sba) if total_sba else 0,
        "cost_per_bua": int(total_cost_inr / total_bua) if total_bua else 0,
        "total_bua_sqft": total_bua,
        "total_sba_sqft": total_sba,
        "total_carpet_sqft": total_carpet,
        "total_units": total_units,
        "boq_breakdown": boq_breakdown,
        "assumptions": [
            "Rule-based estimate used.",
            f"Dynamic base PSF = ₹{round(dynamic_base_psf, 0):,}",
            f"City index ({region}) = {city_idx}×",
            f"Category multiplier ({category}) = {cat_idx}×",
            f"Soil multiplier ({soil_type}) = {soil_mult}×",
            f"Location factor ({micro_market or 'Standard'} / {pincode or '-'}) = {location_factor}×",
            f"Floor multiplier ({total_floors} floors) = {floor_idx}×",
        ],
        "scope_flags": [],
        "location_factor": location_factor,
        "live_rate_index": live_rate_index,
        "dynamic_base_psf": int(dynamic_base_psf),
    }
def predict_estimate(
    forecast_model, forecast_meta,
    city, category, stage,
    num_buildings, num_basement, num_podium, num_tower, total_floors,
    flr_height, total_sba, total_bua, total_units,
    region, micro_market, pincode, soil_type, total_carpet,
    sidebar_rates
):
    # If model exists, use it
    if forecast_model is not None and forecast_meta is not None:
        feature_cols = forecast_meta.get("feature_cols") or forecast_meta.get("best_feature_cols") or []

        input_df = build_forecast_input(
            city=city,
            category=category,
            stage=stage,
            num_towers=num_buildings,
            num_basement=num_basement,
            num_podium=num_podium,
            num_typical_floors=num_tower,
            total_floors=total_floors,
            fth_tower_m=flr_height,
            building_height_m=round(num_tower * flr_height, 1),
            total_saleable_sqft=total_sba,
            total_builtup_sqft=total_bua,
            total_units=total_units,
            sidebar_rates=sidebar_rates,
            feature_cols=feature_cols
        )

        try:
            predicted_psf = float(forecast_model.predict(input_df)[0])
            location_factor_val = get_location_factor(region, micro_market, pincode)
            adjusted_psf = predicted_psf * location_factor_val
            predicted_total_cost = adjusted_psf * total_bua
            predicted_total_cost_cr = round(predicted_total_cost / 1e7, 2)

            boq_breakdown = build_boq_breakdown(
                total_cost_cr=predicted_total_cost_cr,
                total_sba=total_sba,
                total_bua=total_bua,
                category=category,
                num_basement=num_basement
            )

            return {
                "total_cost_cr": predicted_total_cost_cr,
                "range_low_cr": round(predicted_total_cost_cr * 0.90, 2),
                "range_high_cr": round(predicted_total_cost_cr * 1.10, 2),
                "cost_per_sba": int(predicted_total_cost / total_sba) if total_sba else 0,
                "cost_per_bua": int(adjusted_psf),
                "total_bua_sqft": total_bua,
                "total_sba_sqft": total_sba,
                "total_carpet_sqft": total_carpet,
                "total_units": total_units,
                "boq_breakdown": boq_breakdown,
                "assumptions": [
                    "Forecast model used.",
                    f"Model base PSF = ₹{round(predicted_psf, 0):,}",
                    f"Location Factor = {location_factor_val}×"
                ],
                "scope_flags": [],
                "location_factor": location_factor_val,
                "live_rate_index": sidebar_rates.get("steel_rate_inr_per_kg", None),
                "dynamic_base_psf": int(adjusted_psf),
            }
        except Exception:
            pass

    # Fallback
    return rule_based_estimate(
        region=region,
        category=category,
        soil_type=soil_type,
        total_bua=total_bua,
        total_sba=total_sba,
        total_carpet=total_carpet,
        total_units=total_units,
        total_floors=total_floors,
        num_basement=num_basement,
        sidebar_rates=sidebar_rates,
        micro_market=micro_market,
        pincode=pincode
    )

def save_past_estimate(project_data: dict, result: dict):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_name": project_data.get("project_name", ""),
        "client_name": project_data.get("client_name", ""),
        "region": project_data.get("region", ""),
        "state": project_data.get("state", ""),
        "city": project_data.get("city", ""),
        "micro_market": project_data.get("micro_market", ""),
        "pincode": project_data.get("pincode", ""),
        "category": project_data.get("category", ""),
        "stage": project_data.get("stage", ""),
        "soil_type": project_data.get("soil_type", ""),
        "total_units": project_data.get("total_units", 0),
        "total_carpet": project_data.get("total_carpet", 0),
        "total_cost_cr": result.get("total_cost_cr", 0),
        "cost_per_sba": result.get("cost_per_sba", 0),
        "cost_per_bua": result.get("cost_per_bua", 0),
        "location_factor": result.get("location_factor", 1.0),
        "live_rate_index": result.get("live_rate_index", 1.0),
        "dynamic_base_psf": result.get("dynamic_base_psf", 3800),
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(PAST_ESTIMATES_FILE):
        df_old = pd.read_csv(PAST_ESTIMATES_FILE)
        df_new = pd.concat([df_old, df_row], ignore_index=True)
    else:
        df_new = df_row

    df_new.to_csv(PAST_ESTIMATES_FILE, index=False)


def generate_estimate_pdf(project_data: dict, result: dict):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    pdf.setTitle("SmartBuild AI Estimate")

    # HEADER
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "SmartBuild AI — Cost Estimate")
    y -= 28

    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generated On: {datetime.now().strftime('%d-%b-%Y %H:%M')}")
    y -= 20

    # PROJECT SUMMARY
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Project Summary")
    y -= 16

    pdf.setFont("Helvetica", 10)
    lines = [
        f"Project Name: {project_data.get('project_name', '-')}",
        f"Client Name: {project_data.get('client_name', '-')}",
        f"Region: {project_data.get('region', '-')}",
        f"State: {project_data.get('state', '-')}",
        f"City: {project_data.get('city', '-')}",
        f"Micro Market: {project_data.get('micro_market', '-')}",
        f"Pincode: {project_data.get('pincode', '-')}",
        f"Category: {project_data.get('category', '-')}",
        f"Stage: {project_data.get('stage', '-')}",
        f"Soil Type: {project_data.get('soil_type', '-')}",
        f"Total Units: {project_data.get('total_units', 0)}",
        f"Total Carpet Area: {project_data.get('total_carpet', 0):,} sqft",
        f"Total SBA: {result.get('total_sba_sqft', 0):,} sqft",
        f"Total BUA: {result.get('total_bua_sqft', 0):,} sqft",
        f"Location Factor: {result.get('location_factor', 1.0)}×",
        f"Live Rate Index: {result.get('live_rate_index', 1.0)}×",
        f"Dynamic Base PSF: ₹{result.get('dynamic_base_psf', 0):,}",
    ]

    for line in lines:
        pdf.drawString(50, y, line[:110])
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    # ESTIMATE RESULT
    y -= 8
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Estimate Result")
    y -= 18

    pdf.setFont("Helvetica", 10)
    result_lines = [
        f"Total Estimated Cost: ₹{result.get('total_cost_cr', 0)} Cr",
        f"Cost / SBA Sqft: ₹{result.get('cost_per_sba', 0):,}",
        f"Cost / BUA Sqft: ₹{result.get('cost_per_bua', 0):,}",
        f"Range: ₹{result.get('range_low_cr', 0)} Cr – ₹{result.get('range_high_cr', 0)} Cr",
    ]

    for line in result_lines:
        pdf.drawString(50, y, line[:110])
        y -= 14
        if y < 80:
            pdf.showPage()
            y = height - 50
            pdf.setFont("Helvetica", 10)

    # BOQ BREAKDOWN
    y -= 8
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "BOQ Breakdown")
    y -= 18

    pdf.setFont("Helvetica", 9)
    boq_data = result.get("boq_breakdown", {})
    if boq_data:
        for head, vals in boq_data.items():
            text = (
                f"{head}: ₹{vals.get('amount_cr', 0)} Cr | "
                f"₹{vals.get('cost_per_sba', 0):,}/SBA | "
                f"₹{vals.get('cost_per_bua', 0):,}/BUA"
            )
            pdf.drawString(50, y, text[:110])
            y -= 12
            if y < 80:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 9)
    else:
        pdf.drawString(50, y, "BOQ breakdown not available.")
        y -= 12

    # ASSUMPTIONS
    y -= 8
    if y < 80:
        pdf.showPage()
        y = height - 50

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Assumptions")
    y -= 18

    pdf.setFont("Helvetica", 9)
    assumptions = result.get("assumptions", [])
    if assumptions:
        for a in assumptions:
            pdf.drawString(50, y, f"- {str(a)[:110]}")
            y -= 12
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 9)
    else:
        pdf.drawString(50, y, "- No assumptions available.")
        y -= 12

    # SCOPE FLAGS
    scope_flags = result.get("scope_flags", [])
    if scope_flags:
        y -= 8
        if y < 80:
            pdf.showPage()
            y = height - 50

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "Scope / Risk Flags")
        y -= 18

        pdf.setFont("Helvetica", 9)
        for flag in scope_flags:
            pdf.drawString(50, y, f"- {str(flag)[:110]}")
            y -= 12
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont("Helvetica", 9)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()

def build_mep_assumptions(
    project_name="",
    client_name="",
    region="",
    state="",
    city="",
    micro_market="",
    pincode="",
    category="",
    stage="",
    total_units=0,
    total_carpet=0,
    total_bua=0,
    total_sba=0,
    num_basement=0,
    num_ground=0,
    num_podium=0,
    num_typical_floors=0,
    total_floors=0,
    flr_height=0.0,
    no_of_car_parks=0,
    passenger_lift_count=0,
    passenger_lift_speed=0.0,
    passenger_lift_stops=0,
    passenger_lift_capacity=0,
    service_lift_count=0,
    service_lift_speed=0.0,
    service_lift_stops=0,
    service_lift_capacity=0,
    fire_lift_count=0,
    fire_lift_speed=0.0,
    fire_lift_stops=0,
    fire_lift_capacity=0,
    car_lift_count=0,
    car_lift_speed=0.0,
    car_lift_stops=0,
    car_lift_capacity=0,
    ev_charging_pct=5,
    electrical_ht="Yes",
    rising_mains="Cu. Cable",
    phe_system_type="Double Stack",
    fapa_type="Addressable",
    water_supply_internal="CPVC",
    water_supply_shaft="UPVC",
    drainage_internal="DWC",
    drainage_shaft="DWC",
    external_pipe_material="UPVC",
    sprinklers_inside_apartments="Yes",
    fire_alarm_inside_apartments="Yes",
    public_address_inside_apartments="Yes",
    av_required="No",
    it_required="No",
    security_required="No",
    acs_required="Yes",
    cctv_required="Yes",
    bms_required="Yes",
    stp_required="Yes",
    wtp_required="Yes",
    owc_required="Yes",
    solar_electric_required="No",
    solar_water_required="Yes",
    ftth_required="Yes",
    ac_inside_apartments="Not Applicable",
    ac_inside_lobbies="Yes",
    mechanical_parking_type="Stack Car Parking",
    piped_gas_required="No",
    dg_type="Emergency Load"
):
    # ----------------------------------------------------
    # BASIS VALUES (reference / workbook logic)
    # ----------------------------------------------------
    transformer_count = 2
    transformer_capacity_kva_each = 800
    transformer_basis_note = "Standard size as per Voltamp make"

    dg_count = 3
    dg_capacity_kva_each = 500
    dg_basis_note = "Standard capacity as per Greaves make"

    stp_kld_each = 130
    stp_count = 2
    wtp_kld = 170
    owc_kg = 200

    vrv_odu_per_floor = 4
    vrv_odu_hp_each = 20
    vrv_total_tr_reference = 720

    wifi_provision_inr = 500000
    ftth_provision_inr = 1000000
    access_control_provision_inr = 1000000
    bms_provision_inr = 1000000
    fire_suppression_bms_room_inr = 400000
    visitor_management_provision_inr = 200000

    geyser_toilet_ltr = 3
    geyser_kitchen_ltr = 6
    solar_hot_water_ltr = 20000
    solar_electric_kw = 13

    # ----------------------------------------------------
    # DERIVED VALUES (current run)
    # ----------------------------------------------------
    total_units = int(total_units) if total_units else 0
    total_carpet = float(total_carpet) if total_carpet else 0.0
    total_bua = float(total_bua) if total_bua else 0.0
    total_sba = float(total_sba) if total_sba else 0.0

    total_transformer_capacity_kva = transformer_count * transformer_capacity_kva_each
    total_dg_capacity_kva = dg_count * dg_capacity_kva_each
    total_stp_kld = stp_kld_each * stp_count
    considered_ev_points = round((float(no_of_car_parks) * float(ev_charging_pct)) / 100, 0) if no_of_car_parks else 0
    total_vrv_odu_count = int(vrv_odu_per_floor * int(total_floors)) if total_floors else 0
    total_vrv_hp = int(total_vrv_odu_count * vrv_odu_hp_each) if total_vrv_odu_count else 0

    scope_parts = []
    if int(num_basement) > 0:
        scope_parts.append(f"{int(num_basement)} Basement")
    if int(num_ground) > 0:
        scope_parts.append(f"{int(num_ground)} Ground")
    if int(num_podium) > 0:
        scope_parts.append(f"{int(num_podium)} Podium")
    if int(num_typical_floors) > 0:
        scope_parts.append(f"{int(num_typical_floors)} Typical Floors")
    scope_parts.append("Roof")
    scope_text = " + ".join(scope_parts)

    return {
        "project_context": {
            "project_name": project_name or "",
            "client_name": client_name or "",
            "region": region or "",
            "state": state or "",
            "city": city or "",
            "micro_market": micro_market or "",
            "pincode": pincode or "",
            "category": category or "",
            "stage": stage or "",
            "scope_text": scope_text,
            "total_units": total_units,
            "total_carpet_sqft": round(total_carpet, 2),
            "total_bua_sqft": round(total_bua, 2),
            "total_sba_sqft": round(total_sba, 2),
            "num_basement": int(num_basement) if num_basement else 0,
            "num_ground": int(num_ground) if num_ground else 0,
            "num_podium": int(num_podium) if num_podium else 0,
            "num_typical_floors": int(num_typical_floors) if num_typical_floors else 0,
            "total_floors": int(total_floors) if total_floors else 0,
            "floor_to_floor_height_m": float(flr_height) if flr_height else 0.0,
            "car_parks": int(no_of_car_parks) if no_of_car_parks else 0
        },

        "basis": {
            "electrical": {
                "electrical_ht": electrical_ht,
                "rising_mains": rising_mains,
                "transformer_count": transformer_count,
                "transformer_capacity_kva_each": transformer_capacity_kva_each,
                "transformer_basis_note": transformer_basis_note,
                "dg_count": dg_count,
                "dg_capacity_kva_each": dg_capacity_kva_each,
                "dg_basis_note": dg_basis_note,
                "wifi_provision_inr": wifi_provision_inr,
                "ftth_provision_inr": ftth_provision_inr,
                "access_control_provision_inr": access_control_provision_inr,
                "bms_provision_inr": bms_provision_inr,
                "fire_suppression_bms_room_inr": fire_suppression_bms_room_inr,
                "visitor_management_provision_inr": visitor_management_provision_inr,
                "ev_charging_pct": ev_charging_pct,
                "dg_type": dg_type
            },

            "plumbing": {
                "phe_system_type": phe_system_type,
                "water_supply_internal": water_supply_internal,
                "water_supply_shaft": water_supply_shaft,
                "drainage_internal": drainage_internal,
                "drainage_shaft": drainage_shaft,
                "external_pipe_material": external_pipe_material,
                "stp_required": stp_required,
                "stp_kld_each": stp_kld_each,
                "stp_count": stp_count,
                "wtp_required": wtp_required,
                "wtp_kld": wtp_kld,
                "owc_required": owc_required,
                "owc_kg": owc_kg
            },

            "fire_fighting": {
                "sprinklers_inside_apartments": sprinklers_inside_apartments,
                "fire_alarm_inside_apartments": fire_alarm_inside_apartments,
                "public_address_inside_apartments": public_address_inside_apartments,
                "fapa_type": fapa_type
            },

            "elv_security": {
                "av_required": av_required,
                "it_required": it_required,
                "security_required": security_required,
                "acs_required": acs_required,
                "cctv_required": cctv_required,
                "bms_required": bms_required
            },

            "elevators": {
                "passenger_lift_count": int(passenger_lift_count) if passenger_lift_count else 0,
                "passenger_lift_speed": float(passenger_lift_speed) if passenger_lift_speed else 0.0,
                "passenger_lift_stops": int(passenger_lift_stops) if passenger_lift_stops else 0,
                "passenger_lift_capacity": int(passenger_lift_capacity) if passenger_lift_capacity else 0,

                "service_lift_count": int(service_lift_count) if service_lift_count else 0,
                "service_lift_speed": float(service_lift_speed) if service_lift_speed else 0.0,
                "service_lift_stops": int(service_lift_stops) if service_lift_stops else 0,
                "service_lift_capacity": int(service_lift_capacity) if service_lift_capacity else 0,

                "fire_lift_count": int(fire_lift_count) if fire_lift_count else 0,
                "fire_lift_speed": float(fire_lift_speed) if fire_lift_speed else 0.0,
                "fire_lift_stops": int(fire_lift_stops) if fire_lift_stops else 0,
                "fire_lift_capacity": int(fire_lift_capacity) if fire_lift_capacity else 0,

                "car_lift_count": int(car_lift_count) if car_lift_count else 0,
                "car_lift_speed": float(car_lift_speed) if car_lift_speed else 0.0,
                "car_lift_stops": int(car_lift_stops) if car_lift_stops else 0,
                "car_lift_capacity": int(car_lift_capacity) if car_lift_capacity else 0
            },

            "hvac": {
                "ac_inside_apartments": ac_inside_apartments,
                "ac_inside_lobbies": ac_inside_lobbies,
                "vrv_odu_per_floor": vrv_odu_per_floor,
                "vrv_odu_hp_each": vrv_odu_hp_each,
                "vrv_total_tr_reference": vrv_total_tr_reference
            },

            "other_systems": {
                "solar_electric_required": solar_electric_required,
                "solar_water_required": solar_water_required,
                "solar_hot_water_ltr": solar_hot_water_ltr,
                "solar_electric_kw": solar_electric_kw,
                "ftth_required": ftth_required,
                "mechanical_parking_type": mechanical_parking_type,
                "piped_gas_required": piped_gas_required,
                "geyser_toilet_ltr": geyser_toilet_ltr,
                "geyser_kitchen_ltr": geyser_kitchen_ltr
            }
        },

        "derived": {
            "total_transformer_capacity_kva": int(total_transformer_capacity_kva),
            "total_dg_capacity_kva": int(total_dg_capacity_kva),
            "total_stp_kld": int(total_stp_kld),
            "considered_ev_points": int(considered_ev_points),
            "total_vrv_odu_count": int(total_vrv_odu_count),
            "total_vrv_hp": int(total_vrv_hp)
        }
    }

def flatten_mep_assumptions_for_display(mep_assumptions):
    lines = []

    if not mep_assumptions:
        return ["No MEP assumptions available."]

    project_meta = mep_assumptions.get("project_meta", {})
    electrical = mep_assumptions.get("electrical", {})
    elv_security = mep_assumptions.get("elv_security", {})
    plumbing = mep_assumptions.get("plumbing", {})
    fire_fighting = mep_assumptions.get("fire_fighting", {})
    elevator_works = mep_assumptions.get("elevator_works", {})
    hvac = mep_assumptions.get("hvac", {})
    other_systems = mep_assumptions.get("other_systems", {})

def build_civil_assumptions(
    project_name="",
    client_name="",
    region="",
    state="",
    city="",
    micro_market="",
    pincode="",
    category="",
    stage="",
    soil_type="",
    num_buildings=0,
    num_basement=0,
    num_ground=0,
    num_podium=0,
    num_typical_floors=0,
    total_floors=0,
    flr_height=0.0,
    structure_type="",
    shuttering_type="",
    total_units=0,
    total_carpet=0,
    total_bua=0,
    total_sba=0,
    excavation_mode="",
    excavation_soil_pct=0.0,
    excavation_hardrock_pct=0.0,
    excavation_softrock_pct=0.0,
    depth_of_excavation=0.0
):
    # ----------------------------------------------------
    # BASIS VALUES (reference / workbook logic)
    # ----------------------------------------------------
    cement_type = "43/53 grade OPC / PPC"
    cement_rate_per_bag = 230

    steel_type = "Secondary steel Fe 500"
    steel_rate_per_mt = 36000

    concrete_index_cum_per_sqm = 0.47
    steel_index_kg_per_sqft = 4.5
    shuttering_index_sqm_per_cum = 6.14

    cement_wastage_pct = 3
    rmc_wastage_pct = 2
    reinforcement_steel_wastage_pct = 5
    tile_flooring_wastage_pct = 7
    tile_dado_wastage_pct = 7
    marble_granite_stone_wastage_pct = 20

    foundation_basis = "Tower - Isolated foundations"
    structure_basis = "Conventional beam-slab structure with core walls, shear walls and link beams"
    shuttering_sub_structure = "Conventional shuttering"
    shuttering_super_structure = "Mivan formwork"

    masonry_external = "RCC & NSW WALL"
    masonry_internal = "Blockwork & Drywall layout"

    plaster_external = "External plaster & paint; single coat duct plaster"
    plaster_internal = "Gypsum plaster to walls"
    plaster_back_coat = "Back coat plaster in CM behind dado"

    wp_raft = "Membrane Waterproofing"
    wp_terrace = "Conventional Brick Bat Coba"
    wp_toilet = "Chemical Waterproofing + 50mm Protective Screed"
    wp_refuge = "Chemical Waterproofing + 50mm Protective Screed"

    room_flooring = "Vitrified Tile"
    toilet_flooring = "Antiskid Ceramic Tile"
    toilet_dado = "Ceramic Tile"
    wash_basin_counter = "Granite 600mm wide"

    room_wall_finish = "Gypsum plaster"
    toilet_wall_finish = "Back coat plaster up to dado + gypsum plaster above dado"

    room_paint = "Acrylic Emulsion Paint"
    toilet_paint = "Lime Wash above dado"
    staircase_paint = "OBD Paint"

    typical_lobby_flooring = "Vitrified"
    ground_floor_lobby_flooring = "Granite"
    staircase_flooring = "Kota / Concrete Finish"
    common_ceiling_finish = "Gypsum False Ceiling / Putty / Cement Paint"
    common_painting = "AEP / OBD / Cement Paint"

    façade_room_windows = "Aluminium Powder Coated Window with MS Grill Support"
    façade_common_windows = "Aluminium Powder Coated Window with MS Grill Support"
    façade_finish = "Exterior Grade Paint"
    shaft_basis = "Shaft Louvers"
    special_facade = "Low-E Glass Curtain Wall with Stone / Metal Cladding"

    furniture_cost_included_inr = 43576600
    kitchen_equipment_provisional_inr = 12500000

    # ----------------------------------------------------
    # DERIVED VALUES (current run)
    # ----------------------------------------------------
    total_units = int(total_units) if total_units else 0
    total_carpet = float(total_carpet) if total_carpet else 0.0
    total_bua = float(total_bua) if total_bua else 0.0
    total_sba = float(total_sba) if total_sba else 0.0

    total_carpet_sqm = round(total_carpet / 10.764, 2) if total_carpet > 0 else 0.0
    total_bua_sqm = round(total_bua / 10.764, 2) if total_bua > 0 else 0.0
    total_sba_sqm = round(total_sba / 10.764, 2) if total_sba > 0 else 0.0

    concrete_qty_cum_base = total_sba_sqm * concrete_index_cum_per_sqm
    steel_qty_kg_base = total_sba * steel_index_kg_per_sqft
    steel_qty_mt_base = steel_qty_kg_base / 1000 if steel_qty_kg_base > 0 else 0.0
    shuttering_qty_sqm_base = concrete_qty_cum_base * shuttering_index_sqm_per_cum

    concrete_qty_cum_with_wastage = concrete_qty_cum_base * (1 + (rmc_wastage_pct / 100))
    steel_qty_kg_with_wastage = steel_qty_kg_base * (1 + (reinforcement_steel_wastage_pct / 100))
    steel_qty_mt_with_wastage = steel_qty_kg_with_wastage / 1000 if steel_qty_kg_with_wastage > 0 else 0.0
    shuttering_qty_sqm_with_wastage = shuttering_qty_sqm_base

    return {
        "project_context": {
            "project_name": project_name or "",
            "client_name": client_name or "",
            "region": region or "",
            "state": state or "",
            "city": city or "",
            "micro_market": micro_market or "",
            "pincode": pincode or "",
            "category": category or "",
            "stage": stage or "",
            "soil_type": soil_type or "",
            "num_buildings": int(num_buildings) if num_buildings else 0,
            "num_basement": int(num_basement) if num_basement else 0,
            "num_ground": int(num_ground) if num_ground else 0,
            "num_podium": int(num_podium) if num_podium else 0,
            "num_typical_floors": int(num_typical_floors) if num_typical_floors else 0,
            "total_floors": int(total_floors) if total_floors else 0,
            "floor_to_floor_height_m": float(flr_height) if flr_height else 0.0,
            "structure_type": structure_type or "",
            "shuttering_type": shuttering_type or "",
            "total_units": total_units,
            "total_carpet_sqft": round(total_carpet, 2),
            "total_bua_sqft": round(total_bua, 2),
            "total_sba_sqft": round(total_sba, 2),
            "total_carpet_sqm": total_carpet_sqm,
            "total_bua_sqm": total_bua_sqm,
            "total_sba_sqm": total_sba_sqm
        },

        "basis": {
            "basic_rates": {
                "cement_type": cement_type,
                "cement_rate_per_bag": cement_rate_per_bag,
                "steel_type": steel_type,
                "steel_rate_per_mt": steel_rate_per_mt
            },

            "material_wastage": {
                "cement_pct": cement_wastage_pct,
                "rmc_pct": rmc_wastage_pct,
                "reinforcement_steel_pct": reinforcement_steel_wastage_pct,
                "tile_flooring_pct": tile_flooring_wastage_pct,
                "tile_dado_pct": tile_dado_wastage_pct,
                "marble_granite_stone_pct": marble_granite_stone_wastage_pct
            },

            "rcc_indices": {
                "concrete_cum_per_sqm": concrete_index_cum_per_sqm,
                "steel_kg_per_sqft": steel_index_kg_per_sqft,
                "shuttering_sqm_per_cum": shuttering_index_sqm_per_cum
            },

            "shell_core_basis": {
                "foundation_type": foundation_basis,
                "structure_basis": structure_basis,
                "shuttering_sub_structure": shuttering_sub_structure,
                "shuttering_super_structure": shuttering_super_structure
            },

            "earthwork_basis": {
                "excavation_mode": excavation_mode or "",
                "excavation_soil_pct": float(excavation_soil_pct) if excavation_soil_pct else 0.0,
                "excavation_hardrock_pct": float(excavation_hardrock_pct) if excavation_hardrock_pct else 0.0,
                "excavation_softrock_pct": float(excavation_softrock_pct) if excavation_softrock_pct else 0.0,
                "depth_of_excavation_m": float(depth_of_excavation) if depth_of_excavation else 0.0
            },

            "masonry_plaster_waterproofing": {
                "external_walls": masonry_external,
                "internal_walls": masonry_internal,
                "external_plaster": plaster_external,
                "internal_plaster": plaster_internal,
                "back_coat": plaster_back_coat,
                "raft_wp": wp_raft,
                "terrace_wp": wp_terrace,
                "toilet_wp": wp_toilet,
                "refuge_wp": wp_refuge
            },

            "internal_finishes": {
                "room_flooring": room_flooring,
                "toilet_flooring": toilet_flooring,
                "toilet_dado": toilet_dado,
                "wash_basin_counter": wash_basin_counter,
                "room_wall_finish": room_wall_finish,
                "toilet_wall_finish": toilet_wall_finish,
                "room_paint": room_paint,
                "toilet_paint": toilet_paint,
                "staircase_paint": staircase_paint
            },

            "common_area_finishes": {
                "typical_lobby_flooring": typical_lobby_flooring,
                "ground_floor_lobby_flooring": ground_floor_lobby_flooring,
                "staircase_flooring": staircase_flooring,
                "ceiling_finish": common_ceiling_finish,
                "painting": common_painting
            },

            "facade_external_basis": {
                "room_windows": façade_room_windows,
                "common_windows": façade_common_windows,
                "facade_finish": façade_finish,
                "shaft_basis": shaft_basis,
                "special_facade": special_facade,
                "furniture_cost_included_inr": furniture_cost_included_inr,
                "kitchen_equipment_provisional_inr": kitchen_equipment_provisional_inr
            }
        },

        "derived": {
            "construction_area_sqft": round(total_sba, 2),
            "construction_area_sqm": round(total_sba_sqm, 2),
            "concrete_qty_cum_base": round(concrete_qty_cum_base, 2),
            "concrete_qty_cum_with_wastage": round(concrete_qty_cum_with_wastage, 2),
            "steel_qty_kg_base": round(steel_qty_kg_base, 2),
            "steel_qty_mt_base": round(steel_qty_mt_base, 2),
            "steel_qty_kg_with_wastage": round(steel_qty_kg_with_wastage, 2),
            "steel_qty_mt_with_wastage": round(steel_qty_mt_with_wastage, 2),
            "shuttering_qty_sqm_base": round(shuttering_qty_sqm_base, 2),
            "shuttering_qty_sqm_with_wastage": round(shuttering_qty_sqm_with_wastage, 2)
        }
    }


    # =====================================================
    # PROJECT / SCOPE
    # =====================================================
    lines.append("MEP ASSUMPTIONS")
    lines.append(f"Scope = {project_meta.get('scope', 'Not defined')}")

    # =====================================================
    # ELECTRICAL
    # =====================================================
    lines.append("")
    lines.append("ELECTRICAL")

    hs = electrical.get("high_side", {})
    ls = electrical.get("low_side", {})

    if hs:
        lines.append("High Side")
        if "supply_authority_to_substation_cabling" in hs:
            lines.append(f"Supply Authority to Substation Cabling = {hs.get('supply_authority_to_substation_cabling', {}).get('value', 'Not defined')}")
        if "ht_panel" in hs:
            lines.append(f"HT Panel = {hs.get('ht_panel', {}).get('value', 'Not defined')}")
        if "ht_cable_aluminum" in hs:
            lines.append(f"HT Cable (Aluminum) = {hs.get('ht_cable_aluminum', {}).get('value', 'Not defined')}")

        transformers = hs.get("transformers", {})
        if transformers:
            lines.append(
                f"Transformers = {transformers.get('count', 0)} x "
                f"{transformers.get('capacity_kva_each', 0)} KVA "
                f"({transformers.get('make_note', '-')})"
            )

        dg = hs.get("dg", {})
        if dg:
            lines.append(
                f"DG = {dg.get('count', 0)} x "
                f"{dg.get('capacity_kva_each', 0)} KVA "
                f"({dg.get('make_note', '-')})"
            )

    if ls:
        lines.append("Low Side")
        if "lt_xlpe_cables_panels" in ls:
            lines.append(f"LT XLPE Cables & Panels = {ls.get('lt_xlpe_cables_panels', {}).get('value', 'Not defined')}")

        db = ls.get("distribution_boards", {})
        if db:
            lines.append(
                f"Distribution Boards = {db.get('type_1', '-')} | {db.get('type_2', '-')}"
            )

        if "external_light_fittings" in ls:
            lines.append(f"External Light Fittings = {ls.get('external_light_fittings', {}).get('value', 'Not defined')}")

        wiring = ls.get("common_area_wiring", {})
        if wiring:
            lines.append(
                "Common Area Wiring = "
                f"Light Points: {wiring.get('light_points', '-')} | "
                f"Power Sockets: {wiring.get('power_sockets', '-')} | "
                f"Geyser/AC Points: {wiring.get('geyser_ac_points', '-')}"
            )

        fixtures = ls.get("light_fixtures", {})
        if fixtures:
            lines.append(
                "Light Fixtures = "
                f"Lift Lobby ₹{fixtures.get('lift_lobby_inr', 0)} | "
                f"Staircase ₹{fixtures.get('staircase_inr', 0)} | "
                f"Parking/Refuge/Terrace ₹{fixtures.get('parking_refuge_terrace_inr', 0)} | "
                f"Rooms ₹{fixtures.get('rooms_inr', 0)}"
            )

        if "light_power_points_flat" in ls:
            lines.append(f"Light & Power Points / Flat = {ls.get('light_power_points_flat', {}).get('value', 'Not defined')}")

        dtt = ls.get("data_tv_telephone", {})
        if dtt:
            lines.append(
                "Data / TV / Telephone = "
                f"Offices: {dtt.get('offices', '-')} | "
                f"TV: {dtt.get('tv_points', '-')} | "
                f"Rooms: {dtt.get('rooms', '-')} | "
                f"FTTH Provision ₹{dtt.get('ftth_provision_inr', 0):,}"
            )

        if "intercom_vdp" in ls:
            lines.append(f"Intercom / VDP = {ls.get('intercom_vdp', {}).get('value', 'Not defined')}")

        if "wifi_system" in ls:
            lines.append(f"Wi-Fi Provision = ₹{ls.get('wifi_system', {}).get('provision_inr', 0):,}")

        if "earthing_system" in ls:
            lines.append(f"Earthing System = {ls.get('earthing_system', {}).get('value', 'Not defined')}")

        if "lightning_protection" in ls:
            lines.append(f"Lightning Protection = {ls.get('lightning_protection', {}).get('value', 'Not defined')}")

        if "cable_trays" in ls:
            lines.append(f"Cable Trays = {ls.get('cable_trays', {}).get('value', 'Not defined')}")

        ups = ls.get("ups_with_battery", {})
        if ups:
            lines.append(
                f"UPS with Battery = {ups.get('capacity_kva', 0)} kVA for {ups.get('scope', '-')}"
            )

        car = ls.get("car_charging_point", {})
        if car:
            lines.append(
                f"Car Charging = {car.get('basis', '-')} | "
                f"Reference Car Parks: {car.get('reference_car_parks', 0)} | "
                f"Considered Points: {car.get('considered_points', 0)}"
            )
    
    # =====================================================
    # ELV / SECURITY
    # =====================================================
    if elv_security:
        lines.append("")
        lines.append("ELV & SECURITY")

        if "electronic_car_parking_monitoring" in elv_security:
            lines.append(f"Electronic Car Parking & Monitoring System = {elv_security.get('electronic_car_parking_monitoring', {}).get('value', 'Not defined')}")

        cctv = elv_security.get("cctv_system", {})
        if cctv:
            lines.append(f"CCTV System = {cctv.get('indoor', '-')} | Outdoor: {cctv.get('outdoor', '-')}")

        if "access_control_system" in elv_security:
            lines.append(f"Access Control System = ₹{elv_security.get('access_control_system', {}).get('provision_inr', 0):,} provision")

        if "fire_detection_system" in elv_security:
            lines.append(f"Fire Detection System = {elv_security.get('fire_detection_system', {}).get('value', 'Not defined')}")

        if "public_address_system" in elv_security:
            lines.append(f"Public Address System = {elv_security.get('public_address_system', {}).get('value', 'Not defined')}")

        if "bms" in elv_security:
            lines.append(f"BMS = ₹{elv_security.get('bms', {}).get('provision_inr', 0):,} provision")

        if "home_automation_system" in elv_security:
            lines.append(f"Home Automation System = {elv_security.get('home_automation_system', {}).get('value', 'Not defined')}")

        if "fire_suppression_system" in elv_security:
            lines.append(f"Fire Suppression System (BMS Room) = ₹{elv_security.get('fire_suppression_system', {}).get('provision_inr', 0):,} provision")

        if "visitors_management_system" in elv_security:
            lines.append(f"Visitors Management System = ₹{elv_security.get('visitors_management_system', {}).get('provision_inr', 0):,} provision")

    # =====================================================
    # PLUMBING
    # =====================================================
    if plumbing:
        lines.append("")
        lines.append("PLUMBING")

        if "external_plumbing_sanitation" in plumbing:
            lines.append(f"External Plumbing & Sanitation = {plumbing.get('external_plumbing_sanitation', {}).get('value', 'Not defined')}")

        pumps = plumbing.get("pumps_water_supply_system", {})
        if pumps:
            lines.append(
                f"Pumps & Water Supply = "
                f"DWS {pumps.get('dws_pump_sets', 0)} set | "
                f"FWS {pumps.get('fws_pump_sets', 0)} set | "
                f"Booster {pumps.get('booster_pump_sets', 0)} set | "
                f"Rain Water {pumps.get('rain_water_pump_sets', 0)} set | "
                f"Sump {pumps.get('sump_pump_sets', 0)} sets | "
                f"Solar {pumps.get('solar_pump_sets', 0)} set"
            )

        cp = plumbing.get("cp_sanitary_fittings", {})
        if cp:
            lines.append(
                f"CP & Sanitary Fittings = WC ₹{cp.get('wc_inr', 0):,} | "
                f"WB ₹{cp.get('wb_inr', 0):,} | "
                f"Shower ₹{cp.get('shower_inr', 0):,} | "
                f"Urinal ₹{cp.get('urinal_inr', 0):,} | "
                f"Kitchen Sink ₹{cp.get('kitchen_sink_inr', 0):,}"
            )

        pipes = plumbing.get("pipes", {})
        if pipes:
            lines.append(
                "Pipe Materials = "
                f"Internal WS: {pipes.get('internal_toilet_water_supply', '-')} | "
                f"Internal Drain: {pipes.get('internal_toilet_drainage', '-')} | "
                f"Shaft WS: {pipes.get('shaft_water_supply', '-')} | "
                f"Shaft Drain: {pipes.get('shaft_drainage', '-')} | "
                f"Terrace Loop: {pipes.get('terrace_loop', '-')} | "
                f"UGT to OHT: {pipes.get('ugt_to_oht', '-')} | "
                f"Sewage: {pipes.get('sewage', '-')} | "
                f"Storm Water: {pipes.get('storm_water', '-')}"
            )

        if "rain_water_harvesting" in plumbing:
            lines.append(f"Rain Water Harvesting = {plumbing.get('rain_water_harvesting', {}).get('value', 'Not defined')}")

        stp = plumbing.get("stp", {})
        if stp:
            lines.append(f"STP = {stp.get('capacity_kld_each', 0)} KLD x {stp.get('count', 0)} Nos")

        wtp = plumbing.get("wtp", {})
        if wtp:
            lines.append(f"WTP = {wtp.get('capacity_kld', 0)} KLD")

        owc = plumbing.get("owc", {})
        if owc:
            lines.append(f"OWC = {owc.get('capacity_kg', 0)} KG")

    # =====================================================
    # FIRE FIGHTING
    # =====================================================
    if fire_fighting:
        lines.append("")
        lines.append("FIRE FIGHTING")

        if "sprinkler_system" in fire_fighting:
            lines.append(f"Sprinkler System = {fire_fighting.get('sprinkler_system', {}).get('value', 'Not defined')}")

        if "internal_external_hydrant_system" in fire_fighting:
            lines.append(f"Internal & External Hydrant System = {fire_fighting.get('internal_external_hydrant_system', {}).get('value', 'Not defined')}")

        pumping = fire_fighting.get("pumping_system", {})
        if pumping:
            lines.append(
                f"Pumping System = Sprinkler Main Pump {pumping.get('sprinkler_main_pump_nos', 0)} No | "
                f"Hydrant Main Pump {pumping.get('hydrant_main_pump_nos', 0)} No | "
                f"Jockey Pump {pumping.get('jockey_pump_nos', 0)} Nos | "
                f"Diesel Main Pump {pumping.get('diesel_main_pump_nos', 0)} No | "
                f"Booster Pump {pumping.get('booster_pump_sets', 0)} Sets"
            )

        if "piping_system" in fire_fighting:
            lines.append(f"Piping System = {fire_fighting.get('piping_system', {}).get('value', 'Not defined')}")

        if "fire_extinguishers" in fire_fighting:
            lines.append(f"Fire Extinguishers = {fire_fighting.get('fire_extinguishers', {}).get('value', 'Not defined')}")

        if "drencher_system" in fire_fighting:
            lines.append(f"Drencher System = {fire_fighting.get('drencher_system', {}).get('value', 'Not defined')}")

    # =====================================================
    # ELEVATORS
    # =====================================================
    if elevator_works:
        lines.append("")
        lines.append("ELEVATOR WORKS")

        pe = elevator_works.get("passenger_elevator", {})
        if pe:
            lines.append(
                f"Passenger Elevator = {pe.get('count', 0)} Nos | "
                f"{pe.get('speed_mps', 0)} MPS | "
                f"{pe.get('stops', 0)} Stops | "
                f"{pe.get('capacity_pax', 0)} Pax | "
                f"{pe.get('finishes', '-')}"
            )

        se = elevator_works.get("service_elevator", {})
        if se:
            lines.append(
                f"Service Elevator = {se.get('count', 0)} Nos | "
                f"{se.get('speed_mps', 0)} MPS | "
                f"{se.get('stops', 0)} Stops | "
                f"{se.get('capacity_pax', 0)} Pax | "
                f"{se.get('finishes', '-')}"
            )

    # =====================================================
    # HVAC
    # =====================================================
    if hvac:
        lines.append("")
        lines.append("HVAC")

        vrv = hvac.get("vrv_system", {})
        if vrv:
            lines.append(
                f"VRV System = {vrv.get('value', '-')} | "
                f"{vrv.get('odu_per_floor', 0)} Nos x "
                f"{vrv.get('odu_hp_each', 0)} HP at each floor | "
                f"Total {vrv.get('total_tr', 0)} TR"
            )

        if "split_air_conditioners" in hvac:
            lines.append(f"Split Air Conditioners = {hvac.get('split_air_conditioners', {}).get('value', 'Not defined')}")

        if "ventilation_fans" in hvac:
            lines.append(f"Ventilation Fans = {hvac.get('ventilation_fans', {}).get('value', 'Not defined')}")

    # =====================================================
    # OTHER SYSTEMS
    # =====================================================
    if other_systems:
        lines.append("")
        lines.append("OTHER SYSTEMS")

        if "geyser" in other_systems:
            lines.append(f"Geyser = {other_systems.get('geyser', {}).get('value', 'Not defined')}")

        if "piped_gas_system" in other_systems:
            lines.append(f"Piped Gas System = {other_systems.get('piped_gas_system', {}).get('value', 'Not defined')}")

        if "irrigation_system" in other_systems:
            lines.append(f"Irrigation System = {other_systems.get('irrigation_system', {}).get('value', 'Not defined')}")

        if "swimming_pool" in other_systems:
            lines.append(f"Swimming Pool = {other_systems.get('swimming_pool', {}).get('value', 'Not defined')}")

        if "garbage_chute" in other_systems:
            lines.append(f"Garbage Chute = {other_systems.get('garbage_chute', {}).get('value', 'Not defined')}")

        if "refuge_chute" in other_systems:
            lines.append(f"Refuge Chute = {other_systems.get('refuge_chute', {}).get('value', 'Not defined')}")

        shw = other_systems.get("solar_hot_water_system", {})
        if shw:
            lines.append(
                f"Solar Hot Water System = {shw.get('capacity_ltr', 0)} Ltr | "
                f"{shw.get('basis_note', '-')}"
            )

        sep = other_systems.get("solar_electric_power_system", {})
        if sep:
            lines.append(f"Solar Electric Power System = {sep.get('capacity_kw', 0)} kW")

    return lines



    # --------------------------------------------------------
    # Estimate Result
    # --------------------------------------------------------
    doc.add_heading("Estimate Result", level=1)

    result_lines = [
        f"Total Estimated Cost: ₹{result.get('total_cost_cr', 0)} Cr",
        f"Cost / SBA Sqft: ₹{result.get('cost_per_sba', 0):,}",
        f"Cost / BUA Sqft: ₹{result.get('cost_per_bua', 0):,}",
        f"Range: ₹{result.get('range_low_cr', 0)} Cr – ₹{result.get('range_high_cr', 0)} Cr",
    ]

    for line in result_lines:
        doc.add_paragraph(line)

    # --------------------------------------------------------
    # BOQ Breakdown
    # --------------------------------------------------------
    doc.add_heading("BOQ Breakdown", level=1)

    boq = result.get("boq_breakdown", {})
    if boq:
        table = doc.add_table(rows=1, cols=5)
        hdr = table.rows[0].cells
        hdr[0].text = "BOQ Head"
        hdr[1].text = "Amount (Cr)"
        hdr[2].text = "Cost/SBA"
        hdr[3].text = "Cost/BUA"
        hdr[4].text = "% of Total"

        for head, vals in boq.items():
            row = table.add_row().cells
            row[0].text = str(head)
            row[1].text = str(vals.get("amount_cr", 0))
            row[2].text = str(vals.get("cost_per_sba", 0))
            row[3].text = str(vals.get("cost_per_bua", 0))
            row[4].text = str(vals.get("pct", 0))
    else:
        doc.add_paragraph("BOQ breakdown not available.")

    # --------------------------------------------------------
    # Civil Assumptions
    # --------------------------------------------------------
    doc.add_heading("Civil Assumptions", level=1)
    civil_assumptions = result.get("civil_assumptions", {})
    if civil_assumptions:
        for section, value in civil_assumptions.items():
            doc.add_paragraph(f"{section}: {value}")
    else:
        doc.add_paragraph("No civil assumptions available.")

    # --------------------------------------------------------
    # MEP Assumptions
    # --------------------------------------------------------
def flatten_civil_assumptions_for_display(civil_assumptions):
    lines = []

    if not civil_assumptions:
        return ["No civil assumptions available."]

    pm = civil_assumptions.get("project_meta", {})
    ga = civil_assumptions.get("general_assumptions", {})
    br = civil_assumptions.get("basic_rates", {})
    mw = civil_assumptions.get("material_wastage", {})
    bc = civil_assumptions.get("building_configuration", {})
    fh = civil_assumptions.get("floor_to_floor_heights_m", {})
    sc = civil_assumptions.get("shell_core", {})
    rcc = civil_assumptions.get("rcc_indices", {})
    masonry = civil_assumptions.get("masonry", {})
    plaster = civil_assumptions.get("plaster", {})
    wp = civil_assumptions.get("waterproofing", {})
    finishes = civil_assumptions.get("internal_finishes", {})
    common = civil_assumptions.get("common_area_finishes", {})
    facade = civil_assumptions.get("doors_windows_facade", {})
    ext = civil_assumptions.get("external_development", {})

    lines.append("CIVIL ASSUMPTIONS")
    lines.append(f"Project = {pm.get('project_name', '-')}")
    lines.append(f"Client = {pm.get('client', '-')}")

    lines.append("")
    lines.append("GENERAL ASSUMPTIONS")
    site = ga.get("site_logistics", {})
    lines.append(f"Batching Plant = {site.get('batching_plant', '-')}")
    lines.append(f"Labour Accommodation = {site.get('labour_accommodation', '-')}")
    lines.append(f"Construction Equipments = {site.get('construction_equipments', '-')}")

    cs = ga.get("contracting_strategy", {})
    lines.append(f"Earthwork Contracting = {cs.get('earthwork', '-')}")
    lines.append(f"Shell & Core Contracting = {cs.get('shell_core', '-')}")
    lines.append(f"Finishing Works Contracting = {cs.get('finishing_works', '-')}")
    lines.append(f"Facade Works Contracting = {cs.get('facade_works', '-')}")
    lines.append(f"MEP Works Contracting = {cs.get('mep_works', '-')}")

    lines.append("")
    lines.append("BASIC RATES")
    lines.append(f"Cement = {br.get('cement_type', '-')} | ₹{br.get('cement_rate_per_bag', 0):,} / bag")
    lines.append(f"Steel = {br.get('steel_type', '-')} | ₹{br.get('steel_rate_per_mt', 0):,} / MT")

    lines.append("")
    lines.append("MATERIAL WASTAGE")
    lines.append(f"Cement Wastage = {mw.get('cement_pct', 0)}%")
    lines.append(f"RMC Wastage = {mw.get('rmc_pct', 0)}%")
    lines.append(f"Reinforcement Steel Wastage = {mw.get('reinforcement_steel_pct', 0)}%")
    lines.append(f"Tile Flooring Wastage = {mw.get('hard_finishes_tile_flooring_pct', 0)}%")
    lines.append(f"Tile Dado Wastage = {mw.get('hard_finishes_tile_dado_pct', 0)}%")
    lines.append(f"Marble / Granite / Stone Wastage = {mw.get('marble_granite_stone_pct', 0)}%")

    lines.append("")
    lines.append("BUILDING CONFIGURATION")
    lines.append(
        f"Semi Basement {bc.get('semi_basement_level', 0)} | "
        f"Upper Ground {bc.get('upper_ground_level', 0)} | "
        f"First Floor {bc.get('first_floor_level', 0)} | "
        f"Typical Floors {bc.get('typical_level', 0)} | "
        f"Refuge Floors {bc.get('refuge_levels', 0)} | "
        f"Roof Level {bc.get('roof_level', 0)}"
    )

    lines.append("")
    lines.append("FLOOR TO FLOOR HEIGHTS")
    lines.append(f"Lower Ground = {fh.get('lower_ground_level', 0)} m")
    lines.append(f"Ground = {fh.get('ground_level', 0)} m")
    lines.append(f"First Floor = {fh.get('first_floor_level', 0)} m")
    lines.append(f"Typical Floor = {fh.get('typical_level', 0)} m")
    lines.append(f"Refuge Floor = {fh.get('refuge_levels', 0)} m")

    lines.append("")
    lines.append("SHELL & CORE")
    lines.append(f"Earthwork Depth = {sc.get('earthwork_depth_m', 0)} m")
    lines.append(f"Foundation = {sc.get('foundation', '-')}")
    lines.append(f"Structure = {sc.get('sub_structure_super_structure', '-')}")
    lines.append(f"Shuttering — Sub Structure = {sc.get('shuttering_sub_structure', '-')}")
    lines.append(f"Shuttering — Super Structure = {sc.get('shuttering_super_structure', '-')}")

    lines.append("")
    lines.append("RCC INDICES")
    lines.append(f"Construction Area = {rcc.get('construction_area_sqft', 0):,.2f} Sq.Ft")
    lines.append(f"Concrete = {rcc.get('concrete_cum', 0):,.2f} Cu.m | {rcc.get('concrete_cum_per_sqm', 0)} Cu.m/Sqm")
    lines.append(f"Reinforcement = {rcc.get('reinforcement_mt', 0):,.3f} MT | {rcc.get('steel_kg_per_sqft', 0)} Kg/Sq.Ft")
    lines.append(f"Shuttering = {rcc.get('shuttering_sqm', 0):,.2f} Sq.m | {rcc.get('shuttering_sqm_per_cum', 0)} Sqm/Cum")

    lines.append("")
    lines.append("MASONRY / PLASTER / WATERPROOFING")
    lines.append(f"External Walls = {masonry.get('external_walls', '-')}")
    lines.append(f"Internal Walls = {masonry.get('internal_walls', '-')}")
    lines.append(f"External Plaster = {plaster.get('external', '-')}")
    lines.append(f"Internal Plaster = {plaster.get('internal', '-')}")
    lines.append(f"Back Coat = {plaster.get('back_coat', '-')}")
    lines.append(f"Raft Waterproofing = {wp.get('raft', '-')}")
    lines.append(f"Terrace Waterproofing = {wp.get('terraces', '-')}")
    lines.append(f"Toilet Waterproofing = {wp.get('toilets', '-')}")
    lines.append(f"Refuge Area Waterproofing = {wp.get('refuge_area', '-')}")

    lines.append("")
    lines.append("INTERNAL FINISHES")
    lines.append(f"Room Flooring = {finishes.get('room_flooring', '-')}")
    lines.append(f"Toilet Flooring = {finishes.get('toilet_flooring', '-')}")
    lines.append(f"Toilet Dado = {finishes.get('toilet_dado', '-')}")
    lines.append(f"Wash Basin Counter = {finishes.get('wash_basin_counter', '-')}")
    lines.append(f"Room Wall Finish = {finishes.get('room_wall_finish', '-')}")
    lines.append(f"Toilet Wall Finish = {finishes.get('toilet_wall_finish', '-')}")
    lines.append(f"Room Paint = {finishes.get('room_paint', '-')}")
    lines.append(f"Toilet Paint = {finishes.get('toilet_paint', '-')}")
    lines.append(f"Staircase Paint = {finishes.get('staircase_paint', '-')}")

    lines.append("")
    lines.append("COMMON AREA FINISHES")
    lines.append(f"Typical Lobby Flooring = {common.get('typical_lobby_flooring', '-')}")
    lines.append(f"Ground Floor Lobby Flooring = {common.get('ground_floor_lobby_flooring', '-')}")
    lines.append(f"Staircase Flooring = {common.get('staircase_flooring', '-')}")
    lines.append(f"Ceiling Finish = {common.get('ceiling_finish', '-')}")
    lines.append(f"Painting = {common.get('painting', '-')}")

    lines.append("")
    lines.append("DOORS / WINDOWS / FACADE")
    lines.append(f"Student Room Main Door = {facade.get('student_room_main_door', '-')}")
    lines.append(f"Toilet Door = {facade.get('toilet_door', '-')}")
    lines.append(f"Handicap Toilet Door = {facade.get('handicap_toilet_door', '-')}")
    lines.append(f"Staircase Door = {facade.get('staircase_door', '-')}")
    lines.append(f"Room Windows = {facade.get('room_windows', '-')}")
    lines.append(f"Common Area Windows = {facade.get('common_area_windows', '-')}")
    lines.append(f"Facade Finish = {facade.get('facade_finish', '-')}")
    lines.append(f"Shaft = {facade.get('shaft', '-')}")
    lines.append(f"Special Facade = {facade.get('special_facade', '-')}")

    lines.append("")
    lines.append("EXTERNAL DEVELOPMENT")
    lines.append(f"Furniture Cost Included = ₹{ext.get('furniture_cost_included_inr', 0):,}")
    lines.append(f"Gym Equipment = {ext.get('gym_equipment', '-')}")
    lines.append(f"Kitchen Equipment Provisional = ₹{ext.get('kitchen_equipment_provisional_inr', 0):,}")
    lines.append(f"Gates = {ext.get('gates', 0)}")
    lines.append(f"Security Cabin = {ext.get('security_cabin', 0)}")
    lines.append(f"Internal Paved Footpath = {ext.get('internal_paved_footpath', '-')}")
    lines.append(f"Internal Bitumen Road = {ext.get('internal_bitumen_road', '-')}")
    lines.append(f"Paved Parking Area = {ext.get('paved_parking_area', '-')}")
    lines.append(f"Softscape = {ext.get('softscape', '-')}")
    lines.append(f"Signages = {ext.get('signages', '-')}")

    return lines   
# ============================================================
# MODEL LOAD
# ============================================================
forecast_model, forecast_meta = load_forecast_model()

# ============================================================
# HEADER
# ============================================================
st.markdown("<div class='main-title'>SmartBuild AI — New Estimate</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Fill the project sections tab-wise, add location details, and generate a polished cost estimate.</div>",
    unsafe_allow_html=True
)
st.markdown("---")

def apply_calibration(
    predicted_total_cost_cr,
    predicted_psf,
    quality_level,
    execution_type,
    efficiency_ratio,
    manual_calibration_factor
):
    """
    Calibrate estimate using real-world project conditions.
    """

    # --------------------------------------------------------
    # 1. Efficiency factor
    # Assume current app generic efficiency baseline = 70%
    # Lower actual efficiency increases cost realism
    # --------------------------------------------------------
    assumed_efficiency = 70
    efficiency_factor = assumed_efficiency / efficiency_ratio if efficiency_ratio > 0 else 1.0

    # --------------------------------------------------------
    # 2. Quality factor
    # --------------------------------------------------------
    quality_factor_map = {
        "Basic": 0.80,
        "Standard": 0.90,
        "Mid Premium": 1.00,
        "Premium": 1.10,
        "Luxury": 1.20
    }
    quality_factor = quality_factor_map.get(quality_level, 1.0)

    # --------------------------------------------------------
    # 3. Execution factor
    # --------------------------------------------------------
    execution_factor_map = {
        "Standard": 1.00,
        "Optimized EPC": 0.90,
        "Value Engineered": 0.82
    }
    execution_factor = execution_factor_map.get(execution_type, 1.0)

    # --------------------------------------------------------
    # 4. Apply overall calibration
    # --------------------------------------------------------
    calibrated_total_cost_cr = (
        predicted_total_cost_cr
        * efficiency_factor
        * quality_factor
        * execution_factor
        * manual_calibration_factor
    )

    calibrated_total_cost_cr = round(calibrated_total_cost_cr, 2)

    if predicted_total_cost_cr > 0:
        adjustment_ratio = calibrated_total_cost_cr / predicted_total_cost_cr
    else:
        adjustment_ratio = 1.0

    calibrated_psf = round(predicted_psf * adjustment_ratio, 0)

    return {
        "calibrated_total_cost_cr": calibrated_total_cost_cr,
        "calibrated_psf": calibrated_psf,
        "efficiency_factor": round(efficiency_factor, 3),
        "quality_factor": round(quality_factor, 3),
        "execution_factor": round(execution_factor, 3),
        "manual_factor": round(manual_calibration_factor, 3),
        "adjustment_ratio": round(adjustment_ratio, 3)
    }


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_form_state():
    defaults = {
        # Tab 1
        "client_name": "",
        "project_name": "",
        "sector": "Residential",
        "main_category_height": "Mid Rise",
        "sub_category": "Premium",
        "project_status": "Ongoing",
        "country": "India",
        "region": "West",
        "state": "Maharashtra",
        "city": "Mumbai",
        "location": "",
        "micro_market": "",
        "pincode": "",
        "stage": "CD",

        # Tab 2
        "num_buildings": 1,
        "num_basement": 0,
        "num_ground": 1,
        "num_podium": 0,
        "num_amenity_floor": 0,
        "num_service_floor": 0,
        "num_refuge": 0,
        "num_fire_check": 0,
        "num_typical_floors": 1,
        "flr_height": 3.1,
        "podium_height": 3.6,
        "basement_height": 4.5,
        "above_ground_height": 0.0,
        "total_height": 0.0,
        "structure_type": "RCC",
        "shuttering_type": "Conventional+System",

        # Tab 3 – flat mix + areas
        "apartments_per_floor": 0,

        # Flat units
        "u_1 RK": 0,
        "u_Studio": 0,
        "u_1 BHK": 0,
        "u_1.5 BHK": 0,
        "u_2 BHK": 0,
        "u_2.5 BHK": 0,
        "u_3 BHK": 0,
        "u_3.5 BHK": 0,
        "u_4 BHK": 0,
        "u_5 BHK": 0,
        "u_Penthouse": 0,
        "u_Duplex": 0,
        "u_Triplex": 0,
        "u_Any Other": 0,

        # Carpet / unit
        "c_1 RK": 350,
        "c_Studio": 350,
        "c_1 BHK": 500,
        "c_1.5 BHK": 650,
        "c_2 BHK": 900,
        "c_2.5 BHK": 1100,
        "c_3 BHK": 1400,
        "c_3.5 BHK": 1600,
        "c_4 BHK": 2000,
        "c_5 BHK": 2600,
        "c_Penthouse": 3000,
        "c_Duplex": 3000,
        "c_Triplex": 4000,
        "c_Any Other": 500,

        # Area inputs
        "tower_builtup_sqft": 0,
        "non_tower_builtup_sqft": 0,
        "podium_builtup_sqft": 0,
        "basement_builtup_sqft": 0,

        "tower_construction_area_sqft": 0,
        "non_tower_construction_area_sqft": 0,
        "podium_construction_area_sqft": 0,
        "basement_construction_area_sqft": 0,

        "carpet_area_input": 0,
        "amenity_area_sqft": 0,
        "clubhouse_area": 0,
        "retail_area": 0,
        "landscape_area_sqft": 0,
        "no_of_car_parks": 0,

        # Tab 4 compatibility
        "excavation_soil_pct": 63.0,
        "excavation_hardrock_pct": 37.0,
        "excavation_softrock_pct": 0.0,
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_form_state()

# ============================================================
# APPLY PENDING HISTORY LOAD BEFORE WIDGETS ARE CREATED
# ============================================================
if "_pending_history_load" in st.session_state:
    payload = st.session_state["_pending_history_load"]

    for k, v in payload.items():
        st.session_state[k] = v

    del st.session_state["_pending_history_load"]

# # TEMP DEBUG FORCE — Step 20 only
# st.session_state["tower_builtup_sqft"] = 1000
# st.session_state["non_tower_builtup_sqft"] = 200
# st.session_state["tower_construction_area_sqft"] = 1500
# st.session_state["non_tower_construction_area_sqft"] = 300

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Project Details",
    "🏢 Building Config",
    "🏠 Flat Mix",
    "⚙️ Specifications",
    "🏊 Amenities",
    "📂 History"
])

# ============================================================
# APPLY PENDING HISTORY LOAD BEFORE WIDGETS ARE CREATED
# ============================================================
# if "_pending_history_load" in st.session_state:
#     payload = st.session_state["_pending_history_load"]

#     for k, v in payload.items():
#         st.session_state[k] = v

#     del st.session_state["_pending_history_load"]
# ============================================================
# TAB 1 — PROJECT DETAILS
# ============================================================
with tab1:
    st.markdown("<div class='section-header'>Project Details</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        client_name = st.text_input("Client Name", key="client_name")
        project_name = st.text_input("Project Name", key="project_name")
        sector = st.selectbox("Sector", SECTOR_OPTIONS, key="sector")
        main_category_height = st.selectbox("Main Category (Height)", MAIN_CATEGORY_HEIGHT_OPTIONS, key="main_category_height")
        sub_category = st.selectbox("Sub Category (Sector)", SUB_CATEGORY_OPTIONS, key="sub_category")
        project_status = st.selectbox("Project Status", PROJECT_STATUS_OPTIONS, key="project_status")

    with c2:
        country = st.selectbox("Country", COUNTRY_OPTIONS, key="country")
        region = st.selectbox("Region", REGION_OPTIONS, key="region")
        state = st.selectbox("State", STATE_OPTIONS, key="state")
        city = st.text_input("City", key="city")
        location = st.text_input("Location", key="location")
        micro_market = st.text_input("Micro Market", key="micro_market")
        pincode = st.text_input("Pincode", key="pincode")

    stage = st.selectbox("Stage of Budget", STAGE_OF_BUDGET_OPTIONS, key="stage")

# ============================================================
# TAB 2 — BUILDING CONFIG
# ============================================================
with tab2:
    st.markdown("<div class='section-header'>Project & Tower Configuration</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    num_buildings = b1.number_input("Number of Towers", min_value=1, max_value=50, step=1, key="num_buildings")
    num_basement = b2.number_input("Number of Basement", min_value=0, max_value=10, step=1, key="num_basement")
    num_ground = b3.number_input("Number of Ground Floor", min_value=0, max_value=5, step=1, key="num_ground")

    b4, b5, b6 = st.columns(3)
    num_podium = b4.number_input("Number of Podium", min_value=0, max_value=10, step=1, key="num_podium")
    num_amenity_floor = b5.number_input("Number of Amenity Floors", min_value=0, max_value=10, step=1, key="num_amenity_floor")
    num_service_floor = b6.number_input("Number of Service Floors", min_value=0, max_value=10, step=1, key="num_service_floor")

    b7, b8, b9 = st.columns(3)
    num_refuge = b7.number_input("Number of Refuge Floors", min_value=0, max_value=20, step=1, key="num_refuge")
    num_fire_check = b8.number_input("Number of Fire Check Floors", min_value=0, max_value=20, step=1, key="num_fire_check")
    num_typical_floors = b9.number_input("Number of Typical Floors", min_value=1, max_value=100, step=1, key="num_typical_floors")

    total_floors = (
        num_basement
        + num_ground
        + num_podium
        + num_amenity_floor
        + num_service_floor
        + num_refuge
        + num_fire_check
        + num_typical_floors
    )

    st.metric("Total Number of Floors", total_floors)

    st.markdown("<div class='section-header'>Heights</div>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    flr_height = h1.number_input("Tower Floor-to-Floor Height (m)", min_value=2.5, max_value=6.0, step=0.1, key="flr_height")
    podium_height = h2.number_input("Podium Floor-to-Floor Height (m)", min_value=2.5, max_value=8.0, step=0.1, key="podium_height")
    basement_height = h3.number_input("Basement Floor-to-Floor Height (m)", min_value=2.5, max_value=8.0, step=0.1, key="basement_height")

    hh1, hh2 = st.columns(2)
    above_ground_height = hh1.number_input("Above Ground Height (m)", min_value=0.0, max_value=500.0, step=0.1, key="above_ground_height")
    total_height = hh2.number_input("Total Height (m)", min_value=0.0, max_value=500.0, step=0.1, key="total_height")

    s1, s2 = st.columns(2)
    structure_type = s1.selectbox("Type of Structure", STRUCTURE_OPTIONS, key="structure_type")
    shuttering_type = s2.selectbox("Type of Shuttering", SHUTTERING_OPTIONS, key="shuttering_type")
# ============================================================
# APPLY PENDING HISTORY LOAD BEFORE WIDGETS ARE CREATED
# ============================================================
# if "_pending_history_load" in st.session_state:
#     payload = st.session_state["_pending_history_load"]
#     for k, v in payload.items():
#         st.session_state[k] = v
#     del st.session_state["_pending_history_load"]
# ============================================================
# TAB 3 — FLAT MIX + AREA
# ============================================================
with tab3:
    st.markdown("<div class='section-header'>Flat Matrix</div>", unsafe_allow_html=True)

    excel_flat_types = [
        "1 RK", "Studio", "1 BHK", "1.5 BHK", "2 BHK", "2.5 BHK",
        "3 BHK", "3.5 BHK", "4 BHK", "5 BHK", "Penthouse", "Duplex",
        "Triplex", "Any Other"
    ]

    h1, h2, h3, h4 = st.columns([2.2, 1.2, 1.4, 1.2])
    h1.markdown("**Flat Type**")
    h2.markdown("**Units**")
    h3.markdown("**Carpet / Unit (Sq Ft)**")
    h4.markdown("**Total Carpet**")
    st.markdown("---")

    flat_units = {}
    flat_carpet = {}

    for ft in excel_flat_types:
        c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.4, 1.2])

        c1.write(ft)

        units_key = f"u_{ft}"
        carpet_key = f"c_{ft}"

        units_val = c2.number_input(
            f"Units {ft}",
            min_value=0,
            max_value=50000,
            step=1,
            key=units_key,
            label_visibility="collapsed"
        )

        carpet_val = c3.number_input(
            f"Carpet {ft}",
            min_value=0,
            max_value=10000,
            step=10,
            key=carpet_key,
            label_visibility="collapsed"
        )

        subtotal = units_val * carpet_val
        c4.write(f"{subtotal:,}" if subtotal > 0 else "—")

        flat_units[ft] = units_val
        flat_carpet[ft] = carpet_val

    st.markdown("---")

    apartments_per_floor = st.number_input(
        "No. of Apartments per Floor",
        min_value=0,
        max_value=100,
        step=1,
        key="apartments_per_floor"
    )

    # --------------------------------------------------------
    # Flat Mix Summary
    # --------------------------------------------------------
    total_units = int(sum(flat_units.values()))
    total_carpet_mix = int(sum(flat_units[ft] * flat_carpet[ft] for ft in excel_flat_types))

    sm1, sm2 = st.columns(2)
    sm1.metric("Total Units", f"{total_units:,}")
    sm2.metric("Total Carpet From Mix", f"{total_carpet_mix:,} Sq Ft")

    st.markdown("<div class='section-header'>Area Details</div>", unsafe_allow_html=True)

    # -----------------------------
    # Built-up Area Inputs
    # -----------------------------
    a1, a2 = st.columns(2)
    tower_builtup_sqft = a1.number_input(
        "Tower Built-up Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="tower_builtup_sqft"
    )
    non_tower_builtup_sqft = a2.number_input(
        "Non Tower Built-up Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="non_tower_builtup_sqft"
    )

    # -----------------------------
    # Construction Area Inputs
    # -----------------------------
    a3, a4 = st.columns(2)
    tower_construction_area_sqft = a3.number_input(
        "Tower Construction Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="tower_construction_area_sqft"
    )
    non_tower_construction_area_sqft = a4.number_input(
        "Non Tower Construction Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="non_tower_construction_area_sqft"
    )

    # -----------------------------
    # Other area components
    # -----------------------------
    a5, a6, a7 = st.columns(3)
    podium_builtup_sqft = a5.number_input(
        "Podium Built-up Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="podium_builtup_sqft"
    )
    basement_builtup_sqft = a6.number_input(
        "Basement Built-up Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="basement_builtup_sqft"
    )
    carpet_area_input = a7.number_input(
        "Carpet Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="carpet_area_input"
    )

    a8, a9, a10 = st.columns(3)
    podium_construction_area_sqft = a8.number_input(
        "Podium Construction Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="podium_construction_area_sqft"
    )
    basement_construction_area_sqft = a9.number_input(
        "Basement Construction Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="basement_construction_area_sqft"
    )
    amenity_area_sqft = a10.number_input(
        "Amenity Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="amenity_area_sqft"
    )

    a11, a12, a13 = st.columns(3)
    clubhouse_area = a11.number_input(
        "Club House Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="clubhouse_area"
    )
    retail_area = a12.number_input(
        "Retail / Commercial Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="retail_area"
    )
    landscape_area_sqft = a13.number_input(
        "Landscape Area (Sq Ft)",
        min_value=0,
        max_value=100000000,
        step=100,
        key="landscape_area_sqft"
    )

    no_of_car_parks = st.number_input(
        "No. of Car Parks",
        min_value=0,
        max_value=100000,
        step=1,
        key="no_of_car_parks"
    )

    # --------------------------------------------------------
    # FINAL TOTALS
    # Flat matrix derived area + Tower + Non-Tower + Podium + Basement
    # --------------------------------------------------------
    effective_total_carpet = int(total_carpet_mix) if int(total_carpet_mix) > 0 else int(carpet_area_input)

    flat_mix_builtup_sqft = round(effective_total_carpet / 0.67, 0) if effective_total_carpet > 0 else 0
    flat_mix_construction_sqft = round(flat_mix_builtup_sqft * 1.08, 0) if flat_mix_builtup_sqft > 0 else 0

    total_builtup_sqft = (
        flat_mix_builtup_sqft
        + tower_builtup_sqft
        + non_tower_builtup_sqft
        + podium_builtup_sqft
        + basement_builtup_sqft
    )

    total_construction_area_sqft = (
        flat_mix_construction_sqft
        + tower_construction_area_sqft
        + non_tower_construction_area_sqft
        + podium_construction_area_sqft
        + basement_construction_area_sqft
    )

    m1, m2 = st.columns(2)
    m1.metric("Total Built-up Area (Calculated)", f"{int(total_builtup_sqft):,} Sq Ft")
    m2.metric("Total Construction Area (Calculated)", f"{int(total_construction_area_sqft):,} Sq Ft")

    # --------------------------------------------------------
    # Compatibility Mapping for downstream logic
    # --------------------------------------------------------
    total_carpet = effective_total_carpet
    total_bua = int(total_builtup_sqft)
    total_sba = int(total_construction_area_sqft)

    st.info(
        f"Using for current app flow → Carpet: {int(total_carpet):,} Sq Ft | "
        f"Flat-Mix Built-up: {int(flat_mix_builtup_sqft):,} Sq Ft | "
        f"Flat-Mix Construction: {int(flat_mix_construction_sqft):,} Sq Ft | "
        f"Final BUA: {int(total_bua):,} Sq Ft | "
        f"Final Construction Area (SBA alias): {int(total_sba):,} Sq Ft"
    )

    # --------------------------------------------------------
    # Toilet Matrix
    # --------------------------------------------------------
    st.markdown("<div class='section-header'>Toilet Matrix</div>", unsafe_allow_html=True)
    st.caption("For now this is manual / placeholder. In the next step we will make it rule-based from flat mix.")

    t1, t2, t3, t4 = st.columns(4)

    master_toilets = t1.number_input(
        "Master Toilets (Total)",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="master_toilets"
    )
    common_toilets = t2.number_input(
        "Common Toilets (Total)",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="common_toilets"
    )
    powder_toilets = t3.number_input(
        "Powder Toilets (Total)",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="powder_toilets"
    )
    servant_toilets = t4.number_input(
        "Servant / Driver Toilets (Total)",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="servant_toilets"
    )

    total_toilets = master_toilets + common_toilets + powder_toilets + servant_toilets

    tm1, tm2, tm3, tm4, tm5 = st.columns(5)
    tm1.metric("Master", f"{master_toilets:,}")
    tm2.metric("Common", f"{common_toilets:,}")
    tm3.metric("Powder", f"{powder_toilets:,}")
    tm4.metric("Servant", f"{servant_toilets:,}")
    tm5.metric("Total Toilets", f"{total_toilets:,}")

    # --------------------------------------------------------
    # Save final totals into session state
    # --------------------------------------------------------
    st.session_state["total_builtup_sqft"] = int(total_builtup_sqft)
    st.session_state["total_construction_area_sqft"] = int(total_construction_area_sqft)
    st.session_state["history_loaded_total_units"] = int(total_units)
    st.session_state["history_loaded_total_carpet"] = int(total_carpet)
# ============================================================
# TAB 4 — CIVIL / FINISHING / FACADE
# ============================================================
with tab4:
    st.info("Tab 4 temporarily disabled for debugging")
    st.markdown("<div class='section-header'>Earthwork / Civil Assumptions</div>", unsafe_allow_html=True)

    e1, e2 = st.columns(2)
    depth_of_excavation = e1.number_input(
        "Depth of Excavation (m)",
        min_value=0.0,
        max_value=100.0,
        value=14.18,
        step=0.1,
        key="tab4_depth_of_excavation"
    )
    excavation_mode = e2.selectbox(
        "Mode of Excavation",
        EXCAVATION_MODE_OPTIONS,
        index=2,
        key="tab4_excavation_mode"
    )

    st.markdown("<div class='section-header'>Excavation Split (%)</div>", unsafe_allow_html=True)

    ex1, ex2, ex3 = st.columns(3)
    excavation_soil_pct = ex1.number_input(
        "% Soil",
        min_value=0.0,
        max_value=100.0,
        value=63.0,
        step=1.0,
        key="tab4_excavation_soil_pct"
    )
    excavation_hardrock_pct = ex2.number_input(
        "% Hard Rock",
        min_value=0.0,
        max_value=100.0,
        value=37.0,
        step=1.0,
        key="tab4_excavation_hardrock_pct"
    )
    excavation_softrock_pct = ex3.number_input(
        "% Soft Rock",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        key="tab4_excavation_softrock_pct"
    )

    excavation_total_pct = excavation_soil_pct + excavation_hardrock_pct + excavation_softrock_pct

    if excavation_total_pct == 100:
        st.success(f"✅ Excavation split total = {excavation_total_pct:.0f}%")
    else:
        st.warning(f"⚠️ Excavation split total = {excavation_total_pct:.0f}%. It should be 100%.")

    st.markdown("<div class='section-header'>Structure</div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    structure_type = s1.selectbox(
        "Type of Structure",
        STRUCTURE_OPTIONS,
        index=0,
        key="tab4_structure_type"
    )
    shuttering_type = s2.selectbox(
        "Type of Shuttering",
        SHUTTERING_OPTIONS,
        index=2,
        key="tab4_shuttering_type"
    )

    st.markdown("<div class='section-header'>Finishing — Flooring</div>", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    flooring_living_dining = f1.selectbox(
        "Flooring Living & Dining",
        FLOORING_OPTIONS,
        index=1,
        key="tab4_flooring_living_dining"
    )
    flooring_master_bedroom = f2.selectbox(
        "Flooring Master Bedroom",
        FLOORING_OPTIONS,
        index=2,
        key="tab4_flooring_master_bedroom"
    )
    flooring_other_bedrooms = f3.selectbox(
        "Flooring Other Bedrooms",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_other_bedrooms"
    )

    f4, f5, f6 = st.columns(3)
    flooring_servant_room = f4.selectbox(
        "Flooring Servant Room",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_servant_room"
    )
    flooring_kitchen = f5.selectbox(
        "Flooring Kitchen",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_kitchen"
    )
    flooring_master_toilet = f6.selectbox(
        "Flooring Master Toilets",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_master_toilet"
    )

    f7, f8, f9 = st.columns(3)
    flooring_other_toilet = f7.selectbox(
        "Flooring Other Toilets",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_other_toilet"
    )
    flooring_servant_toilet = f8.selectbox(
        "Flooring Servant Toilet",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_servant_toilet"
    )
    flooring_balcony = f9.selectbox(
        "Flooring Deck / Dry Balcony",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_balcony"
    )

    f10, f11 = st.columns(2)
    flooring_common_lobby = f10.selectbox(
        "Flooring Common Lobbies",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_flooring_common_lobby"
    )
    flooring_staircase = f11.selectbox(
        "Flooring Staircase",
        FLOORING_OPTIONS,
        index=3,
        key="tab4_flooring_staircase"
    )

    st.markdown("<div class='section-header'>Finishing — Dado</div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    dado_kitchen = d1.selectbox(
        "Dado Kitchen",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_dado_kitchen"
    )
    dado_master_toilet = d2.selectbox(
        "Dado Master Toilets",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_dado_master_toilet"
    )
    dado_other_toilet = d3.selectbox(
        "Dado Other Toilets",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_dado_other_toilet"
    )

    d4, d5 = st.columns(2)
    dado_servant_toilet = d4.selectbox(
        "Dado Servant Toilet",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_dado_servant_toilet"
    )
    dado_common_lobby = d5.selectbox(
        "Dado Common Lobbies",
        FLOORING_OPTIONS,
        index=0,
        key="tab4_dado_common_lobby"
    )

    st.markdown("<div class='section-header'>Doors</div>", unsafe_allow_html=True)

    dr1, dr2, dr3, dr4 = st.columns(4)
    main_door_type = dr1.selectbox(
        "Main Door",
        DOOR_OPTIONS,
        index=0,
        key="tab4_main_door"
    )
    bedroom_door_type = dr2.selectbox(
        "Bedroom Door",
        DOOR_OPTIONS,
        index=0,
        key="tab4_bedroom_door"
    )
    toilet_door_type = dr3.selectbox(
        "Toilet Door",
        DOOR_OPTIONS,
        index=0,
        key="tab4_toilet_door"
    )
    common_area_door_type = dr4.selectbox(
        "Common Area Door",
        DOOR_OPTIONS,
        index=2,
        key="tab4_common_area_door"
    )

    st.markdown("<div class='section-header'>Facade / Window</div>", unsafe_allow_html=True)

    fa1, fa2, fa3 = st.columns(3)
    facade_system = fa1.selectbox(
        "Type of Facade System",
        FACADE_SYSTEM_OPTIONS,
        index=0,
        key="tab4_facade_system"
    )
    window_configuration = fa2.selectbox(
        "Window Configuration",
        WINDOW_CONFIGURATION_OPTIONS,
        index=0,
        key="tab4_window_configuration"
    )
    window_type = fa3.selectbox(
        "Type of Window",
        WINDOW_TYPE_OPTIONS,
        index=0,
        key="tab4_window_type"
    )

# TAB 5
# ============================================================
# TAB 5 — MEP / AMENITIES / LIFTS
# ============================================================
with tab5:
    st.markdown("<div class='section-header'>Electrical / PHE / Fire / ELV</div>", unsafe_allow_html=True)
    st.info("Tab 5 temporarily disabled for debugging")
    m1, m2, m3 = st.columns(3)
    electrical_ht = m1.selectbox(
        "Electrical Works - HT",
        ["No", "Yes"],
        index=1,
        key="tab5_electrical_ht"
    )
    rising_mains = m2.selectbox(
        "Rising Mains",
        RISING_MAINS_OPTIONS,
        index=0,
        key="tab5_rising_mains"
    )
    sprinklers_inside_apartments = m3.selectbox(
        "Sprinklers inside Apartments",
        ["No", "Yes"],
        index=1,
        key="tab5_sprinklers_inside_apartments"
    )

    p1, p2, p3 = st.columns(3)
    phe_system_type = p1.selectbox(
        "PHE System Type",
        PHE_SYSTEM_OPTIONS,
        index=1,
        key="tab5_phe_system_type"
    )
    fire_alarm_inside_apartments = p2.selectbox(
        "Fire Alarm inside Apartments",
        ["No", "Yes"],
        index=1,
        key="tab5_fire_alarm_inside_apartments"
    )
    public_address_inside_apartments = p3.selectbox(
        "Public Address inside Apartments",
        ["No", "Yes"],
        index=1,
        key="tab5_public_address_inside_apartments"
    )

    p4, p5, p6 = st.columns(3)
    fapa_type = p4.selectbox(
        "FAPA Type",
        FAPA_SYSTEM_OPTIONS,
        index=0,
        key="tab5_fapa_type"
    )
    av_required = p5.selectbox(
        "AV",
        ["No", "Yes"],
        index=0,
        key="tab5_av_required"
    )
    it_required = p6.selectbox(
        "IT",
        ["No", "Yes"],
        index=0,
        key="tab5_it_required"
    )

    p7, p8, p9 = st.columns(3)
    security_required = p7.selectbox(
        "Security",
        ["No", "Yes"],
        index=0,
        key="tab5_security_required"
    )
    acs_required = p8.selectbox(
        "ACS",
        ["No", "Yes"],
        index=1,
        key="tab5_acs_required"
    )
    cctv_required = p9.selectbox(
        "CCTV",
        ["No", "Yes"],
        index=1,
        key="tab5_cctv_required"
    )

    p10 = st.columns(1)[0]
    bms_required = p10.selectbox(
        "BMS",
        ["No", "Yes"],
        index=1,
        key="tab5_bms_required"
    )

    st.markdown("<div class='section-header'>Pipe Material Assumptions</div>", unsafe_allow_html=True)

    pm1, pm2, pm3 = st.columns(3)
    water_supply_internal = pm1.selectbox(
        "Water Supply - Internal",
        PIPE_MATERIAL_OPTIONS,
        index=1,  # CPVC
        key="tab5_water_supply_internal"
    )
    water_supply_shaft = pm2.selectbox(
        "Water Supply - Shaft",
        PIPE_MATERIAL_OPTIONS,
        index=0,  # UPVC
        key="tab5_water_supply_shaft"
    )
    drainage_internal = pm3.selectbox(
        "Drainage - Internal",
        PIPE_MATERIAL_OPTIONS,
        index=5,  # DWC
        key="tab5_drainage_internal"
    )

    pm4, pm5 = st.columns(2)
    drainage_shaft = pm4.selectbox(
        "Drainage - Shaft",
        PIPE_MATERIAL_OPTIONS,
        index=5,  # DWC
        key="tab5_drainage_shaft"
    )
    external_pipe_material = pm5.selectbox(
        "External Pipe Material",
        PIPE_MATERIAL_OPTIONS,
        index=0,  # UPVC
        key="tab5_external_pipe_material"
    )

    st.markdown("<div class='section-header'>Utility / Service Features</div>", unsafe_allow_html=True)

    u1, u2, u3 = st.columns(3)
    stp_required = u1.selectbox(
        "STP",
        ["No", "Yes"],
        index=1,
        key="tab5_stp_required"
    )
    wtp_required = u2.selectbox(
        "WTP",
        ["No", "Yes"],
        index=0,
        key="tab5_wtp_required"
    )
    owc_required = u3.selectbox(
        "OWC",
        ["No", "Yes"],
        index=1,
        key="tab5_owc_required"
    )

    u4, u5, u6 = st.columns(3)
    solar_electric_required = u4.selectbox(
        "Solar Electric",
        ["No", "Yes"],
        index=0,
        key="tab5_solar_electric_required"
    )
    solar_water_required = u5.selectbox(
        "Solar Water",
        ["No", "Yes"],
        index=1,
        key="tab5_solar_water_required"
    )
    ftth_required = u6.selectbox(
        "FTTH / DTH",
        ["No", "Yes"],
        index=1,
        key="tab5_ftth_required"
    )

    u7, u8, u9 = st.columns(3)
    ev_charging_pct = u7.number_input(
        "EV Charging Provision (%)",
        min_value=0,
        max_value=100,
        value=25,
        step=1,
        key="tab5_ev_charging_pct"
    )
    ac_inside_apartments = u8.selectbox(
        "AC inside Apartments",
        AC_OPTIONS,
        index=0,
        key="tab5_ac_inside_apartments"
    )
    ac_inside_lobbies = u9.selectbox(
        "AC inside Typical Lobbies",
        ["No", "Yes"],
        index=1,
        key="tab5_ac_inside_lobbies"
    )

    u10, u11, u12 = st.columns(3)
    mechanical_parking_type = u10.selectbox(
        "Mechanical Parking Type",
        PARKING_OPTIONS,
        index=1,
        key="tab5_mechanical_parking_type"
    )
    piped_gas_required = u11.selectbox(
        "Piped Gas",
        ["No", "Yes"],
        index=1,
        key="tab5_piped_gas_required"
    )
    dg_type = u12.selectbox(
        "Diesel Generator",
        DG_OPTIONS,
        index=0,
        key="tab5_dg_type"
    )

    st.markdown("<div class='section-header'>Mechanical Parking Count</div>", unsafe_allow_html=True)

    mp1, mp2, mp3 = st.columns(3)
    stack_car_parking = mp1.number_input(
        "Stack Car Parking",
        min_value=0,
        max_value=100000,
        value=472,
        step=1,
        key="tab5_stack_car_parking"
    )
    puzzle_car_parking = mp2.number_input(
        "Puzzle Car Parking",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="tab5_puzzle_car_parking"
    )
    parking_tower_count = mp3.number_input(
        "Parking Tower",
        min_value=0,
        max_value=100000,
        value=0,
        step=1,
        key="tab5_parking_tower_count"
    )

    st.markdown("<div class='section-header'>Elevators</div>", unsafe_allow_html=True)

    st.markdown("#### Passenger Elevators")
    l1, l2, l3, l4 = st.columns(4)
    passenger_lift_count = l1.number_input(
        "Passenger Elevator - No.",
        min_value=0,
        max_value=100,
        value=47,
        step=1,
        key="tab5_passenger_lift_count"
    )
    passenger_lift_speed = l2.number_input(
        "Passenger Elevator - Speed (MPS)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        key="tab5_passenger_lift_speed"
    )
    passenger_lift_stops = l3.number_input(
        "Passenger Elevator - Stops",
        min_value=0,
        max_value=200,
        value=34,
        step=1,
        key="tab5_passenger_lift_stops"
    )
    passenger_lift_capacity = l4.number_input(
        "Passenger Elevator - Capacity (Kg)",
        min_value=0,
        max_value=5000,
        value=1300,
        step=25,
        key="tab5_passenger_lift_capacity"
    )

    st.markdown("#### Service Elevators")
    l5, l6, l7, l8 = st.columns(4)
    service_lift_count = l5.number_input(
        "Service Elevator - No.",
        min_value=0,
        max_value=100,
        value=8,
        step=1,
        key="tab5_service_lift_count"
    )
    service_lift_speed = l6.number_input(
        "Service Elevator - Speed (MPS)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        key="tab5_service_lift_speed"
    )
    service_lift_stops = l7.number_input(
        "Service Elevator - Stops",
        min_value=0,
        max_value=200,
        value=34,
        step=1,
        key="tab5_service_lift_stops"
    )
    service_lift_capacity = l8.number_input(
        "Service Elevator - Capacity (Kg)",
        min_value=0,
        max_value=5000,
        value=1625,
        step=25,
        key="tab5_service_lift_capacity"
    )

    st.markdown("#### Fire Elevators")
    l9, l10, l11, l12 = st.columns(4)
    fire_lift_count = l9.number_input(
        "Fire Elevator - No.",
        min_value=0,
        max_value=100,
        value=8,
        step=1,
        key="tab5_fire_lift_count"
    )
    fire_lift_speed = l10.number_input(
        "Fire Elevator - Speed (MPS)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        key="tab5_fire_lift_speed"
    )
    fire_lift_stops = l11.number_input(
        "Fire Elevator - Stops",
        min_value=0,
        max_value=200,
        value=34,
        step=1,
        key="tab5_fire_lift_stops"
    )
    fire_lift_capacity = l12.number_input(
        "Fire Elevator - Capacity (Kg)",
        min_value=0,
        max_value=5000,
        value=1625,
        step=25,
        key="tab5_fire_lift_capacity"
    )

    st.markdown("#### Car Elevators")
    l13, l14, l15, l16 = st.columns(4)
    car_lift_count = l13.number_input(
        "Car Elevator - No.",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        key="tab5_car_lift_count"
    )
    car_lift_speed = l14.number_input(
        "Car Elevator - Speed (MPS)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        key="tab5_car_lift_speed"
    )
    car_lift_stops = l15.number_input(
        "Car Elevator - Stops",
        min_value=0,
        max_value=200,
        value=0,
        step=1,
        key="tab5_car_lift_stops"
    )
    car_lift_capacity = l16.number_input(
        "Car Elevator - Capacity (Kg)",
        min_value=0,
        max_value=10000,
        value=0,
        step=25,
        key="tab5_car_lift_capacity"
    )

    st.markdown("<div class='section-header'>Apartment Amenities</div>", unsafe_allow_html=True)

    aa1, aa2, aa3, aa4 = st.columns(4)
    video_door_phone = aa1.selectbox(
        "Video Door Phone",
        ["No", "Yes"],
        index=1,
        key="tab5_video_door_phone"
    )
    home_automation = aa2.selectbox(
        "Home Automation",
        ["No", "Yes"],
        index=0,
        key="tab5_home_automation"
    )
    modular_kitchen = aa3.selectbox(
        "Modular Kitchen",
        ["No", "Yes"],
        index=0,
        key="tab5_modular_kitchen"
    )
    water_purifier = aa4.selectbox(
        "Water Purifier",
        ["No", "Yes"],
        index=0,
        key="tab5_water_purifier"
    )

    aa5, aa6, aa7 = st.columns(3)
    intercom = aa5.selectbox(
        "Intercom",
        ["No", "Yes"],
        index=1,
        key="tab5_intercom"
    )
    geyser = aa6.selectbox(
        "Geyser",
        ["No", "Yes"],
        index=0,
        key="tab5_geyser"
    )
    iot_required = aa7.selectbox(
        "IOT",
        ["No", "Yes"],
        index=0,
        key="tab5_iot_required"
    )

    st.markdown("<div class='section-header'>Other Amenities</div>", unsafe_allow_html=True)

    oa1, oa2, oa3 = st.columns(3)
    entrance_lobby = oa1.selectbox(
        "Entrance Lobby",
        ["No", "Yes"],
        index=1,
        key="tab5_entrance_lobby"
    )
    clubhouse_required = oa2.selectbox(
        "Clubhouse",
        ["No", "Yes"],
        index=1,
        key="tab5_clubhouse_required"
    )
    ibs_system = oa3.selectbox(
        "IBS System",
        ["No", "Yes"],
        index=0,
        key="tab5_ibs_system"
    )

    oa4, oa5, oa6 = st.columns(3)
    gym_play_equipment = oa4.selectbox(
        "Gym & Play Equipment",
        ["No", "Yes"],
        index=1,
        key="tab5_gym_play_equipment"
    )
    swimming_pool_equipment = oa5.selectbox(
        "Swimming Pool Equipment",
        ["No", "Yes"],
        index=1,
        key="tab5_swimming_pool_equipment"
    )
    other_misc_amenities = oa6.selectbox(
        "Other Misc. Amenities",
        ["No", "Yes"],
        index=0,
        key="tab5_other_misc_amenities"
    )

# ============================================================
# TAB 6 — HISTORY
# ============================================================
with tab6:
    st.markdown("## 📂 History Projects")
    st.caption("View or edit your previously generated project estimates.")

    if os.path.exists(PAST_ESTIMATES_FILE):
        df_hist = pd.read_csv(PAST_ESTIMATES_FILE)
    else:
        df_hist = pd.DataFrame()

    if df_hist.empty:
        st.info("No saved estimates found. Generate an estimate to save history.")
    else:
        df_hist["project_name"] = df_hist["project_name"].astype(str).str.strip()
        project_names = df_hist["project_name"].dropna().unique().tolist()

        if not project_names:
            st.info("No valid project names found in history.")
        else:
            selected = st.selectbox("Select a saved project", project_names, key="history_select")
            selected = str(selected).strip()

            filtered_rows = df_hist[df_hist["project_name"] == selected]

            if filtered_rows.empty:
                st.warning("⚠️ Selected project could not be found in history.")
            else:
                row = filtered_rows.iloc[0]

                hist_project_name = row.get("project_name", "—")
                hist_city = row.get("city", "—")
                hist_region = row.get("region", "West")
                hist_category = row.get("category", "Premium")
                hist_stage = row.get("stage", "CD")
                hist_total_units = row.get("total_units", 0)
                hist_total_bua = row.get("total_bua", row.get("total_bua_sqft", row.get("total_builtup_sqft", 0)))
                hist_total_cost_cr = row.get("total_cost_cr", 0)
                hist_client = row.get("client_name", "")
                hist_total_carpet = row.get("total_carpet", row.get("total_carpet_sqft", 0))

                st.markdown("### 🗂 Project Summary")

                c1, c2, c3 = st.columns(3)
                c1.metric("Project", hist_project_name)
                c2.metric("City", hist_city)
                c3.metric("Category", hist_category)

                c4, c5, c6 = st.columns(3)
                c4.metric("BUA", f"{int(hist_total_bua):,} sqft" if pd.notna(hist_total_bua) else "—")
                c5.metric("Units", f"{int(hist_total_units):,}" if pd.notna(hist_total_units) else "—")
                c6.metric("Cost", f"₹{hist_total_cost_cr} Cr" if pd.notna(hist_total_cost_cr) else "—")

                st.markdown("---")
                st.markdown("### ✏️ Edit Project")

                e1, e2 = st.columns(2)

                with e1:
                    new_name = st.text_input("Project Name", value=hist_project_name, key="hist_project_name")
                    new_city = st.text_input("City", value=str(hist_city), key="hist_city")

                    hist_region = str(hist_region)
                    region_idx = REGION_OPTIONS.index(hist_region) if hist_region in REGION_OPTIONS else 0
                    new_region = st.selectbox("Region", REGION_OPTIONS, index=region_idx, key="hist_region")

                    hist_stage = str(hist_stage)
                    stage_idx = STAGE_OF_BUDGET_OPTIONS.index(hist_stage) if hist_stage in STAGE_OF_BUDGET_OPTIONS else 0
                    new_stage = st.selectbox("Stage", STAGE_OF_BUDGET_OPTIONS, index=stage_idx, key="hist_stage")

                with e2:
                    hist_category = str(hist_category)
                    cat_idx = SUB_CATEGORY_OPTIONS.index(hist_category) if hist_category in SUB_CATEGORY_OPTIONS else 0
                    new_cat = st.selectbox("Category", SUB_CATEGORY_OPTIONS, index=cat_idx, key="hist_category")

                    new_units = st.number_input(
                        "Total Units",
                        min_value=0,
                        value=int(hist_total_units) if pd.notna(hist_total_units) else 0,
                        key="hist_units"
                    )

                    new_bua = st.number_input(
                        "Total BUA (Sq Ft)",
                        min_value=0,
                        value=int(hist_total_bua) if pd.notna(hist_total_bua) else 0,
                        key="hist_bua"
                    )

                    new_client = st.text_input("Client Name", value=str(hist_client), key="hist_client")

                if st.button("💾 Save Changes", key="history_save"):
                    df_hist.loc[
                        df_hist["project_name"] == selected,
                        ["project_name", "city", "region", "stage", "category", "client_name", "total_units", "total_bua"]
                    ] = [
                        new_name,
                        new_city,
                        new_region,
                        new_stage,
                        new_cat,
                        new_client,
                        new_units,
                        new_bua
                    ]
                    df_hist.to_csv(PAST_ESTIMATES_FILE, index=False)
                    st.success("Saved successfully!")

                st.markdown("---")
                st.markdown("### ↩️ Load Into Main Form")

                if st.button("Load this project", key="history_load"):
                    st.session_state["_pending_history_load"] = {
                        # Tab 1
                        "project_name": new_name,
                        "client_name": new_client,
                        "city": new_city,
                        "region": new_region,
                        "sub_category": new_cat,
                        "stage": new_stage,

                        # History fallbacks
                        "history_loaded_total_units": int(new_units),
                        "history_loaded_total_bua": int(new_bua),
                        "history_loaded_total_carpet": int(hist_total_carpet) if pd.notna(hist_total_carpet) else 0,

                        # Tab 3 visible area values
                        "tower_builtup_sqft": int(new_bua),
                        "non_tower_builtup_sqft": 0,
                        "podium_builtup_sqft": 0,
                        "basement_builtup_sqft": 0,
                        "tower_construction_area_sqft": int(row.get("total_sba", row.get("total_sba_sqft", 0))) if pd.notna(row.get("total_sba", row.get("total_sba_sqft", 0))) else 0,
                        "non_tower_construction_area_sqft": 0,
                        "podium_construction_area_sqft": 0,
                        "basement_construction_area_sqft": 0,
                    }

                    st.success("Loaded! Check tabs 1–5 to review fields.")
                    st.rerun()


# ============================================================
# COMPATIBILITY MAPPING FOR OLD DOWNSTREAM LOGIC
# ============================================================
category = st.session_state.get("sub_category", "Premium")
num_tower = st.session_state.get("num_typical_floors", 1)

total_bua = int(st.session_state.get("total_builtup_sqft", 0))
if total_bua == 0:
    total_bua = int(st.session_state.get("history_loaded_total_bua", 0))

total_sba = int(st.session_state.get("total_construction_area_sqft", 0))

total_carpet = int(st.session_state.get("carpet_area_input", 0))
if total_carpet == 0:
    total_carpet = int(total_carpet_mix) if "total_carpet_mix" in locals() else 0
if total_carpet == 0:
    total_carpet = int(st.session_state.get("history_loaded_total_carpet", 0))

if "total_units" not in locals():
    total_units = 0
if int(total_units) == 0:
    total_units = int(st.session_state.get("history_loaded_total_units", 0))

if st.session_state.get("excavation_hardrock_pct", 0) >= 50:
    soil_type = "Hard Rock"
elif st.session_state.get("excavation_soil_pct", 0) >= 50:
    soil_type = "Soil"
elif st.session_state.get("excavation_softrock_pct", 0) >= 50:
    soil_type = "Soft Rock"
else:
    soil_type = "Mixed"

# ============================================================
# SIDEBAR — MARKET / RATE INPUTS
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Market Rates")
    st.caption("Use manual benchmark rates or fetch API/fallback rates.")

    # --------------------------------------------------------
    # Initialize session state defaults once
    # --------------------------------------------------------
    if "steel_rate_mt" not in st.session_state:
        st.session_state["steel_rate_mt"] = 52000

    if "cement_rate_bag" not in st.session_state:
        st.session_state["cement_rate_bag"] = 300

    if "diesel_rate_ltr" not in st.session_state:
        st.session_state["diesel_rate_ltr"] = 90

    if "steel_source_label" not in st.session_state:
        st.session_state["steel_source_label"] = "manual"

    if "sidebar_rate_source" not in st.session_state:
        st.session_state["sidebar_rate_source"] = "Manual"

    # --------------------------------------------------------
    # Rate source selector
    # --------------------------------------------------------
    rate_source = st.selectbox(
        "Rate Source",
        ["Manual", "API/Fallback"],
        index=0 if st.session_state["sidebar_rate_source"] == "Manual" else 1,
        key="sidebar_rate_source"
    )

    # --------------------------------------------------------
    # Apply API / fallback benchmark rates if selected
    # --------------------------------------------------------
    if rate_source == "API/Fallback":
        try:
            # ------------------------------------------------
            # Replace this section later with your real API call
            # Example:
            # live_rates = fetch_live_rates()
            # st.session_state["steel_rate_mt"] = live_rates["Steel TMT"]
            # st.session_state["cement_rate_bag"] = live_rates["Cement"]
            # st.session_state["diesel_rate_ltr"] = live_rates["Diesel"]
            # st.session_state["steel_source_label"] = live_rates["steel_source"]
            # ------------------------------------------------

            # TEMPORARY: force fallback until real API is connected
            raise Exception("API not connected yet")

        except Exception:
            # Realistic benchmark fallback values
            st.session_state["steel_rate_mt"] = 55000
            st.session_state["cement_rate_bag"] = 320
            st.session_state["diesel_rate_ltr"] = 95
            st.session_state["steel_source_label"] = "market_benchmark_fallback"

            st.warning("⚠️ API unavailable. Using fallback benchmark rates.")

    else:
        # Manual mode
        st.session_state["steel_source_label"] = "manual"

    # --------------------------------------------------------
    # Sidebar widgets (bound to session state so values update visibly)
    # --------------------------------------------------------
    steel_rate_mt = st.number_input(
        "Steel TMT (₹/MT)",
        min_value=0,
        max_value=200000,
        step=500,
        key="steel_rate_mt"
    )

    cement_rate_bag = st.number_input(
        "Cement (₹/Bag)",
        min_value=0,
        max_value=1000,
        step=5,
        key="cement_rate_bag"
    )

    diesel_rate_ltr = st.number_input(
        "Diesel (₹/Litre)",
        min_value=0,
        max_value=500,
        step=1,
        key="diesel_rate_ltr"
    )

    st.info(f"Current source: {st.session_state['steel_source_label']}")

    st.markdown("---")
    st.markdown("## ⚙️ App Controls")

    if st.button("🔄 Reset App", use_container_width=True, key="sidebar_reset_app"):
        st.session_state.clear()
        st.rerun()

# ------------------------------------------------------------
# Final sidebar_rates dictionary used by estimate logic
# ------------------------------------------------------------
sidebar_rates = {
    "Steel TMT": st.session_state["steel_rate_mt"],
    "steel_rate_inr_per_kg": (
        st.session_state["steel_rate_mt"] / 1000
        if st.session_state["steel_rate_mt"] else 0
    ),
    "steel_source": st.session_state["steel_source_label"],
    "Cement": st.session_state["cement_rate_bag"],
    "Diesel": st.session_state["diesel_rate_ltr"]
}

# ============================================================
# CALIBRATION SETTINGS
# ============================================================
st.markdown("### 🎯 Calibration Settings")

cal1, cal2 = st.columns(2)

with cal1:
    efficiency_ratio = st.slider(
        "BUA Efficiency (%)",
        min_value=55,
        max_value=85,
        value=60,
        step=1,
        key="cal_efficiency_ratio"
    )

    quality_level = st.selectbox(
        "Quality Level",
        ["Basic", "Standard", "Mid Premium", "Premium", "Luxury"],
        index=2,
        key="cal_quality_level"
    )

with cal2:
    execution_type = st.selectbox(
        "Execution Type",
        ["Standard", "Optimized EPC", "Value Engineered"],
        index=0,
        key="cal_execution_type"
    )
# ============================================================
# GENERATE BUTTON
# ============================================================
st.markdown("---")

if total_units > 0:
    p1, p2, p3, p4, p5 = st.columns(5)
    p1.metric("Project", project_name if project_name else "—")
    p2.metric("Region", region)
    p3.metric("Category", sub_category)
    p4.metric("Units", total_units)
    p5.metric("SBA", f"{total_sba:,} sqft")

g1, g2, g3 = st.columns([1, 2, 1])
with g2:
    generate_clicked = st.button("🚀 Generate Cost Estimate", use_container_width=True)

# Final values used for generation
final_total_units = int(total_units) if int(total_units) > 0 else int(st.session_state.get("history_loaded_total_units", 0))
final_total_carpet = int(total_carpet) if int(total_carpet) > 0 else int(st.session_state.get("history_loaded_total_carpet", 0))

if generate_clicked:
    if final_total_units == 0 or final_total_carpet == 0:
        st.error("❌ Please fill the Flat Mix tab with at least one valid unit type.")
    elif total_bua == 0 or total_sba == 0:
        st.error("❌ Please fill the Tab 3 area fields (Tower/Non-Tower/Podium/Basement) to calculate BUA and Construction Area.")
    else:

        try:
            # ----------------------------------------------------
            # 1. Base prediction
            # ----------------------------------------------------
            base_result = predict_estimate(
                forecast_model=forecast_model,
                forecast_meta=forecast_meta,
                city=city,
                category=sub_category,
                stage=stage,
                num_buildings=num_buildings,
                num_basement=num_basement,
                num_podium=num_podium,
                num_tower=num_typical_floors,
                total_floors=total_floors,
                flr_height=flr_height,
                total_sba=total_sba,
                total_bua=total_bua,
                total_units=final_total_units,
                region=region,
                micro_market=micro_market,
                pincode=pincode,
                soil_type=soil_type,
                total_carpet=final_total_carpet,
                sidebar_rates=sidebar_rates
            )

            base_total_cost_cr = base_result["total_cost_cr"]
            base_psf = base_result["cost_per_bua"]

            # ----------------------------------------------------
            # 2. Steel rate integration
            # ----------------------------------------------------
            benchmark_steel_rate_mt = 52000
            current_steel_rate_mt = sidebar_rates.get("Steel TMT", benchmark_steel_rate_mt)

            steel_index = (
                current_steel_rate_mt / benchmark_steel_rate_mt
                if benchmark_steel_rate_mt > 0 else 1.0
            )

            steel_impact_factor = 0.80 + (0.20 * steel_index)

            # ----------------------------------------------------
            # 3. Smart auto calibration
            # ----------------------------------------------------
            auto_calib = auto_calibration(
                city=city,
                region=region,
                micro_market=micro_market,
                total_floors=total_floors,
                num_basement=num_basement,
                num_podium=num_podium,
                efficiency_ratio=efficiency_ratio,
                quality_level=quality_level,
                execution_type=execution_type,
                total_bua=total_bua,
                category=sub_category
            )

            auto_factor = auto_calib["auto_factor"]

            model_adjusted_cost_cr = round(
                base_total_cost_cr * steel_impact_factor * auto_factor,
                2
            )

            # ----------------------------------------------------
            # 4. CIVIL CORE COST BLOCK — DYNAMIC
            # ----------------------------------------------------
            construction_area_sqft = float(total_sba)
            construction_area_sqm = construction_area_sqft * 0.092903

            concrete_index_cum_per_sqm = 0.470
            steel_index_kg_per_sqft = 4.50
            shuttering_index_sqm_per_cum = 6.14

            concrete_wastage_pct = 0.02
            steel_wastage_pct = 0.05

            steel_rate_per_mt = 36000
            rmc_rate_per_cum = 0
            shuttering_rate_per_sqm = 0

            concrete_cum_base = construction_area_sqm * concrete_index_cum_per_sqm
            steel_kg_base = construction_area_sqft * steel_index_kg_per_sqft
            steel_mt_base = steel_kg_base / 1000
            shuttering_sqm_base = concrete_cum_base * shuttering_index_sqm_per_cum

            concrete_cum = concrete_cum_base * (1 + concrete_wastage_pct)
            steel_kg = steel_kg_base * (1 + steel_wastage_pct)
            steel_mt = steel_kg / 1000
            shuttering_sqm = shuttering_sqm_base

            steel_cost = steel_mt * steel_rate_per_mt
            concrete_cost = concrete_cum * rmc_rate_per_cum
            shuttering_cost = shuttering_sqm * shuttering_rate_per_sqm

            civil_core_cost_total = steel_cost + concrete_cost + shuttering_cost
            civil_core_cost_cr = round(civil_core_cost_total / 1e7, 4)

            civil_core_summary = {
                "construction_area_sqft": round(construction_area_sqft, 2),
                "construction_area_sqm": round(construction_area_sqm, 2),
                "concrete_cum_base": round(concrete_cum_base, 2),
                "concrete_cum_waste": round(concrete_cum, 2),
                "steel_kg_base": round(steel_kg_base, 2),
                "steel_mt_base": round(steel_mt_base, 2),
                "steel_mt_waste": round(steel_mt, 2),
                "shuttering_sqm": round(shuttering_sqm, 2),
                "steel_cost": round(steel_cost, 2),
                "concrete_cost": round(concrete_cost, 2),
                "shuttering_cost": round(shuttering_cost, 2),
                "civil_core_cost_total": round(civil_core_cost_total, 2),
                "civil_core_cost_cr": civil_core_cost_cr
            }

            # ----------------------------------------------------
            # 5. SAFE DEFAULTS FOR OPTIONAL BLOCKS
            # ----------------------------------------------------
            # Keep these as defaults until you wire Masonry / Finishes / MEP blocks cleanly
            masonry_plaster_wp_total_cr = 0.0
            masonry_plaster_wp_summary = {}

            finishes_total_cr = 0.0
            finishes_summary = {}

            mep_total_cr = 0.0
            mep_summary = {}

            # ----------------------------------------------------
            # 6. INTEGRATE MODEL + COST BLOCKS
            # ----------------------------------------------------
            bottom_up_partial_cost_cr = round(
                civil_core_cost_cr
                + masonry_plaster_wp_total_cr
                + finishes_total_cr
                + mep_total_cr,
                4
            )

            bottom_up_gap_cr = round(bottom_up_partial_cost_cr - base_total_cost_cr, 4)
            bottom_up_gap_pct = round(
                (bottom_up_gap_cr / base_total_cost_cr) * 100,
                2
            ) if base_total_cost_cr > 0 else 0

            final_project_cost_cr = max(model_adjusted_cost_cr, bottom_up_partial_cost_cr)

            final_psf = round((final_project_cost_cr * 1e7) / total_bua, 0) if total_bua > 0 else 0

            # ----------------------------------------------------
            # 7. FINAL BOQ
            # ----------------------------------------------------
            boq_breakdown = build_boq_breakdown(
                total_cost_cr=final_project_cost_cr,
                total_sba=total_sba,
                total_bua=total_bua,
                category=sub_category,
                num_basement=num_basement
            )

            # ----------------------------------------------------
            # CIVIL + MEP ASSUMPTIONS OBJECTS
            # ----------------------------------------------------
            civil_assumptions = build_civil_assumptions()
            mep_assumptions = build_mep_assumptions()

            # ----------------------------------------------------
            # 8. FINAL RESULT OBJECT
            # ----------------------------------------------------
            result = {
                "civil_assumptions": civil_assumptions,
                "mep_assumptions": mep_assumptions,
                "total_cost_cr": round(final_project_cost_cr, 2),
                "range_low_cr": round(final_project_cost_cr * 0.90, 2),
                "range_high_cr": round(final_project_cost_cr * 1.10, 2),

                "cost_per_sba": int((final_project_cost_cr * 1e7) / total_sba) if total_sba else 0,
                "cost_per_bua": int(final_psf),

                "total_bua_sqft": total_bua,
                "total_sba_sqft": total_sba,
                "total_carpet_sqft": final_total_carpet,
                "total_units": final_total_units,

                "boq_breakdown": boq_breakdown,

                "location_factor": base_result.get("location_factor", 1.0),
                "live_rate_index": base_result.get("live_rate_index", 1.0),
                "dynamic_base_psf": int(final_psf),

                "civil_core_cost_cr": civil_core_cost_cr,
                "masonry_plaster_wp_total_cr": masonry_plaster_wp_total_cr,
                "finishes_total_cr": finishes_total_cr,
                "mep_total_cr": mep_total_cr,
                "bottom_up_partial_cost_cr": bottom_up_partial_cost_cr,
                "model_adjusted_cost_cr": round(model_adjusted_cost_cr, 2),
                "bottom_up_gap_cr": bottom_up_gap_cr,
                "bottom_up_gap_pct": bottom_up_gap_pct,

                "civil_core_summary": civil_core_summary,
                "masonry_plaster_wp_summary": masonry_plaster_wp_summary,
                "finishes_summary": finishes_summary,
                "mep_summary": mep_summary,
                
                "civil_assumptions": civil_assumptions,
                "mep_assumptions": mep_assumptions,


                "assumptions": [
                    f"Base estimated cost from model / rule engine = ₹{base_total_cost_cr} Cr",
                    f"Base cost / BUA sqft = ₹{base_psf:,}",
                    f"Current Steel Rate = ₹{current_steel_rate_mt:,.0f} / MT",
                    f"Benchmark Steel Rate = ₹{benchmark_steel_rate_mt:,.0f} / MT",
                    f"Steel Index = {round(steel_index, 3)}×",
                    f"Steel Impact Factor = {round(steel_impact_factor, 3)}×",
                    *auto_calib["summary"],
                    f"Model Adjusted Cost = ₹{round(model_adjusted_cost_cr, 2)} Cr",
                    f"Civil Core Cost = ₹{civil_core_cost_cr} Cr",
                    f"Masonry + Plaster + Waterproofing Cost = ₹{masonry_plaster_wp_total_cr} Cr",
                    f"Finishes Cost = ₹{finishes_total_cr} Cr",
                    f"MEP Cost = ₹{mep_total_cr} Cr",
                    f"Bottom-up Partial Cost = ₹{bottom_up_partial_cost_cr} Cr",
                    f"Bottom-up Gap vs Base = ₹{bottom_up_gap_cr} Cr ({bottom_up_gap_pct}%)",
                    "Final Project Cost = max(Model Adjusted Cost, Bottom-up Partial Cost)",
                    f"Civil Core Concrete Qty = {civil_core_summary.get('concrete_cum_waste', 0)} Cu.m",
                    f"Civil Core Steel Qty = {civil_core_summary.get('steel_mt_waste', 0)} MT",
                    f"Civil Core Shuttering Qty = {civil_core_summary.get('shuttering_sqm', 0)} Sq.m",
                    f"Toilet Density = {round((total_toilets / max(final_total_units, 1)), 2) if 'total_toilets' in locals() else 0} toilets/unit",
                    f"Location Factor = {base_result.get('location_factor', 1.0)}×",
                    "BOQ breakdown is allocated after final integration using rule-based percentages."
                ],
                
                "scope_flags": base_result.get("scope_flags", [])
            }

            st.markdown("### 🏗 Civil Assumptions")

            civil_display_lines = flatten_civil_assumptions_for_display(
                    result.get("civil_assumptions", {})
                )

            if civil_display_lines:
                    for line in civil_display_lines:
                        if line.strip() == "":
                            continue
                        elif line.isupper():
                            st.markdown(f"#### {line}")
                        else:
                            st.markdown(f"- {line}")
            else:
                    st.info("No civil assumptions available.")


            st.markdown("### 🔌 MEP Assumptions")

            mep_display_lines = flatten_mep_assumptions_for_display(
            result.get("mep_assumptions", {})
        )

            if mep_display_lines:
                for line in mep_display_lines:
                    if line.strip() == "":
                        continue
                    elif line.isupper():
                        st.markdown(f"#### {line}")
                    else:
                        st.markdown(f"- {line}")
            else:
                st.info("No MEP assumptions available.")

            # ----------------------------------------------------
            # 9. SAVE TO SESSION STATE
            # ----------------------------------------------------
            st.session_state["result"] = result
            st.session_state["project_data"] = {
                "project_name": project_name,
                "client_name": client_name,
                "region": region,
                "state": state,
                "city": city,
                "micro_market": micro_market,
                "pincode": pincode,
                "category": sub_category,
                "stage": stage,
                "soil_type": soil_type,
                "total_units": final_total_units,
                "total_carpet": final_total_carpet,
                "total_bua": total_bua,
                "total_sba": total_sba,
                "live_steel_rate": sidebar_rates.get("Steel TMT"),
                "live_steel_source": sidebar_rates.get("steel_source")
            }

            save_past_estimate(st.session_state["project_data"], st.session_state["result"])

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # ----------------------------------------------------
# DYNAMIC SCOPE TEXT
# ----------------------------------------------------
    scope_parts = []

    if int(num_basement) > 0:
        scope_parts.append(f"{int(num_basement)} Basement")

    if int(num_ground) > 0:
        scope_parts.append(f"{int(num_ground)} Ground")

    if int(num_podium) > 0:
        scope_parts.append(f"{int(num_podium)} Podium")

    if int(num_typical_floors) > 0:
        scope_parts.append(f"{int(num_typical_floors)} Typical Floors")

    scope_parts.append("Roof")

    scope_text = " + ".join(scope_parts)

    
    civil_assumptions = build_civil_assumptions(
        project_name=project_name,
        client_name=client_name,
        region=region,
        state=state,
        city=city,
        micro_market=micro_market,
        pincode=pincode,
        category=sub_category,
        stage=stage,
        soil_type=soil_type,
        num_buildings=num_buildings,
        num_basement=num_basement,
        num_ground=num_ground,
        num_podium=num_podium,
        num_typical_floors=num_typical_floors,
        total_floors=total_floors,
        flr_height=flr_height,
        structure_type=structure_type,
        shuttering_type=shuttering_type,
        total_units=final_total_units,
        total_carpet=final_total_carpet,
        total_bua=total_bua,
        total_sba=total_sba,
        excavation_mode=excavation_mode,
        excavation_soil_pct=excavation_soil_pct,
        excavation_hardrock_pct=excavation_hardrock_pct,
        excavation_softrock_pct=excavation_softrock_pct,
        depth_of_excavation=depth_of_excavation
    )

    mep_assumptions = build_mep_assumptions(
        project_name=project_name,
        client_name=client_name,
        region=region,
        state=state,
        city=city,
        micro_market=micro_market,
        pincode=pincode,
        category=sub_category,
        stage=stage,
        total_units=final_total_units,
        total_carpet=final_total_carpet,
        total_bua=total_bua,
        total_sba=total_sba,
        num_basement=num_basement,
        num_ground=num_ground,
        num_podium=num_podium,
        num_typical_floors=num_typical_floors,
        total_floors=total_floors,
        flr_height=flr_height,
        no_of_car_parks=no_of_car_parks,
        passenger_lift_count=passenger_lift_count,
        passenger_lift_speed=passenger_lift_speed,
        passenger_lift_stops=passenger_lift_stops,
        passenger_lift_capacity=passenger_lift_capacity,
        service_lift_count=service_lift_count,
        service_lift_speed=service_lift_speed,
        service_lift_stops=service_lift_stops,
        service_lift_capacity=service_lift_capacity,
        fire_lift_count=fire_lift_count,
        fire_lift_speed=fire_lift_speed,
        fire_lift_stops=fire_lift_stops,
        fire_lift_capacity=fire_lift_capacity,
        car_lift_count=car_lift_count,
        car_lift_speed=car_lift_speed,
        car_lift_stops=car_lift_stops,
        car_lift_capacity=car_lift_capacity,
        ev_charging_pct=ev_charging_pct,
        electrical_ht=electrical_ht,
        rising_mains=rising_mains,
        phe_system_type=phe_system_type,
        fapa_type=fapa_type,
        water_supply_internal=water_supply_internal,
        water_supply_shaft=water_supply_shaft,
        drainage_internal=drainage_internal,
        drainage_shaft=drainage_shaft,
        external_pipe_material=external_pipe_material,
        sprinklers_inside_apartments=sprinklers_inside_apartments,
        fire_alarm_inside_apartments=fire_alarm_inside_apartments,
        public_address_inside_apartments=public_address_inside_apartments,
        av_required=av_required,
        it_required=it_required,
        security_required=security_required,
        acs_required=acs_required,
        cctv_required=cctv_required,
        bms_required=bms_required,
        stp_required=stp_required,
        wtp_required=wtp_required,
        owc_required=owc_required,
        solar_electric_required=solar_electric_required,
        solar_water_required=solar_water_required,
        ftth_required=ftth_required,
        ac_inside_apartments=ac_inside_apartments,
        ac_inside_lobbies=ac_inside_lobbies,
        mechanical_parking_type=mechanical_parking_type,
        piped_gas_required=piped_gas_required,
        dg_type=dg_type
    )

# ============================================================
# RESULTS
# ============================================================
    result = st.session_state["result"]
    project_data = st.session_state.get("project_data", {})

    st.markdown("---")
    st.success("✅ Estimate Generated Successfully!")

    st.markdown("## 📊 Estimate Result")
    st.caption("Location-adjusted estimate using forecast model or fallback rule-based logic, plus live-rate / steel signal where available.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Total Estimated Cost",
        f"₹{result.get('total_cost_cr', 0)} Cr",
        delta=f"Range ₹{result.get('range_low_cr', 0)} – ₹{result.get('range_high_cr', 0)} Cr"
    )
    r2.metric("Cost / SBA Sqft", f"₹{result.get('cost_per_sba', 0):,}")
    r3.metric("Cost / BUA Sqft", f"₹{result.get('cost_per_bua', 0):,}")
    r4.metric("Total BUA", f"{result.get('total_bua_sqft', 0):,} sqft")

    rr1, rr2, rr3 = st.columns(3)
    rr1.metric("Dynamic Base PSF", f"₹{result.get('dynamic_base_psf', 0):,}")
    rr2.metric("Live Rate Index", f"{result.get('live_rate_index', 1.0)}")
    rr3.metric("Location Factor", f"{result.get('location_factor', 1.0)}×")

    st.markdown("### Project Snapshot")
    st.write(
        f"**{project_data.get('project_name') or 'Unnamed Project'}** | "
        f"{project_data.get('region', '—')} | "
        f"{project_data.get('category', '—')} | "
        f"Stage: {project_data.get('stage', '—')}"
    )

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("### 📋 BOQ Breakdown")

        boq_data = result.get("boq_breakdown", {})
        boq_rows = []

        for head, vals in boq_data.items():
            boq_rows.append({
                "BOQ Head": head,
                "Amount (Cr)": vals.get("amount_cr", 0),
                "Cost/SBA": vals.get("cost_per_sba", 0),
                "Cost/BUA": vals.get("cost_per_bua", 0),
                "% of Total": vals.get("pct", 0)
            })

        df_boq = pd.DataFrame(boq_rows)

        if not df_boq.empty:
            st.dataframe(df_boq, use_container_width=True, hide_index=True)

            boq_total_cr = round(df_boq["Amount (Cr)"].sum(), 2)
            expected_total_cr = round(result.get("total_cost_cr", 0), 2)

            st.markdown("### 🧪 BOQ Debug Check")
            st.write("Predicted Total Cost (Cr):", expected_total_cr)
            st.write("BOQ Total (Cr):", boq_total_cr)

            if abs(boq_total_cr - expected_total_cr) > 1.0:
                st.warning(f"⚠️ BOQ total mismatch: {boq_total_cr} Cr vs {expected_total_cr} Cr")
            else:
                st.success(f"✅ BOQ total matches predicted cost closely: {boq_total_cr} Cr")

            csv = df_boq.to_csv(index=False)
            st.download_button(
                "⬇️ Download BOQ CSV",
                csv,
                file_name=f"{project_data.get('project_name') or 'estimate'}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("BOQ breakdown is not available yet.")

    with right:
        st.markdown("### 📈 Cost Distribution")

        boq_data = result.get("boq_breakdown", {})
        if boq_data:
            chart_df = pd.DataFrame({
                "Amount (Cr)": [v.get("amount_cr", 0) for v in boq_data.values()]
            }, index=list(boq_data.keys()))
            st.bar_chart(chart_df, use_container_width=True, height=360)
        else:
            st.info("Cost distribution will appear once BOQ is available.")

    a1, a2 = st.columns(2)

    with a1:
        st.markdown("### 📝 Assumptions")
        assumptions_list = result.get("assumptions", [])
        if assumptions_list:
            for a in assumptions_list:
                st.markdown(f"- {a}")
        else:
            st.info("No assumptions available.")

    with a2:
        st.markdown("### ⚠️ Scope / Risk Flags")
        scope_flags = result.get("scope_flags", [])
        if scope_flags:
            for flag in scope_flags:
                st.warning(flag)
        else:
            st.success("No major scope gaps detected for this configuration.")

    st.markdown("### 🏗 Component Cost Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Civil Core", f"₹{result.get('civil_core_cost_cr', 0)} Cr")
    c2.metric("Masonry+Plaster+WP", f"₹{result.get('masonry_plaster_wp_total_cr', 0)} Cr")
    c3.metric("Finishes", f"₹{result.get('finishes_total_cr', 0)} Cr")
    c4.metric("MEP", f"₹{result.get('mep_total_cr', 0)} Cr")

    c5, c6, c7 = st.columns(3)
    c5.metric("Bottom-up Partial", f"₹{result.get('bottom_up_partial_cost_cr', 0)} Cr")
    c6.metric("Model Adjusted", f"₹{result.get('model_adjusted_cost_cr', 0)} Cr")
    c7.metric("Bottom-up Gap", f"{result.get('bottom_up_gap_pct', 0)}%")


    # ------------------------------------------------------------
    # MEP Assumptions Display
    # ------------------------------------------------------------
    st.markdown("### 🔌 MEP Assumptions")

    mep_display_lines = flatten_mep_assumptions_for_display(
        result.get("mep_assumptions", {})
    )

    if mep_display_lines:
        for line in mep_display_lines:
            if line.strip() == "":
                continue
            elif line.isupper():
                st.markdown(f"#### {line}")
            else:
                st.markdown(f"- {line}")
    else:
        st.info("No MEP assumptions available.")

    # ------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------
    pdf_bytes = generate_estimate_pdf(project_data, result)
    if pdf_bytes is not None:
        st.download_button(
            label="📄 Download Estimation PDF",
            data=pdf_bytes,
            file_name=f"{project_data.get('project_name') or 'smartbuild_estimate'}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.info("Install reportlab to enable PDF download: python -m pip install reportlab")

    
# ============================================================
# PAST ESTIMATES PREVIEW
# ============================================================
st.markdown("---")
st.markdown("### 📋 Past Estimates (Latest 10)")

if os.path.exists(PAST_ESTIMATES_FILE):
    df_past = pd.read_csv(PAST_ESTIMATES_FILE)
    st.dataframe(df_past.tail(10), use_container_width=True, hide_index=True)
else:
    st.info("No past estimates yet. Generate your first estimate.")


