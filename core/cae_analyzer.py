import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# ── Process limits by material ──────────────────────────────────────
PROCESS_LIMITS: dict = {}
PROCESS_LIMITS.update({
    "CATAMOLD-304L": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 130),
        "pack_time": (5, 20),
        "max_shear_rate": 18000,
        "max_pressure": 140,
        "freeze_temp": 120,
    },
    "CATAMOLD-4140": {
        "melt_temp": (180, 205),
        "mold_temp": (35, 55),
        "inj_speed": (15, 45),
        "pack_pressure": (85, 135),
        "pack_time": (5, 20),
        "max_shear_rate": 18000,
        "max_pressure": 145,
        "freeze_temp": 120,
    },
    "CATAMOLD-17-4PH": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 140),
        "pack_time": (5, 20),
        "max_shear_rate": 20000,
        "max_pressure": 150,
        "freeze_temp": 120,
    },
    "CATAMOLD-316L": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 130),
        "pack_time": (5, 20),
        "max_shear_rate": 18000,
        "max_pressure": 140,
        "freeze_temp": 120,
    },
    "CATAMOLD-Fe-Ni": {
        "melt_temp": (172, 195),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (78, 125),
        "pack_time": (5, 18),
        "max_shear_rate": 17000,
        "max_pressure": 138,
        "freeze_temp": 118,
    },
    "CATAMOLD-Al6061": {
        "melt_temp": (160, 185),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (70, 115),
        "pack_time": (4, 15),
        "max_shear_rate": 22000,
        "max_pressure": 130,
        "freeze_temp": 110,
    },
    "CATAMOLD-Al7075": {
        "melt_temp": (165, 190),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (75, 120),
        "pack_time": (4, 15),
        "max_shear_rate": 22000,
        "max_pressure": 135,
        "freeze_temp": 110,
    },
    "CATAMOLD-Ti-6Al-4V": {
        "melt_temp": (180, 210),
        "mold_temp": (35, 55),
        "inj_speed": (10, 40),
        "pack_pressure": (90, 150),
        "pack_time": (8, 25),
        "max_shear_rate": 15000,
        "max_pressure": 160,
        "freeze_temp": 130,
    },
    "CATAMOLD-TiH2": {
        "melt_temp": (178, 205),
        "mold_temp": (35, 55),
        "inj_speed": (10, 40),
        "pack_pressure": (88, 148),
        "pack_time": (8, 25),
        "max_shear_rate": 15000,
        "max_pressure": 158,
        "freeze_temp": 128,
    },

    "WAXBASE-304L": {
        "melt_temp": (160, 185),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (70, 120),
        "pack_time": (4, 18),
        "max_shear_rate": 18000,
        "max_pressure": 130,
        "freeze_temp": 110,
    },
    "WAXBASE-4140": {
        "melt_temp": (165, 190),
        "mold_temp": (30, 50),
        "inj_speed": (18, 55),
        "pack_pressure": (75, 125),
        "pack_time": (4, 18),
        "max_shear_rate": 18000,
        "max_pressure": 135,
        "freeze_temp": 112,
    },
    "WAXBASE-17-4PH": {
        "melt_temp": (160, 185),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (70, 120),
        "pack_time": (4, 18),
        "max_shear_rate": 18000,
        "max_pressure": 130,
        "freeze_temp": 110,
    },
    "WAXBASE-316L": {
        "melt_temp": (160, 185),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (70, 120),
        "pack_time": (4, 18),
        "max_shear_rate": 18000,
        "max_pressure": 130,
        "freeze_temp": 110,
    },
    "WAXBASE-Fe-Ni": {
        "melt_temp": (158, 183),
        "mold_temp": (25, 45),
        "inj_speed": (20, 60),
        "pack_pressure": (68, 118),
        "pack_time": (4, 16),
        "max_shear_rate": 17000,
        "max_pressure": 128,
        "freeze_temp": 108,
    },
    "WAXBASE-Al6061": {
        "melt_temp": (150, 175),
        "mold_temp": (20, 40),
        "inj_speed": (22, 65),
        "pack_pressure": (65, 110),
        "pack_time": (4, 14),
        "max_shear_rate": 20000,
        "max_pressure": 125,
        "freeze_temp": 105,
    },
    "WAXBASE-Al7075": {
        "melt_temp": (155, 180),
        "mold_temp": (20, 40),
        "inj_speed": (22, 65),
        "pack_pressure": (68, 112),
        "pack_time": (4, 14),
        "max_shear_rate": 20000,
        "max_pressure": 128,
        "freeze_temp": 105,
    },
    "WAXBASE-Ti-6Al-4V": {
        "melt_temp": (170, 195),
        "mold_temp": (30, 50),
        "inj_speed": (12, 45),
        "pack_pressure": (85, 145),
        "pack_time": (6, 22),
        "max_shear_rate": 15000,
        "max_pressure": 155,
        "freeze_temp": 125,
    },
    "WAXBASE-TiH2": {
        "melt_temp": (168, 192),
        "mold_temp": (30, 50),
        "inj_speed": (12, 45),
        "pack_pressure": (83, 142),
        "pack_time": (6, 22),
        "max_shear_rate": 15000,
        "max_pressure": 152,
        "freeze_temp": 123,
    },

    "PP": {
        "melt_temp": (200, 240),
        "mold_temp": (20, 60),
        "inj_speed": (40, 100),
        "pack_pressure": (50, 80),
        "pack_time": (4, 10),
        "max_shear_rate": 60000,
        "max_pressure": 120,
        "freeze_temp": 130,
    },
    "PE-HD": {
        "melt_temp": (190, 230),
        "mold_temp": (20, 55),
        "inj_speed": (35, 95),
        "pack_pressure": (45, 75),
        "pack_time": (4, 10),
        "max_shear_rate": 55000,
        "max_pressure": 115,
        "freeze_temp": 120,
    },
    "PE-LD": {
        "melt_temp": (180, 220),
        "mold_temp": (20, 50),
        "inj_speed": (35, 90),
        "pack_pressure": (45, 70),
        "pack_time": (4, 10),
        "max_shear_rate": 52000,
        "max_pressure": 110,
        "freeze_temp": 115,
    },
    "ABS": {
        "melt_temp": (210, 250),
        "mold_temp": (50, 80),
        "inj_speed": (25, 70),
        "pack_pressure": (60, 90),
        "pack_time": (5, 12),
        "max_shear_rate": 40000,
        "max_pressure": 140,
        "freeze_temp": 100,
    },
    "PC": {
        "melt_temp": (280, 320),
        "mold_temp": (80, 120),
        "inj_speed": (20, 60),
        "pack_pressure": (80, 120),
        "pack_time": (6, 15),
        "max_shear_rate": 35000,
        "max_pressure": 160,
        "freeze_temp": 145,
    },
    "PC+ABS": {
        "melt_temp": (230, 260),
        "mold_temp": (60, 80),
        "inj_speed": (30, 80),
        "pack_pressure": (70, 100),
        "pack_time": (5, 15),
        "max_shear_rate": 50000,
        "max_pressure": 160,
        "freeze_temp": 125,
    },
    "PA6": {
        "melt_temp": (235, 270),
        "mold_temp": (70, 90),
        "inj_speed": (20, 60),
        "pack_pressure": (75, 110),
        "pack_time": (6, 18),
        "max_shear_rate": 30000,
        "max_pressure": 170,
        "freeze_temp": 210,
    },
    "PA66": {
        "melt_temp": (250, 290),
        "mold_temp": (75, 95),
        "inj_speed": (20, 60),
        "pack_pressure": (80, 120),
        "pack_time": (8, 20),
        "max_shear_rate": 28000,
        "max_pressure": 180,
        "freeze_temp": 220,
    },
    "PA66+GF30": {
        "melt_temp": (270, 300),
        "mold_temp": (70, 90),
        "inj_speed": (20, 60),
        "pack_pressure": (80, 120),
        "pack_time": (8, 20),
        "max_shear_rate": 30000,
        "max_pressure": 180,
        "freeze_temp": 220,
    },
    "PA12": {
        "melt_temp": (170, 200),
        "mold_temp": (30, 50),
        "inj_speed": (25, 70),
        "pack_pressure": (60, 90),
        "pack_time": (5, 12),
        "max_shear_rate": 25000,
        "max_pressure": 140,
        "freeze_temp": 150,
    },
    "POM": {
        "melt_temp": (190, 220),
        "mold_temp": (80, 100),
        "inj_speed": (25, 70),
        "pack_pressure": (70, 100),
        "pack_time": (5, 12),
        "max_shear_rate": 35000,
        "max_pressure": 150,
        "freeze_temp": 130,
    },
    "PBT": {
        "melt_temp": (230, 260),
        "mold_temp": (70, 90),
        "inj_speed": (25, 75),
        "pack_pressure": (70, 100),
        "pack_time": (6, 14),
        "max_shear_rate": 30000,
        "max_pressure": 150,
        "freeze_temp": 170,
    },
    "PET": {
        "melt_temp": (240, 270),
        "mold_temp": (70, 95),
        "inj_speed": (25, 70),
        "pack_pressure": (75, 105),
        "pack_time": (6, 15),
        "max_shear_rate": 30000,
        "max_pressure": 155,
        "freeze_temp": 180,
    },
    "PPS": {
        "melt_temp": (300, 340),
        "mold_temp": (130, 160),
        "inj_speed": (20, 60),
        "pack_pressure": (90, 130),
        "pack_time": (8, 20),
        "max_shear_rate": 25000,
        "max_pressure": 180,
        "freeze_temp": 240,
    },
    "PPS+GF40": {
        "melt_temp": (310, 350),
        "mold_temp": (135, 165),
        "inj_speed": (20, 55),
        "pack_pressure": (95, 135),
        "pack_time": (8, 20),
        "max_shear_rate": 25000,
        "max_pressure": 185,
        "freeze_temp": 245,
    },
    "PEEK": {
        "melt_temp": (360, 400),
        "mold_temp": (160, 200),
        "inj_speed": (15, 50),
        "pack_pressure": (100, 140),
        "pack_time": (10, 25),
        "max_shear_rate": 20000,
        "max_pressure": 190,
        "freeze_temp": 330,
    },
    "PEI": {
        "melt_temp": (330, 370),
        "mold_temp": (150, 180),
        "inj_speed": (15, 45),
        "pack_pressure": (95, 135),
        "pack_time": (10, 22),
        "max_shear_rate": 20000,
        "max_pressure": 185,
        "freeze_temp": 300,
    },
    "PSU": {
        "melt_temp": (315, 350),
        "mold_temp": (145, 175),
        "inj_speed": (15, 50),
        "pack_pressure": (90, 125),
        "pack_time": (8, 20),
        "max_shear_rate": 20000,
        "max_pressure": 180,
        "freeze_temp": 290,
    },
    "LCP": {
        "melt_temp": (320, 360),
        "mold_temp": (140, 170),
        "inj_speed": (20, 60),
        "pack_pressure": (85, 130),
        "pack_time": (6, 18),
        "max_shear_rate": 25000,
        "max_pressure": 180,
        "freeze_temp": 300,
    },
    "PMMA": {
        "melt_temp": (220, 250),
        "mold_temp": (65, 85),
        "inj_speed": (20, 60),
        "pack_pressure": (60, 90),
        "pack_time": (5, 12),
        "max_shear_rate": 25000,
        "max_pressure": 130,
        "freeze_temp": 105,
    },
    "HIPS": {
        "melt_temp": (210, 240),
        "mold_temp": (55, 75),
        "inj_speed": (25, 70),
        "pack_pressure": (55, 85),
        "pack_time": (5, 12),
        "max_shear_rate": 30000,
        "max_pressure": 130,
        "freeze_temp": 95,
    },
    "PVC": {
        "melt_temp": (170, 200),
        "mold_temp": (50, 70),
        "inj_speed": (20, 55),
        "pack_pressure": (60, 85),
        "pack_time": (5, 12),
        "max_shear_rate": 22000,
        "max_pressure": 125,
        "freeze_temp": 90,
    },
    "TPU": {
        "melt_temp": (180, 210),
        "mold_temp": (40, 60),
        "inj_speed": (20, 60),
        "pack_pressure": (55, 80),
        "pack_time": (4, 10),
        "max_shear_rate": 22000,
        "max_pressure": 120,
        "freeze_temp": 95,
    },
    "TPE": {
        "melt_temp": (175, 205),
        "mold_temp": (35, 55),
        "inj_speed": (20, 60),
        "pack_pressure": (50, 75),
        "pack_time": (4, 10),
        "max_shear_rate": 22000,
        "max_pressure": 115,
        "freeze_temp": 90,
    },
    "EPDM": {
        "melt_temp": (170, 200),
        "mold_temp": (30, 50),
        "inj_speed": (20, 55),
        "pack_pressure": (45, 70),
        "pack_time": (4, 10),
        "max_shear_rate": 20000,
        "max_pressure": 110,
        "freeze_temp": 85,
    },
})


def load_cae_data(file_or_path) -> pd.DataFrame:
    """
    CSV column reference:
      x, y, pressure, temperature, fill_time, velocity_x, velocity_y
    x,y : grid coordinates (mm)
    pressure : injection pressure (MPa)
    temperature : melt temperature (°C)
    fill_time : fill time (s)
    """
    if hasattr(file_or_path, "read"):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    required = ["x", "y", "pressure", "temperature", "fill_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def analyze_cae(df: pd.DataFrame, material: str = "PC+ABS") -> dict:
    """
    Full CAE result analysis → returns defect risk scores + optimal conditions
    """
    limits = PROCESS_LIMITS.get(material, PROCESS_LIMITS["PC+ABS"])

    pressure = df["pressure"].values
    temperature = df["temperature"].values
    fill_time = df["fill_time"].values

    # ── Basic statistics ──────────────────────────────────
    stats = {
        "max_pressure_MPa": float(pressure.max()),
        "avg_pressure_MPa": float(pressure.mean()),
        "max_temperature_C": float(temperature.max()),
        "avg_temperature_C": float(temperature.mean()),
        "fill_time_s": float(fill_time.max()),
        "pressure_gradient": float(_calc_gradient(df)),
    }

    # ── Defect risk scoring ────────────────────────
    defect_risks = _score_defect_risks(df, limits, stats)

    # ── Derive optimal conditions ─────────────────────────────
    optimal = _derive_optimal_conditions(stats, limits, defect_risks)

    # ── Build grid maps (for visualization) ──────────────────
    grid_maps = _build_grid_maps(df)

    return {
        "stats": stats,
        "defect_risks": defect_risks,
        "optimal_conditions": optimal,
        "grid_maps": grid_maps,
        "material": material,
    }


def _calc_gradient(df: pd.DataFrame) -> float:
    """Estimate pressure gradient (gate → end of fill)"""
    if "x" not in df.columns:
        return 0.0
    sorted_df = df.sort_values("x")
    if len(sorted_df) < 2:
        return 0.0
    p_range = sorted_df["pressure"].max() - sorted_df["pressure"].min()
    x_range = sorted_df["x"].max() - sorted_df["x"].min()
    return float(p_range / x_range) if x_range > 0 else 0.0


def _score_defect_risks(df, limits, stats) -> dict:
    """
    Convert each defect risk to a 0~1 score
    0 = safe / 1 = very dangerous
    """
    risks = {}

    # Short shot risk (insufficient pressure)
    p_max = stats["max_pressure_MPa"]
    p_limit = limits["max_pressure"]
    if p_max < p_limit * 0.4:
        risks["short_shot"] = {"score": 0.8, "level": "HIGH",
                               "detail": f"Max pressure {p_max:.0f} MPa — possible short fill"}
    elif p_max < p_limit * 0.6:
        risks["short_shot"] = {"score": 0.4, "level": "MED",
                               "detail": "Fill borderline — consider increasing injection speed"}
    else:
        risks["short_shot"] = {"score": 0.1, "level": "LOW", "detail": "Normal"}

    # Weld line risk (estimated temperature drop at fill end)
    temp_std = float(df["temperature"].std())
    if temp_std > 20:
        risks["weld_line"] = {"score": 0.7, "level": "HIGH",
                              "detail": f"Temperature deviation {temp_std:.1f}°C — Weld line expected"}
    elif temp_std > 10:
        risks["weld_line"] = {"score": 0.4, "level": "MED",
                              "detail": "Temperature deviation caution — review gate location"}
    else:
        risks["weld_line"] = {"score": 0.1, "level": "LOW", "detail": "Normal"}

    # Sink mark risk (estimated from low pressure + thick cross-section)
    p_avg = stats["avg_pressure_MPa"]
    if p_avg < p_limit * 0.3:
        risks["sink_mark"] = {"score": 0.6, "level": "HIGH",
                              "detail": "Possible insufficient packing pressure — sink risk"}
    else:
        risks["sink_mark"] = {"score": 0.2, "level": "LOW", "detail": "Normal"}

    # Warpage risk (uneven temperature distribution)
    t_max = stats["max_temperature_C"]
    t_avg = stats["avg_temperature_C"]
    if (t_max - t_avg) > 30:
        risks["warpage"] = {"score": 0.7, "level": "HIGH",
                            "detail": f"Hot spot {t_max:.0f}°C — uneven cooling, warpage risk"}
    elif (t_max - t_avg) > 15:
        risks["warpage"] = {"score": 0.4, "level": "MED",
                            "detail": "Cooling channel placement review recommended"}
    else:
        risks["warpage"] = {"score": 0.1, "level": "LOW", "detail": "Normal"}

    # Air trap (simple estimation at fill end)
    fill_uniformity = float(df["fill_time"].std() / df["fill_time"].mean()) if df["fill_time"].mean() > 0 else 0
    if fill_uniformity > 0.4:
        risks["air_trap"] = {"score": 0.6, "level": "HIGH",
                             "detail": "Uneven fill — air trap risk zones detected"}
    else:
        risks["air_trap"] = {"score": 0.2, "level": "LOW", "detail": "Normal"}

    return risks


def _derive_optimal_conditions(stats, limits, defect_risks) -> dict:
    """
    Derive optimal process conditions from defect risk scores + material limits
    Rule-based weighted optimization
    """
    # Start each parameter at midpoint and adjust based on risk scores
    melt_lo, melt_hi = limits["melt_temp"]
    mold_lo, mold_hi = limits["mold_temp"]
    spd_lo, spd_hi = limits["inj_speed"]
    pack_lo, pack_hi = limits["pack_pressure"]
    ptime_lo, ptime_hi = limits["pack_time"]

    # High warpage risk → lower mold temperature, increase pack time
    warpage_score = defect_risks.get("warpage", {}).get("score", 0)
    sink_score = defect_risks.get("sink_mark", {}).get("score", 0)
    short_score = defect_risks.get("short_shot", {}).get("score", 0)

    # Optimal melt temp: lower to reduce warpage risk
    melt_opt = melt_hi - (melt_hi - melt_lo) * warpage_score * 0.5
    mold_opt = mold_lo + (mold_hi - mold_lo) * 0.4  # lower-mid range

    # Injection speed: raise if short shot risk detected
    spd_opt = spd_lo + (spd_hi - spd_lo) * (0.5 + short_score * 0.3)

    # Packing pressure: raise if sink mark risk detected
    pack_opt = pack_lo + (pack_hi - pack_lo) * (0.5 + sink_score * 0.4)

    # Packing time: increase if warpage/sink risk detected
    ptime_opt = ptime_lo + (ptime_hi - ptime_lo) * (0.4 + max(warpage_score, sink_score) * 0.4)

    return {
        "melt_temperature": {
            "range": limits["melt_temp"],
            "optimal": round(melt_opt, 1),
            "unit": "°C",
        },
        "mold_temperature": {
            "range": limits["mold_temp"],
            "optimal": round(mold_opt, 1),
            "unit": "°C",
        },
        "injection_speed": {
            "range": limits["inj_speed"],
            "optimal": round(spd_opt, 1),
            "unit": "mm/s",
        },
        "packing_pressure": {
            "range": limits["pack_pressure"],
            "optimal": round(pack_opt, 1),
            "unit": "MPa",
        },
        "packing_time": {
            "range": limits["pack_time"],
            "optimal": round(ptime_opt, 1),
            "unit": "sec",
        },
    }


def _build_grid_maps(df: pd.DataFrame) -> dict:
    """
    Build smooth 2D grid maps from x,y coordinates (with interpolation)
    """
    if "x" not in df.columns or "y" not in df.columns:
        return {}

    try:
        # 1. Set grid range (slightly padded beyond actual data)
        points = df[['x', 'y']].values
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # Resolution setting (e.g. 100x100 grid)
        grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

        maps = {}
        for col in ["pressure", "temperature", "fill_time"]:
            if col not in df.columns:
                continue
            
            values = df[col].values
            
            # 2. Fill gaps with linear interpolation
            # Instead of forcing zeros, compute from surrounding reference points.
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            
            # 3. Fill NaN values at edges after interpolation using ffill/bfill
            grid_z = pd.DataFrame(grid_z).ffill().bfill().values
            
            # 4. Apply Gaussian smoothing for visualization
            grid_z = gaussian_filter(grid_z.astype(float), sigma=1.0)
            
            # Plotly Heatmap expects [y, x] structure — apply transpose (.T)
            maps[col] = {
                "z": grid_z.T.tolist(), 
                "x": np.linspace(x_min, x_max, 100).tolist(),
                "y": np.linspace(y_min, y_max, 100).tolist(),
            }
        return maps
    except Exception as e:
        print(f"Grid mapping error: {e}")
        return {}


def generate_sample_cae_csv(n_points: int = 200, material: str = "PC+ABS") -> pd.DataFrame:
    """
    Generate sample data for use when no actual CAE results are available
    Physically plausible pressure/temperature distribution simulation
    """
    np.random.seed(42)
    limits = PROCESS_LIMITS.get(material, PROCESS_LIMITS["PC+ABS"])

    # Grid coordinates (assuming 100mm x 60mm part)
    x = np.random.uniform(0, 100, n_points)
    y = np.random.uniform(0, 60, n_points)

    # Gate assumed at (0, 30)
    gate_x, gate_y = 0, 30
    dist_from_gate = np.sqrt((x - gate_x) ** 2 + (y - gate_y) ** 2)
    max_dist = dist_from_gate.max()

    # Pressure: decreases with distance from gate + noise
    p_max = limits["max_pressure"] * 0.85
    pressure = p_max * (1 - dist_from_gate / max_dist * 0.75) + np.random.normal(0, 3, n_points)
    pressure = np.clip(pressure, 5, limits["max_pressure"])

    # Temperature: high near gate, lower at fill end + hot spot
    t_lo, t_hi = limits["melt_temp"]
    base_temp = (t_lo + t_hi) / 2
    temperature = base_temp - dist_from_gate / max_dist * 25 + np.random.normal(0, 5, n_points)
    # Add hot spot (upper-right region)
    hot_mask = (x > 70) & (y > 45)
    temperature[hot_mask] += 15
    temperature = np.clip(temperature, limits["mold_temp"][0] + 20, t_hi + 10)

    # Fill time: proportional to distance + thickness variation
    fill_time = dist_from_gate / max_dist * 1.8 + np.random.normal(0, 0.05, n_points)
    fill_time = np.clip(fill_time, 0.01, 2.5)

    df = pd.DataFrame({
        "x": np.round(x, 2),
        "y": np.round(y, 2),
        "pressure": np.round(pressure, 2),
        "temperature": np.round(temperature, 2),
        "fill_time": np.round(fill_time, 3),
    })

    return df
# MIM materials
PROCESS_LIMITS.update({
    "17-4PH": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 140),
        "pack_time": (5, 20),
        "max_shear_rate": 20000,
        "max_pressure": 150,
        "freeze_temp": 120,
    },
    "316L": {
        "melt_temp": (175, 200),
        "mold_temp": (30, 50),
        "inj_speed": (15, 50),
        "pack_pressure": (80, 130),
        "pack_time": (5, 20),
        "max_shear_rate": 18000,
        "max_pressure": 140,
        "freeze_temp": 120,
    },
    "Ti-6Al-4V": {
        "melt_temp": (180, 210),
        "mold_temp": (35, 55),
        "inj_speed": (10, 40),
        "pack_pressure": (90, 150),
        "pack_time": (8, 25),
        "max_shear_rate": 15000,
        "max_pressure": 160,
        "freeze_temp": 130,
    },
})
