from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

PARAM_NAMES = [
    "asc_car",
    "asc_pt",
    "beta_time",
    "beta_cost",
    "beta_walk",
    "beta_park",
    "beta_wait",
    "beta_caravail_car",
    "beta_income_high_car",
    "beta_age_senior_pt",
    "beta_female_ebike",
]

DEFAULT_RP_SCHEMA = {
    "age_group": ["18-30", "31-45", "46-60", "60+"],
    "income_group": ["<5k", "5k-10k", "10k-20k", ">20k"],
    "gender": ["male", "female"],
    "family_size": [1, 2, 3, 4, 5],
    "car_availability": [0, 1],
    "occupation": ["student", "office", "service", "other"],
}

DEFAULT_CONFIG = {
    "dynamic_ppo": {
        "seed": 42,
        "input_dim": 6,
        "train_respondents": 300,
        "train_epochs": 80,
        "train_lr": 0.03,
    },
    "self_attention": {
        "seed": 42,
        "window_length": 32,
        "hidden_dim": 64,
        "num_heads": 4,
        "sample_target_dim": 16,
        "target_sample_size": 200,
        "explore_epsilon": 0.15,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "train_respondents": 300,
        "train_epochs": 200,
        "train_lr": 0.03,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "target_kl": 0.03,
        "online_lr": 0.005,
        "online_epochs": 2,
        "online_batch_size": 128,
        "count_loss_weight": 1.0,
        "slot_select_loss_weight": 0.8,
        "mask_loss_weight": 0.8,
        "value_loss_weight": 1.2,
        "score_loss_weight": 0.4,
        "sample_target_loss_weight": 0.3,
    },
    "data_sources": {
        "popsim_stats_path": "popSimStats_gz_template.json",
        "zone_geojson_path": "gz_districts_template.geojson",
    },
}


def param_vector_from_dict(param_dict: dict[str, float]) -> list[float]:
    return [float(param_dict[name]) for name in PARAM_NAMES]


def rp_flags(profile: dict) -> dict[str, float]:
    age = profile.get("age_group", "31-45")
    income = profile.get("income_group", "5k-10k")
    gender = profile.get("gender", "male")
    return {
        "car_availability": float(profile.get("car_availability", 0) or 0),
        "income_high": 1.0 if income in {"10k-20k", ">20k"} else 0.0,
        "age_senior": 1.0 if age in {"46-60", "60+"} else 0.0,
        "female": 1.0 if gender == "female" else 0.0,
    }


def default_profile() -> dict:
    return {
        "age_group": "31-45",
        "income_group": "5k-10k",
        "gender": "male",
        "family_size": 3,
        "car_availability": 1,
        "occupation": "office",
    }
