"""
Clinical Biomarker Utilities for Liver Fibrosis Staging.

Provides standard clinical hepatology biomarker calculations, risk cutoffs,
and physiological cohort generator for Multi-Modal Diagnostic Fusion:
- FIB-4 Index (Sterling et al., Hepatology 2006)
- APRI: AST to Platelet Ratio Index (Wai et al., Hepatology 2003)
- De Ritis Ratio: AST / ALT
- FibroScan Transient Elastography (kPa) cutoffs
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from pathlib import Path


def compute_fib4(age: float, ast: float, alt: float, platelets: float) -> float:
    """
    Compute FIB-4 Index: (Age * AST) / (Platelets * sqrt(ALT))
    
    Args:
        age: Patient age in years
        ast: Aspartate aminotransferase in U/L
        alt: Alanine aminotransferase in U/L
        platelets: Platelet count in 10^9 / L
        
    Returns:
        float: Calculated FIB-4 index rounded to 2 decimal places.
    """
    if platelets <= 0 or alt <= 0:
        return 0.0
    val = (age * ast) / (platelets * np.sqrt(alt))
    return float(np.round(val, 2))


def compute_apri(ast: float, platelets: float, uln: float = 40.0) -> float:
    """
    Compute APRI (AST to Platelet Ratio Index):
    APRI = ((AST / ULN) / Platelets) * 100
    
    Args:
        ast: Aspartate aminotransferase in U/L
        platelets: Platelet count in 10^9 / L
        uln: Upper limit of normal for AST (standard 40 U/L)
        
    Returns:
        float: Calculated APRI score rounded to 2 decimal places.
    """
    if platelets <= 0 or uln <= 0:
        return 0.0
    val = ((ast / uln) / platelets) * 100.0
    return float(np.round(val, 2))


def compute_ast_alt_ratio(ast: float, alt: float) -> float:
    """
    Compute De Ritis Ratio: AST / ALT
    Normal < 1.0; Ratio > 1.0 or > 2.0 frequently indicates progressive cirrhosis.
    """
    if alt <= 0:
        return 0.0
    return float(np.round(ast / alt, 2))


def interpret_fib4(fib4: float) -> Dict[str, str]:
    """
    Clinical risk tiering based on AASLD / EASL guidelines:
    - FIB-4 < 1.45: Low risk of advanced fibrosis (rule out F3-F4, NPV > 90%)
    - 1.45 <= FIB-4 <= 3.25: Indeterminate risk (requires second-line testing / biopsy)
    - FIB-4 > 3.25: High risk of advanced fibrosis / cirrhosis (PPV ~ 65-80%)
    """
    if fib4 < 1.45:
        return {
            'risk_level': 'Low Risk',
            'metavir_equiv': 'F0 - F1',
            'color': '#28a745',  # green
            'badge': 'success',
            'action': 'Routine monitoring recommended; advanced fibrosis unlikely (NPV > 90%).'
        }
    elif fib4 <= 3.25:
        return {
            'risk_level': 'Indeterminate Risk',
            'metavir_equiv': 'F2 - F3',
            'color': '#ffc107',  # yellow/amber
            'badge': 'warning',
            'action': 'Intermediate zone; second-line testing (VCTE/FibroScan or biopsy) advised.'
        }
    else:
        return {
            'risk_level': 'High Risk',
            'metavir_equiv': 'F3 - F4 (Cirrhosis)',
            'color': '#dc3545',  # red
            'badge': 'danger',
            'action': 'High suspicion of advanced fibrosis/cirrhosis; hepatology referral needed.'
        }


def interpret_apri(apri: float) -> Dict[str, str]:
    """
    Clinical risk tiering for APRI:
    - APRI < 0.5: Significant fibrosis unlikely (F0-F1)
    - 0.5 <= APRI <= 1.5: Indeterminate zone
    - APRI > 1.5: Significant fibrosis / cirrhosis likely
    """
    if apri < 0.5:
        return {
            'risk_level': 'Low Risk',
            'metavir_equiv': 'F0 - F1',
            'color': '#28a745',
            'badge': 'success',
            'action': 'Significant fibrosis unlikely.'
        }
    elif apri <= 1.5:
        return {
            'risk_level': 'Moderate / Indeterminate',
            'metavir_equiv': 'F2',
            'color': '#ffc107',
            'badge': 'warning',
            'action': 'Intermediate zone.'
        }
    else:
        return {
            'risk_level': 'High Risk',
            'metavir_equiv': 'F3 - F4',
            'color': '#dc3545',
            'badge': 'danger',
            'action': 'Significant fibrosis / cirrhosis probable.'
        }


def interpret_fibroscan(kpa: float) -> Dict[str, str]:
    """
    Liver stiffness measurement (LSM in kPa via FibroScan VCTE):
    - < 7.0 kPa: Normal / Mild (F0 - F1)
    - 7.0 - 9.5 kPa: Significant fibrosis (F2)
    - 9.5 - 12.5 kPa: Severe fibrosis (F3)
    - > 12.5 kPa: Cirrhosis (F4)
    """
    if kpa < 7.0:
        return {'stage': 'F0-F1', 'desc': 'Normal to mild fibrosis', 'color': '#28a745'}
    elif kpa < 9.5:
        return {'stage': 'F2', 'desc': 'Moderate / Significant portal fibrosis', 'color': '#17a2b8'}
    elif kpa < 12.5:
        return {'stage': 'F3', 'desc': 'Severe / Bridging fibrosis', 'color': '#fd7e14'}
    else:
        return {'stage': 'F4', 'desc': 'Cirrhosis', 'color': '#dc3545'}


def generate_matched_clinical_cohort(
    labels: np.ndarray,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic, physiologically calibrated clinical tabular data
    conditioned on true METAVIR stages (F0-F4) based on AASLD/EASL literature distributions.
    
    Biomarker distributions per stage:
      F0: Age ~ 43, AST ~ 28, ALT ~ 32, Platelets ~ 265, FibroScan ~ 4.8 kPa, BMI ~ 24.5
      F1: Age ~ 48, AST ~ 38, ALT ~ 46, Platelets ~ 235, FibroScan ~ 6.4 kPa, BMI ~ 26.2
      F2: Age ~ 53, AST ~ 52, ALT ~ 58, Platelets ~ 195, FibroScan ~ 8.4 kPa, BMI ~ 27.8
      F3: Age ~ 57, AST ~ 72, ALT ~ 68, Platelets ~ 155, FibroScan ~ 11.2 kPa, BMI ~ 29.1
      F4: Age ~ 61, AST ~ 95, ALT ~ 74, Platelets ~ 105, FibroScan ~ 19.5 kPa, BMI ~ 29.8
    """
    rng = np.random.default_rng(random_seed)
    n = len(labels)
    
    stage_params = {
        0: {'age': (43, 8), 'ast': (28, 6), 'alt': (32, 8), 'plt': (265, 35), 'kpa': (4.8, 0.8), 'bmi': (24.5, 2.5), 'alb': (4.4, 0.3), 'bili': (0.7, 0.2)},
        1: {'age': (48, 8), 'ast': (38, 8), 'alt': (46, 11), 'plt': (235, 32), 'kpa': (6.4, 0.9), 'bmi': (26.2, 2.8), 'alb': (4.2, 0.3), 'bili': (0.8, 0.2)},
        2: {'age': (53, 7), 'ast': (52, 12), 'alt': (58, 14), 'plt': (195, 30), 'kpa': (8.4, 1.1), 'bmi': (27.8, 3.0), 'alb': (4.0, 0.4), 'bili': (1.0, 0.3)},
        3: {'age': (57, 7), 'ast': (72, 16), 'alt': (68, 16), 'plt': (155, 28), 'kpa': (11.2, 1.3), 'bmi': (29.1, 3.2), 'alb': (3.7, 0.4), 'bili': (1.3, 0.4)},
        4: {'age': (61, 7), 'ast': (95, 22), 'alt': (74, 18), 'plt': (105, 25), 'kpa': (19.5, 4.5), 'bmi': (29.8, 3.5), 'alb': (3.2, 0.5), 'bili': (2.1, 0.6)}
    }
    
    records = []
    for i, lbl in enumerate(labels):
        p = stage_params[int(lbl)]
        age = float(np.clip(rng.normal(*p['age']), 21, 80))
        ast = float(np.clip(rng.normal(*p['ast']), 15, 250))
        alt = float(np.clip(rng.normal(*p['alt']), 12, 220))
        platelets = float(np.clip(rng.normal(*p['plt']), 30, 450))
        kpa = float(np.clip(rng.normal(*p['kpa']), 2.5, 65.0))
        bmi = float(np.clip(rng.normal(*p['bmi']), 18.0, 42.0))
        alb = float(np.clip(rng.normal(*p['alb']), 2.0, 5.2))
        bili = float(np.clip(rng.normal(*p['bili']), 0.2, 6.0))
        sex = int(rng.choice([0, 1]))  # 0=Female, 1=Male
        
        fib4 = compute_fib4(age, ast, alt, platelets)
        apri = compute_apri(ast, platelets)
        de_ritis = compute_ast_alt_ratio(ast, alt)
        
        records.append({
            'patient_id': f'PT_{i+1:05d}',
            'age': np.round(age, 1),
            'sex': sex,
            'bmi': np.round(bmi, 1),
            'ast_u_l': np.round(ast, 1),
            'alt_u_l': np.round(alt, 1),
            'platelets_10_9_l': np.round(platelets, 1),
            'albumin_g_dl': np.round(alb, 2),
            'total_bilirubin_mg_dl': np.round(bili, 2),
            'fibroscan_kpa': np.round(kpa, 2),
            'fib4_score': fib4,
            'apri_score': apri,
            'ast_alt_ratio': de_ritis,
            'true_stage': int(lbl),
            'true_metavir': f'F{int(lbl)}'
        })
        
    return pd.DataFrame(records)
