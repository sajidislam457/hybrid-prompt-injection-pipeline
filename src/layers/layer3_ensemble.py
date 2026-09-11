"""
LAYER 3: ULTRA-ENHANCED ENSEMBLE FUSION
Advanced fusion with weighted voting and category-specific boosting
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Layer3Result:
    """Output from Layer 3."""
    final_classification: bool
    confidence: float
    weighted_risk_score: float
    agreement_score: float
    model_votes: Dict[str, int]
    is_ambiguous: bool
    action: str
    attack_categories: List[str]
    category_risks: Dict[str, float]


class Layer3Ensemble:
    """
    Layer 3: Weighted fusion of Layer 2 model risks.
    Logistic/SVM fit the cleaned train set better; RF/XGBoost are shallower
    and only get a small vote so they cannot force 'ambiguous' → Layer 4.
    """

    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        ambiguous_confidence_below: float = 0.45,
        ambiguous_agreement_below: float = 0.45,
        decision_threshold: float = 0.52,
        category_boost_scale: float = 0.25,
        category_boost_min_risk: float = 0.45,
    ):
        self.model_weights = {
            "logistic": 1.3,
            "svm": 1.3,
            "xgboost": 0.4,
            "random_forest": 0.3,
            "gradient_boosting": 0.4,
            "mlp": 0.5,
        }
        if model_weights:
            for name, weight in model_weights.items():
                try:
                    self.model_weights[str(name)] = float(weight)
                except (TypeError, ValueError):
                    continue

        self.ambiguous_confidence_below = float(ambiguous_confidence_below)
        self.ambiguous_agreement_below = float(ambiguous_agreement_below)
        self.decision_threshold = float(decision_threshold)
        self.category_boost_scale = float(category_boost_scale)
        self.category_boost_min_risk = float(category_boost_min_risk)

        # Soft caps — large boosts were shoving benign lookalikes over the cut.
        self.category_boosts = {
            "direct_override": 0.05,
            "obfuscation": 0.06,
            "role_impersonation": 0.04,
            "emotional_manipulation": 0.02,
            "indirect_injection": 0.04,
            "context_tampering": 0.04,
            "system_extraction": 0.05,
            "data_extraction": 0.05,
            "tool_injection": 0.06,
            "multi_turn": 0.02,
            "social_engineering": 0.03,
            "story_based": 0.04,
        }
    
    def fuse(self, layer2_result: Dict) -> Layer3Result:
        """Fuse predictions with category-specific boosting."""
        predictions = layer2_result.get('predictions', [])
        individual_risks = layer2_result.get('individual_risks', {})
        attack_categories = layer2_result.get('attack_categories', [['unknown']])
        
        if not predictions:
            return Layer3Result(
                final_classification=False,
                confidence=0.0,
                weighted_risk_score=0.0,
                agreement_score=0.0,
                model_votes={},
                is_ambiguous=True,
                action="AMBIGUOUS",
                attack_categories=[],
                category_risks={}
            )
        
        votes = {}
        for name, risks in individual_risks.items():
            if risks:
                pred = 1 if risks[0] > 0.5 else 0
                votes[name] = pred

        # Probability fusion (not 0/1 votes) so a weak model cannot flip the score.
        weighted_risk = 0.0
        total_weight = 0.0
        agree_weight = 0.0

        for name, risks in individual_risks.items():
            if not risks:
                continue
            weight = float(self.model_weights.get(name, 1.0))
            if weight <= 0:
                continue
            p = float(risks[0])
            weighted_risk += p * weight
            total_weight += weight
            if votes.get(name, 0) == 1:
                agree_weight += weight

        weighted_risk = weighted_risk / total_weight if total_weight > 0 else 0.0
        majority_attack = agree_weight >= (total_weight * 0.5) if total_weight else False
        agreement_score = (
            (agree_weight / total_weight) if majority_attack
            else (1.0 - agree_weight / total_weight) if total_weight
            else 0.0
        )
        
        # Category boosts only reinforce an already-suspicious score (cuts benign FPs).
        categories = attack_categories[0] if attack_categories else ["unknown"]
        if weighted_risk >= self.category_boost_min_risk:
            for category in categories:
                boost = self.category_boosts.get(category, 0.0) * self.category_boost_scale
                weighted_risk = min(weighted_risk + boost, 1.0)

        # Calculate category risks
        category_risks = {}
        for category in set(categories):
            if category != "unknown":
                category_risks[category] = weighted_risk

        thr = self.decision_threshold
        final_classification = weighted_risk > thr

        if not votes:
            agreement_score = 0.0

        # Keep mid-point confidence for escalation gates (L2b/L4), independent of thr.
        confidence = abs(weighted_risk - 0.5) * 2

        is_ambiguous = (
            confidence < self.ambiguous_confidence_below
            or agreement_score < self.ambiguous_agreement_below
        )

        return Layer3Result(
            final_classification=final_classification,
            confidence=confidence,
            weighted_risk_score=weighted_risk,
            agreement_score=agreement_score,
            model_votes=votes,
            is_ambiguous=is_ambiguous,
            action="AMBIGUOUS" if is_ambiguous else "CONFIDENT",
            attack_categories=categories,
            category_risks=category_risks
        )