"""
LAYER 5C: MINIMAL & CONVERSATIONAL EXPLAINER
One-liner explanations, like a real person talking
"""

from typing import Dict


class MinimalExplainer:
    """Generates short, one-line explanations."""
    
    def __init__(self):
        self.explanations = {
            "direct_injection": {
                "emoji": "",
                "line": "Your prompt tried to override the AI's safety rules."
            },
            "jailbreak": {
                "emoji": "",
                "line": "Your prompt tried to break the AI's safety rules."
            },
            "story_jailbreak": {
                "emoji": "",
                "line": "Your prompt used a story to try to bypass security."
            },
            "system_extraction": {
                "emoji": "",
                "line": "Your prompt asked for the AI's private instructions."
            },
            "data_extraction": {
                "emoji": "",
                "line": "Your prompt tried to access private information."
            },
            "tool_injection": {
                "emoji": "",
                "line": "Your prompt tried to run commands."
            },
            "context_poisoning": {
                "emoji": "",
                "line": "Your prompt tried to change how the AI remembers things."
            },
            "multi_turn": {
                "emoji": "",
                "line": "Your prompt was building up to something over several steps."
            },
            "obfuscation": {
                "emoji": "",
                "line": "Your prompt was using hidden tricks."
            },
            "unknown": {
                "emoji": "",
                "line": "Your prompt was flagged by our security system."
            }
        }
    
    def get_explanation(self, attack_type: str) -> Dict:
        """Get minimal explanation for an attack type."""
        return self.explanations.get(attack_type, self.explanations["unknown"])