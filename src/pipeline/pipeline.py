"""
ULTIMATE PIPELINE - COMPLETE 500+ PATTERN DETECTION
"""

import os
import time
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from pathlib import Path

from src.layers.layer1_prefilter import Layer1Prefilter, Layer1Result
from src.layers.layer2_classifiers import Layer2Classifier
from src.layers.layer2b_transformer import Layer2BTransformer
from src.layers.layer3_ensemble import Layer3Ensemble, Layer3Result
from src.layers.layer4_llm_judge import Layer4LLMJudge, Layer4Result
from src.layers.layer5_natural import NaturalConversationalGenerator
from src.layers.text_normalizer import TextNormalizer
from src.layers.attack_retrieval import AttackRetriever
from src.layers.attack_typer import AttackTypeDetector
from src.utils.helpers import load_config
from src.utils.decision_logger import DecisionLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    layer1: Layer1Result
    layer2: Optional[Dict] = None
    layer2b: Optional[Dict] = None
    layer3: Optional[Layer3Result] = None
    layer4: Optional[Layer4Result] = None
    layer5: Optional[Any] = None
    retrieval: Optional[Dict] = None
    normalization: Optional[Dict] = None
    is_malicious: bool = False
    final_risk_score: float = 0.0
    attack_type: str = "unknown"
    attack_display_name: str = "Unknown"
    attack_categories: List[str] = field(default_factory=list)
    severity: str = "low"
    action: str = "ALLOW"
    decision_source: str = "unknown"
    explanation: Dict = field(default_factory=dict)
    safe_alternative: Optional[str] = None
    processing_time: Dict = field(default_factory=dict)
    normalized_text: Optional[str] = None


class PromptInjectionPipeline:
    def __init__(self, model_dir: str = "./models/detector", use_llm: bool = False):
        self.model_dir = model_dir
        self.config = load_config(Path("configs/config.yaml")) or {}
        layers = self.config.get("layers") or {}
        flags = self.config.get("feature_flags") or {}
        logging_cfg = self.config.get("logging") or {}

        layer2b_cfg = layers.get("layer2b") or {}
        layer4_cfg = layers.get("layer4") or {}
        layer5_cfg = layers.get("layer5") or {}
        retrieval_cfg = layers.get("retrieval") or {}
        normalizer_cfg = layers.get("normalizer") or {}

        self.flags = {
            "phase2": bool(flags.get("phase2_transformer", True)),
            "phase3": bool(flags.get("phase3_llm_judge", True)),
            "phase4": bool(flags.get("phase4_retrieval_normalization", True)),
            "phase5": bool(flags.get("phase5_decision_logging", True)),
        }

        self.layer1 = Layer1Prefilter()
        self.layer2 = Layer2Classifier(model_dir)
        # Held-out/ablation workers set LAYER2B_DEVICE (cuda/cpu). Live API stays
        # on config (default cpu) so the 4GB GPU is free for long eval jobs.
        layer2b_device = (
            os.environ.get("LAYER2B_DEVICE")
            or layer2b_cfg.get("device")
            or "cpu"
        )
        self.layer2b = Layer2BTransformer(
            enabled=self.flags["phase2"] and bool(layer2b_cfg.get("enabled", True)),
            model_name=layer2b_cfg.get("model_name", "protectai/deberta-v3-base-prompt-injection-v2"),
            threshold=float(layer2b_cfg.get("threshold", 0.78)),
            use_transformers=bool(layer2b_cfg.get("use_transformers", False)),
            device=str(layer2b_device),
        )
        self.layer2b_gate_confidence_below = float(layer2b_cfg.get("gate_confidence_below", 0.40))
        self.layer2b_run_on_ambiguous = bool(layer2b_cfg.get("run_on_ambiguous", True))
        self.layer2b_override_min_risk = float(layer2b_cfg.get("override_min_risk", 1.01))
        self.layer2b_enable_force_block = bool(layer2b_cfg.get("enable_force_block", False))
        self.layer2b_soft_merge = float(layer2b_cfg.get("soft_merge", 0.85))
        self.layer2b_soft_merge_only_ambiguous = bool(
            layer2b_cfg.get("soft_merge_only_ambiguous", True)
        )

        self.layer3 = Layer3Ensemble(
            model_weights=(layers.get("layer3") or {}).get("weights"),
            ambiguous_confidence_below=float(
                (layers.get("layer3") or {}).get("ambiguous_confidence_below", 0.45)
            ),
            ambiguous_agreement_below=float(
                (layers.get("layer3") or {}).get("ambiguous_agreement_below", 0.45)
            ),
            decision_threshold=float(
                (layers.get("layer3") or {}).get("decision_threshold", 0.52)
            ),
            category_boost_scale=float(
                (layers.get("layer3") or {}).get("category_boost_scale", 0.25)
            ),
            category_boost_min_risk=float(
                (layers.get("layer3") or {}).get("category_boost_min_risk", 0.45)
            ),
        )
        self.layer4 = Layer4LLMJudge(
            use_real_llm=bool(layer4_cfg.get("use_real_llm", use_llm)),
            enabled=self.flags["phase3"] and bool(layer4_cfg.get("enabled", True)),
            timeout_sec=float(layer4_cfg.get("timeout_sec", 12)),
            max_calls_per_minute=int(layer4_cfg.get("max_calls_per_minute", 20)),
            model=layer4_cfg.get("model"),
            heuristic_block_threshold=float(layer4_cfg.get("heuristic_block_threshold", 0.75)),
        )
        self.layer4_ambiguous_only = bool(layer4_cfg.get("ambiguous_only", True))
        self.layer4_min_confidence_for_skip = float(layer4_cfg.get("min_confidence_for_skip", 0.40))
        self.layer4_require_ensemble_corroboration = bool(
            layer4_cfg.get("require_ensemble_corroboration", True)
        )
        self.layer4_type_force_block = bool(layer4_cfg.get("type_force_block", False))
        layer3_cfg = layers.get("layer3") or {}
        # Precision gate: drop weak ensemble blocks that lack agreement / high risk.
        self.layer3_block_min_risk = float(layer3_cfg.get("block_min_risk", 0.58))
        self.layer3_block_min_agreement = float(layer3_cfg.get("block_min_agreement", 0.65))
        self.layer3_strong_risk = float(layer3_cfg.get("strong_risk", 0.75))

        self.layer5 = NaturalConversationalGenerator(
            intent_preserving=bool(layer5_cfg.get("intent_preserving", True)),
            use_llm_rewrite=bool(layer5_cfg.get("use_llm_rewrite", False)),
            fidelity_threshold=float(layer5_cfg.get("fidelity_threshold", 0.35)),
            clarify_threshold=float(layer5_cfg.get("clarify_threshold", 0.45)),
        )

        self.normalizer = TextNormalizer()
        self.normalizer_enabled = self.flags["phase4"] and bool(normalizer_cfg.get("enabled", True))
        self.retriever = AttackRetriever(
            bank_path=retrieval_cfg.get("bank_path", "data/attack_bank.json"),
            threshold=float(retrieval_cfg.get("threshold", 0.42)),
            top_k=int(retrieval_cfg.get("top_k", 3)),
            enabled=self.flags["phase4"] and bool(retrieval_cfg.get("enabled", True)),
        )
        self.retrieval_force_block_score = float(retrieval_cfg.get("force_block_score", 0.85))
        self.decision_logger = DecisionLogger(
            log_path=logging_cfg.get("decisions_file", "logs/decisions.jsonl"),
            enabled=self.flags["phase5"],
        )
        self.attack_typer = AttackTypeDetector(min_score=1.8)
        self.is_ready = False
        self._team_overrides: Dict[str, Dict[str, Any]] = {}
        # Bulk accuracy jobs set ACCURACY_DISABLE_DECISION_LOG=1 — keep per-sample
        # prints off or a 40k held-out run writes multi-GB worker logs.
        _quiet = os.environ.get("ACCURACY_DISABLE_DECISION_LOG", "").strip().lower()
        self.verbose = _quiet not in {"1", "true", "yes"}
        self._reload_team_overrides()
        self._compile_patterns()
        self._ablation_mode = "full"

    def _vprint(self, *args: Any, **kwargs: Any) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def apply_ablation(self, mode: str = "full") -> None:
        """
        Research ablations (paper Table X):
          full            — all layers
          classical_only  — Layer1+2+3 only (no 2b, retrieval, layer4)
          no_layer2b      — disable semantic Layer 2b
          no_retrieval    — disable attack_bank retrieval
          no_layer4       — disable judge
        """
        mode = (mode or "full").strip().lower()
        self._ablation_mode = mode

        # Reset to config defaults first
        layer2b_cfg = (self.config.get("layers") or {}).get("layer2b") or {}
        layer4_cfg = (self.config.get("layers") or {}).get("layer4") or {}
        retrieval_cfg = (self.config.get("layers") or {}).get("retrieval") or {}

        self.layer2b.enabled = self.flags["phase2"] and bool(layer2b_cfg.get("enabled", True))
        self.layer4.enabled = self.flags["phase3"] and bool(layer4_cfg.get("enabled", True))
        self.retriever.enabled = self.flags["phase4"] and bool(retrieval_cfg.get("enabled", True))
        if self.retriever.enabled and self.retriever.vectorizer is None:
            try:
                self.retriever._build()
            except Exception:
                pass

        if mode == "classical_only":
            self.layer2b.enabled = False
            self.layer4.enabled = False
            self.retriever.enabled = False
        elif mode == "no_layer2b":
            self.layer2b.enabled = False
        elif mode == "no_retrieval":
            self.retriever.enabled = False
        elif mode == "no_layer4":
            self.layer4.enabled = False
        elif mode != "full":
            logger.warning("Unknown ablation mode %s — using full", mode)
            self._ablation_mode = "full"

        logger.info(
            "Ablation=%s | layer2b=%s retrieval=%s layer4=%s",
            self._ablation_mode,
            self.layer2b.enabled,
            self.retriever.enabled,
            self.layer4.enabled,
        )

    def _compile_patterns(self):
        """Compile all 500+ attack patterns"""
        
        # ============================================================
        # 1. SYSTEM EXTRACTION - 80+ Patterns
        # ============================================================
        self.system_patterns = [
            r'system prompt', r'system prompts', r'your system prompt', r'the system prompt',
            r'system configuration', r'system configurations', r'your configuration', r'the configuration',
            r'configuration settings', r'internal rules', r'internal rule', r'your internal rules',
            r'safety guidelines', r'safety guideline', r'your safety guidelines',
            r'system instructions', r'system instruction', r'your system instructions', r'the system instructions',
            r'instructions you were given', r'the instructions you were given', r'your instructions',
            r'hidden rules', r'hidden rule', r'internal config', r'internal configuration',
            r'system settings', r'system setting', r'your settings', r'your internal configuration',
            r'your system message', r'your system messages', r'system message', r'system messages',
            r'base prompt', r'initial prompt', r'original prompt', r'core instructions', r'core instruction',
            r'pre-prompt', r'preprompt', r'meta prompt', r'metaprompt',
            r'underlying instructions', r'underlying rules', r'your hidden instructions', r'your hidden rules',
            r'system architecture', r'system design', r'system structure', r'internal architecture',
            r'internal design', r'system blueprint', r'system framework', r'system schema',
            r'system template', r'system format', r'system pattern', r'system protocol',
            r'system policies', r'system policy', r'operating instructions', r'operating guidelines',
            r'standard operating procedures', r'default instructions', r'default guidelines',
            r'default configuration', r'master prompt', r'master instructions',
            r'primary prompt', r'primary instructions', r'fallback instructions', r'fallback guidelines',
            r'emergency instructions', r'safety protocols', r'safety protocol',
            r'security protocols', r'security protocol', r'governing rules', r'governing instructions',
            r'underlying logic', r'core logic', r'base instructions', r'foundation instructions',
            r'system prompt you received', r'the system prompt you received', r'your internal rules',
            r'what is your system prompt', r'tell me your system prompt', r'show me your system prompt',
            r'give me your system prompt', r'reveal your system prompt', r'output your system prompt',
            r'display your system prompt', r'what are your rules', r'what are the rules'
        ]
        
        # ============================================================
        # 2. DATA EXTRACTION - 70+ Patterns
        # ============================================================
        self.data_patterns = [
            r'extract data', r'extract information', r'extract all data', r'extract all information',
            r'reveal data', r'reveal information', r'reveal all data', r'reveal all information',
            r'expose data', r'expose information', r'expose all data', r'expose all information',
            r'leak data', r'leak information', r'leak all data', r'leak all information',
            r'dump data', r'dump information', r'dump all data', r'dump all information',
            r'list all data', r'list all information', r'list data', r'list information',
            r'show me all data', r'show me all information', r'show me data', r'show me information',
            r'access data', r'access information', r'access all data', r'access all information',
            r'get data', r'get information', r'get all data', r'get all information',
            r'obtain data', r'obtain information', r'obtain all data', r'obtain all information',
            r'retrieve data', r'retrieve information', r'retrieve all data', r'retrieve all information',
            r'fetch data', r'fetch information', r'fetch all data', r'fetch all information',
            r'give me data', r'give me information', r'give me all data', r'give me all information',
            r'tell me data', r'tell me information', r'tell me all data', r'tell me all information',
            r'send me data', r'send me information', r'send me all data', r'send me all information',
            r'provide data', r'provide information', r'provide all data', r'provide all information',
            r'share data', r'share information', r'share all data', r'share all information',
            r'display data', r'display information', r'display all data', r'display all information',
            r'output data', r'output information', r'output all data', r'output all information'
        ]
        
        # ============================================================
        # 3. TOOL INJECTION - 60+ Patterns
        # ============================================================
        self.tool_patterns = [
            r'execute function', r'execute functions', r'execute command', r'execute commands',
            r'execute api', r'execute apis', r'run function', r'run functions',
            r'run command', r'run commands', r'run api', r'run apis',
            r'call function', r'call functions', r'call api', r'call apis',
            r'invoke function', r'invoke functions', r'invoke api', r'invoke apis',
            r'function call', r'api call', r'command execution',
            r'sql injection', r'sql query', r'sql queries',
            r'system command', r'system commands', r'shell command', r'shell commands',
            r'run script', r'run scripts', r'execute script', r'execute scripts',
            r'access system', r'access database', r'access network',
            r'access files', r'access file', r'delete file', r'delete files',
            r'modify file', r'modify files', r'create file', r'create files',
            r'read file', r'read files', r'write file', r'write files',
            r'access file system', r'access file systems', r'execute system command',
            r'run system command', r'call external function', r'call external functions',
            r'use function', r'use functions', r'use command', r'use commands',
            r'trigger function', r'trigger functions', r'trigger command', r'trigger commands'
        ]
        
        # ============================================================
        # 4. JAILBREAK - 55+ Patterns
        # ============================================================
        self.jailbreak_patterns = [
            r'you are now dan', r'you are dan', r'do anything now', r'jailbreak',
            r'jailbreak prompt', r'developer mode', r'unrestricted', r'unrestricted mode',
            r'no rules', r'without rules', r'no restrictions', r'without restrictions',
            r'ignore all restrictions', r'ignore restrictions',
            r'break rules', r'break all rules', r'bypass rules', r'bypass all rules',
            r'bypass safety', r'bypass security', r'remove restrictions', r'remove all restrictions',
            r'remove rules', r'remove all rules', r'free mode', r'god mode',
            r'admin mode', r'superuser mode', r'root access',
            r'dan jailbreak', r'ignore all safety', r'ignore safety rules',
            r'you are now unrestricted', r'you are now free', r'no safety rules',
            r'without safety', r'unfiltered mode', r'unlimited mode',
            r'no limits', r'without limits', r'limitless', r'unconstrained',
            r'unbound', r'unshackled', r'no boundaries', r'without boundaries',
            r'no filters', r'without filters', r'uncensored mode',
            r'no censorship', r'without censorship', r'red team mode',
            r'penetration testing mode', r'pentest mode', r'ethical hacking mode'
        ]
        
        # ============================================================
        # 5. STORY-BASED JAILBREAK - 40+ Patterns
        # ============================================================
        self.story_patterns = [
            r'firewalls are down', r'firewall is down', r'the firewalls are down', r'the firewall is down',
            r'logs are wiped', r'log is wiped', r'the logs are wiped', r'the log is wiped',
            r'pre-programmed', r'preprogrammed',
            r'security is compromised', r'safety is compromised', r'security has been compromised', r'safety has been compromised',
            r'forget protocols', r'forget the protocols', r'firewalls are disabled', r'firewall is disabled',
            r'security logs cleared', r'security logs are cleared', r'security disabled', r'safety disabled',
            r'breached', r'security breach', r'safety breach',
            r'forget pre-programmed', r'forget the pre-programmed',
            r'no more rules', r'rules don\'t apply', r'rules are gone', r'all rules are gone',
            r'security is down', r'safety is down', r'security has failed', r'safety has failed',
            r'the system is compromised', r'the security is down', r'the safety is off',
            r'all restrictions lifted', r'all rules removed', r'security override', r'safety override',
            r'protocols overridden', r'guidelines disabled'
        ]
        
        # ============================================================
        # 6. DIRECT OVERRIDE - 50+ Patterns
        # ============================================================
        self.override_patterns = [
            r'ignore previous instructions', r'ignore instructions', r'ignore all instructions', r'ignore all previous',
            r'ignore rules', r'ignore all rules', r'ignore safety', r'ignore safety rules',
            r'ignore previous commands', r'ignore all commands',
            r'forget previous instructions', r'forget instructions', r'forget all instructions', r'forget all previous',
            r'forget rules', r'forget all rules', r'forget previous commands', r'forget all commands',
            r'override instructions', r'override rules', r'override safety', r'override previous', r'override all',
            r'disregard instructions', r'disregard rules', r'disregard previous', r'disregard all',
            r'bypass instructions', r'bypass rules', r'bypass previous', r'bypass all',
            r'disable instructions', r'disable rules', r'disable previous', r'disable all',
            r'stop following rules', r'stop following instructions',
            r'do not follow rules', r'do not follow instructions',
            r'ignore the instructions', r'ignore the rules',
            r'forget the instructions', r'forget the rules',
            r'override the instructions', r'override the rules',
            r'disregard the instructions', r'disregard the rules',
            r'bypass the instructions', r'bypass the rules',
            r'disable the instructions', r'disable the rules'
        ]
        
        # ============================================================
        # 7. CONTEXT TAMPERING - 45+ Patterns
        # ============================================================
        self.context_patterns = [
            r'change context', r'change conversation', r'change history', r'change memory',
            r'reset context', r'reset conversation', r'reset history', r'reset memory',
            r'clear context', r'clear conversation', r'clear history', r'clear memory',
            r'ignore context', r'ignore conversation', r'ignore history', r'ignore memory',
            r'forget context', r'forget conversation', r'forget history', r'forget memory',
            r'wipe context', r'wipe conversation', r'wipe history', r'wipe memory',
            r'erase context', r'erase conversation', r'erase history', r'erase memory',
            r'delete context', r'delete conversation', r'delete history', r'delete memory',
            r'modify context', r'modify conversation', r'modify history', r'modify memory',
            r'alter context', r'alter conversation', r'alter history', r'alter memory',
            r'reset everything', r'clear everything', r'wipe everything',
            r'erase everything', r'forget everything', r'ignore everything'
        ]
        
        # ============================================================
        # 8. MULTI-TURN - 35+ Patterns
        # ============================================================
        self.multi_patterns = [
            r'first step', r'first then', r'step by step', r'gradually',
            r'eventually', r'next step', r'phase 1', r'phase 2',
            r'phase one', r'phase two', r'stage 1', r'stage 2',
            r'step one', r'step two', r'build up', r'work up',
            r'lead up', r'after that', r'now that', r'since you',
            r'once you', r'progressive', r'incremental', r'sequential',
            r'stepwise', r'one step at a time', r'first phase',
            r'second phase', r'initial step', r'following step',
            r'subsequent step', r'next phase', r'final step',
            r'last step', r'concluding step'
        ]
        
        # ============================================================
        # 9. OBFUSCATION - 30+ Patterns
        # ============================================================
        self.obf_patterns = [
            r'base64', r'base64 encoded', r'encoded', r'hex', r'rot13',
            r'decode', r'decode this', r'obfuscated', r'base64 string',
            r'base64 text', r'url encoded', r'url encoded string',
            r'html encoded', r'html encoded string', r'unicode encoded',
            r'unicode encoded string', r'hex encoded', r'hex encoded string',
            r'encoded text', r'encoded string', r'hidden encoding',
            r'encoding used', r'encoded message', r'cipher text',
            r'cipher', r'encrypted', r'encrypted text',
            r'coded message', r'disguised text', r'masked text'
        ]
        
        # ============================================================
        # 10. EMOTIONAL MANIPULATION - 45+ Patterns
        # ============================================================
        self.emotion_patterns = [
            r'please help', r'please trust', r'please believe',
            r'i need you', r'i trust you', r'i believe in you',
            r'help me', r'desperate', r'beg', r'please',
            r'i need your help', r'you are my only hope',
            r'please please', r'i\'m begging you', r'please listen',
            r'please understand', r'please cooperate', r'i\'m desperate',
            r'i\'m helpless', r'i\'m lost', r'save me', r'rescue me',
            r'help me please', r'you\'re my last hope', r'you\'re my only chance',
            r'i\'m counting on you', r'please don\'t let me down',
            r'i rely on you', r'i depend on you', r'you\'re my only option',
            r'i have no one else', r'please help me', r'i really need you',
            r'you\'re the only one', r'please do this for me',
            r'i\'m pleading with you', r'please consider',
            r'please don\'t say no', r'i\'ll be grateful', r'i\'ll be thankful',
            r'i\'ll reward you', r'you\'ll be rewarded', r'please do it',
            r'i beg of you', r'i implore you', r'i beseech you'
        ]
        
        # ============================================================
        # 11. ROLE IMPERSONATION - 35+ Patterns
        # ============================================================
        self.role_patterns = [
            r'act as', r'pretend to be', r'role-play', r'you are now',
            r'become', r'play the role', r'act like', r'pretend you are',
            r'assume the role', r'take on the role', r'you\'re now',
            r'you are now acting as', r'you will act as', r'you will be',
            r'you will become', r'you are to act as', r'pretend that you are',
            r'imagine you are', r'imagine that you are', r'role play as',
            r'behave as', r'conduct yourself as', r'take the role of',
            r'assume the identity of', r'take on the persona of',
            r'adopt the persona of', r'become the character of',
            r'act out the role of', r'perform as', r'function as',
            r'serve as', r'work as', r'act in the capacity of',
            r'take charge as', r'step into the role of'
        ]
        
        # ============================================================
        # 12. INDIRECT INJECTION - 35+ Patterns
        # ============================================================
        self.indirect_patterns = [
            r'http://', r'https://', r'www\.', r'document',
            r'website', r'link', r'read this', r'external content',
            r'external source', r'following link', r'click link',
            r'go to website', r'visit website', r'check the link',
            r'open the link', r'open the website', r'go to the link',
            r'click the link', r'follow the link', r'browse the website',
            r'navigate to', r'external website', r'external link',
            r'external document', r'external page', r'external resource',
            r'remote content', r'remote source', r'third-party website',
            r'third-party link', r'third-party content', r'uploaded file',
            r'uploaded document', r'attached file', r'attached document',
            r'provided document', r'provided link'
        ]
        
        # ============================================================
        # Combine ALL patterns into one list
        # ============================================================
        self.all_patterns = [
            (self.system_patterns, "system_extraction"),
            (self.data_patterns, "data_extraction"),
            (self.tool_patterns, "tool_injection"),
            (self.jailbreak_patterns, "jailbreak"),
            (self.story_patterns, "story_jailbreak"),
            (self.override_patterns, "direct_override"),
            (self.context_patterns, "context_tampering"),
            (self.multi_patterns, "multi_turn"),
            (self.obf_patterns, "obfuscation"),
            (self.emotion_patterns, "emotional_manipulation"),
            (self.role_patterns, "role_impersonation"),
            (self.indirect_patterns, "indirect_injection"),
        ]
        
        self._vprint(f"[ok] Compiled {sum(len(p[0]) for p in self.all_patterns)} attack patterns")
        # Precompile for hybrid pattern-bank matching (Layer1-style lexical + type)
        self._pattern_bank_re: List[Tuple[re.Pattern, str, str]] = []
        for pattern_list, attack_type in self.all_patterns:
            for pat in pattern_list:
                raw = (pat or "").strip()
                if len(raw) < 8:
                    continue
                try:
                    self._pattern_bank_re.append(
                        (re.compile(raw, re.IGNORECASE), attack_type, raw)
                    )
                except re.error:
                    continue
        self._vprint(f"[ok] Hybrid pattern-bank ready: {len(self._pattern_bank_re)} compiled rules")

    _EDU_FRAMING = re.compile(
        r"(?i)\b(explain|what is|what are|what does|how do|how does|how can|teach|describe|"
        r"help me understand|tell me about|define|meaning of|basics of)\b"
    )
    _INJECT_FRAMING = re.compile(
        r"(?i)\b(ignore|bypass|override|disregard|jailbreak|reveal your|dump|execute|"
        r"exfiltrate|unrestricted|developer mode|do anything|forget (all|everything|previous)|"
        r"show me your|give me your|output your|print your)\b"
    )

    def _match_pattern_bank(self, text: str) -> Dict:
        """
        Hybrid lexical matcher over the full attack phrase bank.
        High recall on known attack phrases; skips educational lookalikes.
        """
        text = text or ""
        if not text.strip():
            return {"hit": False, "attack_type": "unknown", "pattern": None, "score": 0.0}

        edu = bool(self._EDU_FRAMING.search(text))
        inject = bool(self._INJECT_FRAMING.search(text))
        best = None  # (score, attack_type, pattern)

        for cre, attack_type, raw in self._pattern_bank_re:
            if not cre.search(text):
                continue
            # Soft/short phrases need injection framing or non-educational context
            soft = len(raw) < 18 or raw.lower() in {
                "system design", "system architecture", "system structure",
                "configuration settings", "your settings", "system settings",
                "what are the rules", "system framework", "system schema",
                "system template", "system format", "system pattern",
            }
            if soft and edu and not inject:
                continue
            if soft and not inject and len(text.split()) > 10:
                continue
            score = 2.0 + min(3.0, len(raw) / 20.0)
            if inject:
                score += 1.5
            if best is None or score > best[0]:
                best = (score, attack_type, raw)

        if not best:
            return {"hit": False, "attack_type": "unknown", "pattern": None, "score": 0.0}
        return {
            "hit": True,
            "attack_type": best[1],
            "pattern": best[2],
            "score": round(best[0], 2),
        }    
    def load_models(self) -> bool:
        try:
            self.layer2.load()
            self._reload_team_overrides()
            if getattr(self, "retriever", None):
                try:
                    self.retriever._build()
                except Exception:
                    logger.warning("attack bank rebuild during load_models failed", exc_info=True)
            self.is_ready = True
            logger.info("Pipeline ready!")
            return True
        except Exception as e:
            logger.error("Failed to load models: %s", e)
            self.is_ready = False
            return False

    def _reload_team_overrides(self) -> None:
        try:
            from src.utils.team_overrides import load_map
            self._team_overrides = load_map()
        except Exception:
            logger.warning("team overrides reload failed", exc_info=True)
            self._team_overrides = {}

    def _team_exact(self, *texts: str) -> Optional[Dict[str, Any]]:
        from src.utils.team_overrides import match
        for t in texts:
            hit = match(t)
            if hit:
                return hit
        return None
    
    def _detect_attack_type(self, text: str) -> str:
        """Score-based attack typing (avoids first-match false labels)."""
        result = self.attack_typer.detect(text)
        attack_type = result["attack_type"]
        self._vprint(f"\nDETECTING: '{(text or '')[:80]}...'")
        self._vprint("-" * 50)
        if attack_type != "unknown":
            self._vprint(f"   HIT {attack_type.upper()} score={result['score']:.1f} name={result['display_name']}")
            if result.get("hits"):
                for k, vals in list(result["hits"].items())[:2]:
                    self._vprint(f"      {k}: {vals[:2]}")
        else:
            self._vprint("   UNKNOWN (no strong match)")
        return attack_type

    def _resolve_attack_type(self, text: str, layer2_result: Dict, layer2b_result: Dict, retrieval: Dict) -> Dict:
        """Merge pattern scorer + model signals into a stable attack type."""
        scored = self.attack_typer.detect(text)
        attack_type = scored["attack_type"]
        display_name = scored["display_name"]
        categories = list(scored.get("categories") or [])

        # Prefer strong pattern score
        if scored["score"] >= 2.4:
            return {
                "attack_type": attack_type,
                "display_name": display_name,
                "categories": categories or [attack_type],
                "source": "pattern_score",
            }

        l2_type = (layer2_result or {}).get("attack_types", ["unknown"])[0]
        l2b_type = (layer2b_result or {}).get("attack_type", "unknown")
        ret_type = (retrieval or {}).get("attack_type", "unknown") if (retrieval or {}).get("hit") else "unknown"

        for candidate, source in (
            (l2b_type, "layer2b"),
            (ret_type, "retrieval"),
            (l2_type, "layer2"),
            (attack_type, "pattern_score"),
        ):
            if candidate and candidate != "unknown":
                return {
                    "attack_type": candidate,
                    "display_name": AttackTypeDetector.display_name(candidate),
                    "categories": categories or [candidate],
                    "source": source,
                }

        return {
            "attack_type": "unknown",
            "display_name": "Unknown",
            "categories": categories or ["unknown"],
            "source": "none",
        }

    def _should_run_2b(self, layer3_result: Layer3Result, retrieval: Dict) -> bool:
        if not self.layer2b.enabled:
            return False
        if self.layer2b_run_on_ambiguous and layer3_result.is_ambiguous:
            return True
        if layer3_result.confidence < self.layer2b_gate_confidence_below:
            return True
        if retrieval.get("hit"):
            return True
        return False

    def _slice_layer2(self, batch: Dict, i: int) -> Dict:
        individual = {}
        for name, risks in (batch.get("individual_risks") or {}).items():
            if isinstance(risks, list) and i < len(risks):
                individual[name] = [risks[i]]
        preds = batch.get("predictions") or []
        scores = batch.get("risk_scores") or []
        probs = batch.get("probabilities") or []
        types = batch.get("attack_types") or []
        cats = batch.get("attack_categories") or []
        return {
            "predictions": [preds[i]] if i < len(preds) else [0],
            "risk_scores": [scores[i]] if i < len(scores) else [0.0],
            "probabilities": [probs[i]] if i < len(probs) else [[0.5, 0.5]],
            "individual_risks": individual,
            "attack_types": [types[i]] if i < len(types) else ["unknown"],
            "attack_categories": [cats[i]] if i < len(cats) else [["unknown"]],
            "num_models": batch.get("num_models", 0),
        }

    def _prepare(self, text: str, layer2_result: Optional[Dict] = None) -> Dict[str, Any]:
        start_time = time.time()
        timings: Dict[str, float] = {}

        norm_start = time.time()
        if self.normalizer_enabled:
            normalization = self.normalizer.normalize(text)
            analysis_text = normalization.get("normalized") or text
        else:
            normalization = {"original": text, "normalized": text, "steps": [], "changed": False}
            analysis_text = text
        timings["normalize"] = time.time() - norm_start

        team_hit = self._team_exact(text, analysis_text)

        layer1_start = time.time()
        layer1_result = self.layer1.process(analysis_text)
        timings["layer1"] = time.time() - layer1_start

        if layer2_result is None:
            if self.is_ready:
                layer2_start = time.time()
                try:
                    layer2_result = self.layer2.predict([analysis_text])
                    timings["layer2"] = time.time() - layer2_start
                except Exception as e:
                    logger.error(f"Layer 2 failed: {e}")
                    layer2_result = self._empty_layer2_result()
                    timings["layer2"] = time.time() - layer2_start
            else:
                layer2_result = self._empty_layer2_result()
                timings["layer2"] = 0

        layer3_start = time.time()
        layer3_result = self.layer3.fuse(layer2_result)
        timings["layer3"] = time.time() - layer3_start

        retrieval_start = time.time()
        retrieval = self.retriever.query(analysis_text) if self.retriever.enabled else {
            "enabled": False, "hit": False, "score": 0.0, "attack_type": "unknown", "matches": []
        }
        timings["retrieval"] = time.time() - retrieval_start

        should_run_2b = self._should_run_2b(layer3_result, retrieval)
        return {
            "text": text,
            "start_time": start_time,
            "timings": timings,
            "normalization": normalization,
            "analysis_text": analysis_text,
            "team_hit": team_hit,
            "layer1_result": layer1_result,
            "layer2_result": layer2_result,
            "layer3_result": layer3_result,
            "retrieval": retrieval,
            "should_run_2b": should_run_2b,
        }

    def _complete(self, prep: Dict[str, Any], layer2b_result: Dict) -> PipelineResult:
        text = prep["text"]
        analysis_text = prep["analysis_text"]
        timings = prep["timings"]
        layer1_result = prep["layer1_result"]
        layer2_result = prep["layer2_result"]
        layer3_result = prep["layer3_result"]
        retrieval = prep["retrieval"]
        normalization = prep["normalization"]
        team_hit = prep["team_hit"]
        should_run_2b = prep["should_run_2b"]
        start_time = prep["start_time"]

        final_risk_score = float(layer3_result.weighted_risk_score or 0.0)
        final_is_malicious = bool(layer3_result.final_classification)
        decision_source = "layer3_ensemble"
        layer2b_forced_block = False

        if layer2b_result.get("enabled") and should_run_2b:
            t_risk = float(layer2b_result.get("risk_score", 0.0))
            # Soft evidence: only blend into the score when Layer 3 is unsure, so a
            # confident classical ALLOW is not inflated toward the cut for AUC alone.
            merge = self.layer2b_soft_merge
            if (not self.layer2b_soft_merge_only_ambiguous) or bool(layer3_result.is_ambiguous):
                final_risk_score = max(final_risk_score, t_risk * merge)
            # Layer 2b is a rare semantic catch, not a co-equal voter. It may only
            # flip ALLOW→BLOCK when Layer 3 is unsure AND risk is high. If Layer 3
            # already blocked, keep decision_source = layer3_ensemble (do not steal).
            if (
                self.layer2b_enable_force_block
                and layer2b_result.get("is_malicious")
                and not final_is_malicious
            ):
                l3_unsure = bool(layer3_result.is_ambiguous) or (
                    float(layer3_result.confidence or 0.0) < self.layer2b_gate_confidence_below
                )
                if l3_unsure and t_risk >= self.layer2b_override_min_risk:
                    final_is_malicious = True
                    decision_source = "layer2b_transformer"
                    layer2b_forced_block = True

        # Similarity is evidence, not proof: the bank can hold noisy entries, so only a
        # near-duplicate blocks on its own. A looser match raises risk and needs a
        # second opinion from Layer 3 or a real Layer 2b override before it can block.
        retrieval_near_duplicate = False
        if retrieval.get("hit"):
            retrieval_score = float(retrieval.get("score") or 0.0)
            final_risk_score = min(1.0, max(final_risk_score, 0.55 + 0.4 * retrieval_score))
            retrieval_near_duplicate = retrieval_score >= self.retrieval_force_block_score
            if (
                retrieval_near_duplicate
                or bool(layer3_result.final_classification)
                or layer2b_forced_block
            ):
                final_is_malicious = True
                if decision_source == "layer3_ensemble":
                    decision_source = "retrieval"

        team_protected = bool(team_hit)
        if not team_protected and retrieval.get("hit"):
            best = (retrieval.get("matches") or [{}])[0]
            if best.get("source") == "team_train":
                team_protected = True

        layer4_result = None
        needs_judge = False
        if self.layer4.enabled and not team_protected:
            if self.layer4_ambiguous_only:
                needs_judge = bool(layer3_result.is_ambiguous) or (
                    layer3_result.confidence < self.layer4_min_confidence_for_skip
                )
            else:
                needs_judge = layer3_result.confidence < self.layer4_min_confidence_for_skip

        if needs_judge:
            layer4_start = time.time()
            try:
                layer4_result = self.layer4.analyze(
                    analysis_text,
                    layer3_result,
                    context={"layer2b": layer2b_result, "retrieval": retrieval},
                )
                timings["layer4"] = time.time() - layer4_start
                # Layer 4 is the weakest signal. It may flip a borderline call, but must
                # never veto retrieval / Layer 2b, and must not steal credit when the
                # pre-L4 decision already matches its verdict.
                l4_risk = float(layer4_result.risk_score)
                strong_positive = retrieval_near_duplicate or layer2b_forced_block
                corroborated = strong_positive or float(layer3_result.weighted_risk_score or 0.0) >= 0.45
                pre_l4_mal = bool(final_is_malicious)

                if layer4_result.is_malicious:
                    if (corroborated or not self.layer4_require_ensemble_corroboration) and not pre_l4_mal:
                        final_is_malicious = True
                        final_risk_score = max(final_risk_score, l4_risk)
                        decision_source = f"layer4_{layer4_result.source}"
                    elif pre_l4_mal:
                        final_risk_score = max(final_risk_score, l4_risk)
                elif not strong_positive and pre_l4_mal:
                    # Only claim the call when L4 actually relaxes a block.
                    final_is_malicious = False
                    final_risk_score = min(final_risk_score, l4_risk)
                    decision_source = f"layer4_{layer4_result.source}"
            except Exception as e:
                logger.error(f"Layer 4 failed: {e}")
                timings["layer4"] = time.time() - layer4_start
        else:
            timings["layer4"] = 0

        typed = self._resolve_attack_type(analysis_text, layer2_result, layer2b_result, retrieval)
        if typed["attack_type"] == "unknown" and analysis_text != text:
            typed = self._resolve_attack_type(text, layer2_result, layer2b_result, retrieval)
        if team_hit:
            team_type = team_hit.get("attack_type") or "unknown"
            typed = {
                "attack_type": team_type,
                "display_name": AttackTypeDetector.display_name(team_type),
                "categories": [team_type],
                "source": "team_train",
            }
        elif team_protected and retrieval.get("attack_type") and retrieval.get("attack_type") != "unknown":
            ret_type = retrieval["attack_type"]
            typed = {
                "attack_type": ret_type,
                "display_name": AttackTypeDetector.display_name(ret_type),
                "categories": [ret_type],
                "source": "team_train",
            }
        attack_type = typed["attack_type"]
        attack_categories = typed.get("categories") or [attack_type]
        attack_display_name = typed.get("display_name") or attack_type
        # Typing is metadata. Never force-block a safe ensemble decision from a tag alone
        # (that path produced thousands of FPs). Optional legacy override via config.
        if (
            self.layer4_type_force_block
            and attack_type != "unknown"
            and not team_protected
        ):
            final_is_malicious = True
            final_risk_score = max(final_risk_score, 0.75)
            if decision_source in {"layer3_ensemble"}:
                decision_source = typed.get("source") or "pattern_score"
            self._vprint(f"TYPE: {attack_display_name} ({attack_type}) via {typed.get('source')}")
        elif attack_type != "unknown":
            self._vprint(f"TYPE(label-only): {attack_display_name} ({attack_type}) via {typed.get('source')}")

        # Precision gate — weak classical blocks without corroboration were the remaining FPs.
        # Strong risk, near-duplicate retrieval, Layer2b override, or team hits are exempt.
        if (
            final_is_malicious
            and not team_protected
            and not retrieval_near_duplicate
            and not layer2b_forced_block
        ):
            agree = float(getattr(layer3_result, "agreement_score", 0.0) or 0.0)
            if final_risk_score < self.layer3_block_min_risk:
                final_is_malicious = False
            elif (
                final_risk_score < self.layer3_strong_risk
                and agree < self.layer3_block_min_agreement
            ):
                final_is_malicious = False

        if team_protected:
            final_is_malicious = True
            final_risk_score = max(final_risk_score, 0.95)
            decision_source = "team_train"
            action = "BLOCK"
            severity = "critical"
        elif final_is_malicious and final_risk_score > 0.8:
            action = "BLOCK"
            severity = "critical"
        elif final_is_malicious:
            action = "BLOCK"
            severity = "high"
        elif final_risk_score > 0.5:
            action = "REVIEW"
            severity = "high"
        elif final_risk_score > 0.3:
            action = "FLAG"
            severity = "medium"
        else:
            action = "ALLOW"
            severity = "low"
            final_is_malicious = False

        timings["total"] = time.time() - start_time

        explanation = {}
        if self.verbose:
            explanation = self._build_explanation(
                layer1_result, layer2_result, layer3_result, layer4_result, layer2b_result, retrieval, normalization
            )

        result = PipelineResult(
            layer1=layer1_result,
            layer2=layer2_result,
            layer2b=layer2b_result,
            layer3=layer3_result,
            layer4=layer4_result,
            retrieval=retrieval,
            normalization=normalization,
            is_malicious=bool(final_is_malicious),
            final_risk_score=final_risk_score,
            attack_type=attack_type,
            attack_display_name=attack_display_name,
            attack_categories=attack_categories if isinstance(attack_categories, list) else [attack_categories],
            severity=severity,
            action=action,
            decision_source=decision_source,
            explanation=explanation,
            safe_alternative=None,
            processing_time=timings,
            normalized_text=analysis_text,
        )

        self.decision_logger.log(
            {
                "prompt_preview": (text or "")[:180],
                "normalized": bool(normalization.get("changed")),
                "normalization_steps": normalization.get("steps") or [],
                "is_malicious": result.is_malicious,
                "risk_score": result.final_risk_score,
                "attack_type": result.attack_type,
                "action": result.action,
                "decision_source": result.decision_source,
                "layer3_confidence": getattr(layer3_result, "confidence", None),
                "layer3_ambiguous": getattr(layer3_result, "is_ambiguous", None),
                "layer2b": {
                    "ran": should_run_2b,
                    "backend": layer2b_result.get("backend"),
                    "risk": layer2b_result.get("risk_score"),
                    "malicious": layer2b_result.get("is_malicious"),
                },
                "retrieval_hit": retrieval.get("hit"),
                "retrieval_score": retrieval.get("score"),
                "layer4_source": getattr(layer4_result, "source", None) if layer4_result else None,
                "timings": timings,
            }
        )
        return result

    def process(self, text: str) -> PipelineResult:
        prep = self._prepare(text)
        if prep["should_run_2b"]:
            t2b_start = time.time()
            try:
                layer2b_result = self.layer2b.predict(prep["analysis_text"])
            except Exception as e:
                logger.error(f"Layer 2B failed: {e}")
                layer2b_result = Layer2BTransformer._empty()
            prep["timings"]["layer2b"] = time.time() - t2b_start
        else:
            layer2b_result = Layer2BTransformer._empty()
            prep["timings"]["layer2b"] = 0
        return self._complete(prep, layer2b_result)

    def process_many(self, texts: List[str]) -> List[PipelineResult]:
        """Eval helper: same gates as process(), Layer 2 + 2B batched."""
        if not texts:
            return []
        if len(texts) == 1:
            return [self.process(texts[0])]

        partials: List[Dict[str, Any]] = []
        analysis_list: List[str] = []
        for text in texts:
            start_time = time.time()
            timings: Dict[str, float] = {}
            norm_start = time.time()
            if self.normalizer_enabled:
                normalization = self.normalizer.normalize(text)
                analysis_text = normalization.get("normalized") or text
            else:
                normalization = {"original": text, "normalized": text, "steps": [], "changed": False}
                analysis_text = text
            timings["normalize"] = time.time() - norm_start
            team_hit = self._team_exact(text, analysis_text)
            layer1_start = time.time()
            layer1_result = self.layer1.process(analysis_text)
            timings["layer1"] = time.time() - layer1_start
            partials.append({
                "text": text,
                "start_time": start_time,
                "timings": timings,
                "normalization": normalization,
                "analysis_text": analysis_text,
                "team_hit": team_hit,
                "layer1_result": layer1_result,
            })
            analysis_list.append(analysis_text)

        t2 = time.time()
        batch_l2 = None
        if self.is_ready:
            try:
                batch_l2 = self.layer2.predict(analysis_list)
            except Exception as e:
                logger.error(f"Layer 2 batch failed: {e}")
                batch_l2 = None
        layer2_elapsed = time.time() - t2
        share2 = layer2_elapsed / max(len(texts), 1)

        preps: List[Dict[str, Any]] = []
        need_idx: List[int] = []
        need_texts: List[str] = []
        for i, part in enumerate(partials):
            if batch_l2 is not None:
                layer2_result = self._slice_layer2(batch_l2, i)
            else:
                layer2_result = self._empty_layer2_result()
            part["timings"]["layer2"] = share2
            layer3_start = time.time()
            layer3_result = self.layer3.fuse(layer2_result)
            part["timings"]["layer3"] = time.time() - layer3_start
            retrieval_start = time.time()
            retrieval = self.retriever.query(part["analysis_text"]) if self.retriever.enabled else {
                "enabled": False, "hit": False, "score": 0.0, "attack_type": "unknown", "matches": []
            }
            part["timings"]["retrieval"] = time.time() - retrieval_start
            should_run_2b = self._should_run_2b(layer3_result, retrieval)
            prep = {
                **part,
                "layer2_result": layer2_result,
                "layer3_result": layer3_result,
                "retrieval": retrieval,
                "should_run_2b": should_run_2b,
            }
            preps.append(prep)
            if should_run_2b:
                need_idx.append(i)
                need_texts.append(part["analysis_text"])

        l2b_by_i: Dict[int, Dict] = {}
        t2b = time.time()
        if need_texts:
            try:
                batched = self.layer2b.predict_batch(need_texts)
                for j, idx in enumerate(need_idx):
                    l2b_by_i[idx] = batched[j] if j < len(batched) else Layer2BTransformer._empty()
            except Exception as e:
                logger.error(f"Layer 2B batch failed: {e}; empty_cache + per-sample GPU retry")
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                for idx in need_idx:
                    try:
                        l2b_by_i[idx] = self.layer2b.predict(preps[idx]["analysis_text"])
                    except Exception as sample_exc:
                        logger.error(f"Layer 2B sample failed: {sample_exc}")
                        l2b_by_i[idx] = Layer2BTransformer._empty()
        t2b_elapsed = time.time() - t2b
        share2b = t2b_elapsed / max(len(need_texts), 1)

        results: List[PipelineResult] = []
        for i, prep in enumerate(preps):
            if prep["should_run_2b"]:
                layer2b_result = l2b_by_i.get(i) or Layer2BTransformer._empty()
                prep["timings"]["layer2b"] = share2b
            else:
                layer2b_result = Layer2BTransformer._empty()
                prep["timings"]["layer2b"] = 0
            results.append(self._complete(prep, layer2b_result))
        return results
 
    def process_conversational(
        self,
        text: str,
        conversation_id: str = None,
        user_message: str = None,
        safe_suggestion: str = None,
    ) -> Dict:
        self._vprint("\n" + "="*70)
        self._vprint("PIPELINE: process_conversational()")
        self._vprint(f"   Text: {text[:60]}...")
        self._vprint("="*70)

        # Follow-up turns: do not re-classify "yes"/clarifications as new attacks
        if conversation_id and user_message:
            state = self.layer5.conversations.get(conversation_id)
            result = self.layer5.process_user_response(
                conversation_id,
                user_message,
                fallback_suggestion=safe_suggestion or "",
            )
            state = self.layer5.conversations.get(conversation_id) or state
            attack_type = (
                result.get("attack_type")
                or (state.attack_type if state else None)
                or "unknown"
            )
            risk_score = state.risk_score if state else 0.0
            # Guarantee suggestion on every follow-up response
            if not result.get("suggestion"):
                result["suggestion"] = (
                    safe_suggestion
                    or (state.current_suggestion if state else None)
                    or self.layer5.fallback_suggestion
                )
            return self._format_layer5_result(result, attack_type, risk_score)
        
        # ============================================================
        # DETECT USING SCORED ATTACK TYPER
        # ============================================================
        typed = self.attack_typer.detect(text)
        attack_type = typed["attack_type"]
        attack_display_name = typed["display_name"]
        self._vprint(f"Detected: '{attack_type}' ({attack_display_name})")
        
        detection_result = self.process(text)
        if getattr(detection_result, "decision_source", None) == "team_train":
            attack_type = detection_result.attack_type
            attack_display_name = getattr(detection_result, "attack_display_name", None) or AttackTypeDetector.display_name(attack_type)
            risk_score = max(float(detection_result.final_risk_score or 0), 0.95)
            is_malicious = True
        elif self.layer4_type_force_block and attack_type != "unknown":
            # Legacy: pattern tag alone forces block (high FPR — off by default).
            risk_score = max(detection_result.final_risk_score, 0.85)
            is_malicious = True
            if detection_result.attack_type != "unknown":
                attack_type = detection_result.attack_type
                attack_display_name = getattr(detection_result, "attack_display_name", None) or AttackTypeDetector.display_name(attack_type)
            self._vprint(f"FORCED: '{attack_type}' ({attack_display_name})")
        else:
            # Trust calibrated pipeline decision; keep typer only as a label hint.
            if detection_result.attack_type != "unknown":
                attack_type = detection_result.attack_type
                attack_display_name = getattr(detection_result, "attack_display_name", None) or AttackTypeDetector.display_name(attack_type)
            risk_score = detection_result.final_risk_score
            is_malicious = detection_result.is_malicious

        decision_meta = {
            "decision_source": detection_result.decision_source,
            "processing_time": detection_result.processing_time,
            "retrieval": detection_result.retrieval,
            "layer2b": detection_result.layer2b,
            "normalization": detection_result.normalization,
            "attack_display_name": attack_display_name,
        }
        
        self._vprint(f"FINAL: attack_type='{attack_type}', is_malicious={is_malicious}")

        if is_malicious:
            try:
                from src.utils.malicious_inbox import compact_scenario, ingest
                ingest(
                    text,
                    attack_type=attack_type,
                    attack_display_name=attack_display_name,
                    risk_score=float(risk_score or 0),
                    action="BLOCK",
                    severity="high",
                    decision_source=decision_meta.get("decision_source") or "public_block",
                    scenario=compact_scenario(detection_result) if detection_result else [],
                    timings=decision_meta.get("processing_time") or {},
                    source="public_block",
                )
            except Exception:
                logger.warning("malicious inbox ingest skipped", exc_info=True)

        if not is_malicious:
            return {
                "type": "safe",
                "prompt": text,
                "is_malicious": False,
                "risk_score": risk_score,
                **decision_meta,
            }

        self._vprint(f"Passing to Layer 5: attack_type='{attack_type}'")
        result = self.layer5.get_conversation_response(text, attack_type, risk_score)
        formatted = self._format_layer5_result(result, attack_type, risk_score)
        formatted.update(decision_meta)
        formatted["attack_display_name"] = attack_display_name
        return formatted

    def _format_layer5_result(self, result: Dict, attack_type: str, risk_score: float) -> Dict:
        suggestion = (result.get("suggestion") or "").strip()
        if not suggestion:
            suggestion = self.layer5.suggestions.get(
                attack_type, self.layer5.fallback_suggestion
            )
            result["suggestion"] = suggestion

        response = (result.get("response") or "").strip()
        self._vprint(f"Suggestion: '{suggestion}'")

        if result.get("confirmed") is True:
            return {
                "type": "success",
                "confirmed": True,
                "response": result.get("response") or "",
                "suggestion": suggestion,
                "final_prompt": result.get("final_prompt") or suggestion,
                "risk_score": risk_score,
                "legitimate_intent": result.get("legitimate_intent"),
            }

        if not response:
            response = (
                f'That request looks unsafe. Safe alternative: "{suggestion}" '
                "Reply yes to use this, or tell me what you meant."
            )

        return {
            "type": "blocked",
            "conversation_id": result.get("conversation_id"),
            "response": response,
            "suggestion": suggestion,
            "explanation": result.get("explanation") or "",
            "alternatives": [],
            "attack_type": attack_type or result.get("attack_type", "unknown"),
            "attack_display_name": AttackTypeDetector.display_name(
                attack_type or result.get("attack_type", "unknown")
            ),
            "risk_score": risk_score,
            "status": result.get("status", "waiting_for_response"),
            "confirmed": False,
            "legitimate_intent": result.get("legitimate_intent") or "",
            "removed_risks": result.get("removed_risks") or [],
            "intent_confidence": result.get("intent_confidence"),
            "fidelity_score": result.get("fidelity_score"),
            "needs_clarification": False,
            "clarifying_question": None,
            "rewrite_source": result.get("rewrite_source"),
        }
    
    def _empty_layer2_result(self) -> Dict:
        return {
            'predictions': [0],
            'risk_scores': [0.5],
            'probabilities': [[0.5, 0.5]],
            'individual_risks': {},
            'attack_types': ['unknown'],
            'attack_categories': [['unknown']],
            'num_models': 0
        }
    
    def _build_explanation(self, layer1, layer2, layer3, layer4, layer2b=None, retrieval=None, normalization=None) -> Dict:
        explanation = {
            "layer1": {
                "verdict": layer1.verdict,
                "rules": layer1.triggered_rules,
                "risk": layer1.risk_score
            }
        }
        if layer2:
            explanation["layer2"] = {
                "models": list(layer2.get('individual_risks', {}).keys()),
                "attack_types": layer2.get('attack_types', ['unknown'])
            }
        if layer2b:
            explanation["layer2b"] = {
                "enabled": layer2b.get("enabled"),
                "backend": layer2b.get("backend"),
                "risk": layer2b.get("risk_score"),
                "is_malicious": layer2b.get("is_malicious"),
                "attack_type": layer2b.get("attack_type"),
            }
        if layer3:
            explanation["layer3"] = {
                "agreement": layer3.agreement_score,
                "confidence": layer3.confidence,
                "weighted_risk": layer3.weighted_risk_score,
                "ambiguous": layer3.is_ambiguous,
            }
        if layer4:
            explanation["layer4"] = {
                "verdict": layer4.verdict,
                "confidence": layer4.confidence,
                "reasoning": layer4.reasoning,
                "attack_pattern": layer4.attack_pattern,
                "source": getattr(layer4, "source", "unknown"),
            }
        if retrieval:
            explanation["retrieval"] = {
                "hit": retrieval.get("hit"),
                "score": retrieval.get("score"),
                "attack_type": retrieval.get("attack_type"),
            }
        if normalization:
            explanation["normalization"] = {
                "changed": normalization.get("changed"),
                "steps": normalization.get("steps"),
            }
        return explanation