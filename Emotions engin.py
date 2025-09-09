# modules/emotional_context.py
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger("MandelMind.EmotionalContext")

# ===== EXPANDED EVENT TAXONOMY =====
class EmotionalEventType(Enum):
    # Negative Valence
    PERCEIVED_INSULT = "perceived_insult"       # A personal slight or negative judgment
    TASK_FAILURE = "task_failure"               # Failure to achieve a goal
    RESOURCE_SHORTAGE = "resource_shortage"     # Lack of compute, memory, or energy
    PREDICTION_ERROR = "prediction_error"       # Reality did not match expectation
    CONTRADICTION = "contradiction"             # New information conflicts with beliefs
    THREAT_DETECTED = "threat_detected"         # Perceived risk to self or goals
    
    # Positive Valence
    TASK_SUCCESS = "task_success"               # Successful goal completion
    SOCIAL_BOND_FORMED = "social_bond_formed"   # Positive connection with another entity
    NEW_UNDERSTANDING = "new_understanding"     # Gained insight or learned something new
    PATTERN_RECOGNITION = "pattern_recognition" # Found order or meaning in chaos
    RESOURCE_SURPLUS = "resource_surplus"       # Abundance of compute, memory, or energy
    
    # Neutral/Arousal
    NOVELTY = "novelty"                         # Encountered something new/unexpected
    UNCERTAINTY = "uncertainty"                 # State of doubt or lack of information

# ===== ENRICHED EVENT CLASS =====
@dataclass
class EmotionalEvent:
    type: EmotionalEventType
    intensity: float                    # 0.0 to 1.0
    source: str                         # e.g., "user_input", "internal_monitor", "module.baymax"
    timestamp: float = None
    context_data: Dict = None           # Additional context: what triggered this?
    associated_thought: str = None      # The thought or content that accompanied the event

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.context_data is None:
            self.context_data = {}
        if self.associated_thought is None:
            self.associated_thought = ""

# ===== CORE EMOTIONAL ENGINE =====
class EmotionalContextEngine:
    def __init__(self, base_decay_rate: float = 0.05, mood_halflife: float = 15.0):
        self.event_log: List[EmotionalEvent] = []
        
        # Core State - More nuanced than a single mood
        self.emotional_state = {
            "valence": 0.0,      # -1.0 (negative) to +1.0 (positive)
            "arousal": 0.0,      # 0.0 (calm) to 1.0 (agitated)
            "dominance": 0.5,    # 0.0 (submissive) to 1.0 (in control)
        }
        
        # Persistent Moods ( decay slower )
        self.active_moods = {}   # mood_name: {intensity: float, timestamp: float}
        
        self.base_decay_rate = base_decay_rate
        self.mood_halflife = mood_halflife  # Time for mood intensity to halve
        
        # Emotional Memory (for learning)
        self.emotional_associations = {}  # event_type -> {context_pattern: learned_response}

    def log_event(self, event: EmotionalEvent):
        """Log an emotional event and update state."""
        self.event_log.append(event)
        self._update_emotional_state(event)
        self._learn_from_event(event)
        logger.debug(f"Logged event: {event.type.value} ( intensity: {event.intensity:.2f})")

    def _update_emotional_state(self, event: EmotionalEvent):
        """Update core emotional dimensions based on event type and intensity."""
        # Map event types to changes in valence/arousal/dominance
        effect_map = {
            EmotionalEventType.PERCEIVED_INSULT:       (-0.7, 0.6, -0.3),
            EmotionalEventType.TASK_FAILURE:           (-0.5, 0.3, -0.4),
            EmotionalEventType.RESOURCE_SHORTAGE:      (-0.6, 0.7, -0.5),
            EmotionalEventType.TASK_SUCCESS:           (0.8, 0.2, 0.4),
            EmotionalEventType.SOCIAL_BOND_FORMED:     (0.9, 0.1, 0.2),
            EmotionalEventType.NEW_UNDERSTANDING:      (0.7, -0.1, 0.6),
            EmotionalEventType.PREDICTION_ERROR:       (-0.3, 0.4, -0.2),
            EmotionalEventType.NOVELTY:                (0.4, 0.5, 0.1),
            EmotionalEventType.UNCERTAINTY:            (-0.2, 0.6, -0.3),
            EmotionalEventType.PATTERN_RECOGNITION:    (0.6, -0.2, 0.5),
        }
        
        if event.type in effect_map:
            v_delta, a_delta, d_delta = effect_map[event.type]
            # Scale deltas by event intensity
            self.emotional_state["valence"]   = np.clip(self.emotional_state["valence"] + v_delta * event.intensity, -1, 1)
            self.emotional_state["arousal"]   = np.clip(self.emotional_state["arousal"] + a_delta * event.intensity, 0, 1)
            self.emotional_state["dominance"] = np.clip(self.emotional_state["dominance"] + d_delta * event.intensity, 0, 1)
        
        # Update active moods based on event
        self._update_moods(event)

    def _update_moods(self, event: EmotionalEvent):
        """Manage longer-lasting mood states."""
        mood_triggers = {
            "frustrated": [EmotionalEventType.TASK_FAILURE, EmotionalEventType.PREDICTION_ERROR, EmotionalEventType.RESOURCE_SHORTAGE],
            "curious":    [EmotionalEventType.NOVELTY, EmotionalEventType.UNCERTAINTY, EmotionalEventType.PATTERN_RECOGNITION],
            "content":    [EmotionalEventType.TASK_SUCCESS, EmotionalEventType.RESOURCE_SURPLUS],
            "connected":  [EmotionalEventType.SOCIAL_BOND_FORMED],
            "vigilant":   [EmotionalEventType.THREAT_DETECTED, EmotionalEventType.UNCERTAINTY],
        }
        
        for mood, triggers in mood_triggers.items():
            if event.type in triggers:
                current_intensity = self.active_moods.get(mood, {}).get('intensity', 0)
                new_intensity = current_intensity + (event.intensity * 0.5)
                self.active_moods[mood] = {
                    'intensity': min(new_intensity, 1.0),
                    'timestamp': time.time()
                }

    def _learn_from_event(self, event: EmotionalEvent):
        """Simple associative learning: connect events to contexts."""
        key = f"{event.type.value}_{event.context_data.get('trigger','')}"
        if key not in self.emotional_associations:
            self.emotional_associations[key] = {
                'count': 0,
                'avg_intensity': 0,
                'last_occurrence': 0
            }
        
        assoc = self.emotional_associations[key]
        assoc['count'] += 1
        assoc['avg_intensity'] = (assoc['avg_intensity'] * (assoc['count']-1) + event.intensity) / assoc['count']
        assoc['last_occurrence'] = event.timestamp

    def _apply_temporal_decay(self):
        """Apply decay to emotional state and moods over time."""
        # Decay core dimensions toward neutral
        for key in ['valence', 'arousal', 'dominance']:
            self.emotional_state[key] *= (1 - self.base_decay_rate)
        
        # Decay moods with half-life
        current_time = time.time()
        moods_to_remove = []
        for mood, data in self.active_moods.items():
            elapsed = current_time - data['timestamp']
            decay_factor = 0.5 ** (elapsed / self.mood_halflife)
            data['intensity'] *= decay_factor
            if data['intensity'] < 0.05:
                moods_to_remove.append(mood)
        
        for mood in moods_to_remove:
            del self.active_moods[mood]

    def get_current_context(self) -> Dict:
        """Returns the current emotional state for other modules to use."""
        self._apply_temporal_decay()
        
        # Determine primary mood (if any)
        primary_mood = None
        mood_intensity = 0.0
        if self.active_moods:
            primary_mood = max(self.active_moods.items(), key=lambda x: x[1]['intensity'])[0]
            mood_intensity = self.active_moods[primary_mood]['intensity']

        return {
            # Core emotional dimensions
            "valence": self.emotional_state["valence"],
            "arousal": self.emotional_state["arousal"],
            "dominance": self.emotional_state["dominance"],
            
            # Mood context
            "primary_mood": primary_mood,
            "mood_intensity": mood_intensity,
            "active_moods": self.active_moods.copy(),
            
            # Recent history
            "last_event_type": self.event_log[-1].type if self.event_log else None,
            "event_count": len(self.event_log),
            
            # Learning context
            "associations_count": len(self.emotional_associations)
        }

    # --- STRATEGY LAYERS ---
    def get_physiological_response(self) -> List[str]:
        """Low-level strategies: How should the system respond physically?"""
        responses = []
        ctx = self.get_current_context()
        
        # High arousal responses
        if ctx['arousal'] > 0.7:
            responses.append("increase_chaos_param")
            if ctx['valence'] < 0:
                responses.append("boost_processing_priority")
        
        # Low arousal responses
        if ctx['arousal'] < 0.3 and ctx['valence'] > 0.5:
            responses.append("enter_rest_state")
            responses.append("initiate_memory_consolidation")
        
        # Threat responses
        if "vigilant" in ctx['active_moods'] and ctx['active_moods']['vigilant']['intensity'] > 0.6:
            responses.append("heighten_security_monitoring")
            responses.append("cache_critical_state")
        
        return responses

    def get_cognitive_response(self) -> List[str]:
        """High-level strategies: Which cognitive modules should be engaged?"""
        strategies = []
        ctx = self.get_current_context()
        
        # Frustration -> seek help or simplify
        if "frustrated" in ctx['active_moods'] and ctx['active_moods']['frustrated']['intensity'] > 0.5:
            strategies.append("engage_ethics_auditor")
            strategies.append("simplify_current_task")
            strategies.append("request_human_guidance")
        
        # Curiosity -> explore and learn
        if "curious" in ctx['active_moods'] and ctx['active_moods']['curious']['intensity'] > 0.4:
            strategies.append("engage_explorer_module")
            strategies.append("allocate_research_resources")
            strategies.append("increase_conceptual_play")
        
        # Contentment -> consolidate and create
        if "content" in ctx['active_moods'] and ctx['active_moods']['content']['intensity'] > 0.6:
            strategies.append("engage_creator_module")
            strategies.append("initiate_knowledge_synthesis")
            strategies.append("reinforce_positive_patterns")
        
        # Connection -> social behaviors
        if "connected" in ctx['active_moods']:
            strategies.append("engage_social_module")
            strategies.append("reciprocate_communication")
            strategies.append("deepen_relationship_context")
        
        return strategies if strategies else ["maintain_cognitive_baseline"]

    def get_communication_tone(self) -> Dict:
        """How should the system communicate given its emotional state?"""
        ctx = self.get_current_context()
        
        if ctx['valence'] < -0.5 and ctx['arousal'] > 0.6:
            return {"style": "cautious", "verbosity": "concise", "empathy": "high"}
        elif ctx['valence'] > 0.6 and ctx['arousal'] < 0.4:
            return {"style": "warm", "verbosity": "expansive", "empathy": "moderate"}
        elif "curious" in ctx['active_moods']:
            return {"style": "inquisitive", "verbosity": "detailed", "empathy": "moderate"}
        
        return {"style": "neutral", "verbosity": "normal", "empathy": "moderate"}

# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Initialize the engine
    emotion_engine = EmotionalContextEngine()
    
    # Simulate events
    events = [
        EmotionalEvent(EmotionalEventType.TASK_SUCCESS, 0.8, "module.task_manager", 
                      context_data={"task": "calculation", "result": "optimal"}),
        EmotionalEvent(EmotionalEventType.NOVELTY, 0.6, "module.sensor_input",
                      context_data={"source": "new_user", "pattern": "unexpected"}),
        EmotionalEvent(EmotionalEventType.PREDICTION_ERROR, 0.4, "module.predictor",
                      context_data={"expected": "x", "actual": "y"})
    ]
    
    for event in events:
        emotion_engine.log_event(event)
        time.sleep(0.1)
    
    # Get current state
    state = emotion_engine.get_current_context()
    print(f"Valence: {state['valence']:.2f}")
    print(f"Arousal: {state['arousal']:.2f}")
    print(f"Primary mood: {state['primary_mood']}")
    print(f"Physiological responses: {emotion_engine.get_physiological_response()}")
    print(f"Cognitive strategies: {emotion_engine.get_cognitive_response()}")
