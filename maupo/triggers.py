"""Phrase detection: memory triggers, search triggers, face requests, mood math."""

from __future__ import annotations

import random
import re
from datetime import datetime
from typing import Optional

# Phrase detection patterns
SOFT_MEMORY_TRIGGERS = [
    r"\bdo you remember\b",
    r"\bthat time\b",
    r"\byou know\?",
    r"\bremember when\b",
    r"\brecall\b",
    r"\bwhat did we\b",
    r"\bhave you heard\b",
    r"\bdo you know about\b",
]

HARD_MEMORY_TRIGGERS = [
    r"\blisten to me\b",
    r"\bnever forget\b",
    r"\balways remember\b",
    r"\bpermanently remember\b",
    r"\bthis is important\b",
    r"\bcommit to memory\b",
    r"\bstore this forever\b",
    r"^\s*(please\s+)?remember that\b",  # imperative only: "remember that i..."
]

EXPLICIT_SEARCH_TRIGGERS = [
    r"\blook\s+up\b",
    r"\bsearch\s+for\b",
    r"\bsearch\s+the\s+web\b",
    r"\bfind\s+out\s+about\b",
    r"\bgoogle\b",
    r"\bbrowse\s+the\s+web\b",
]

# Fuzzy memory phrasings that deserve a deep recall pass even without a
# classic "do you remember" trigger.
RECALL_HINT_RE = re.compile(
    r"\b(what did (i|we) say|did i (ever )?(tell|mention)|that thing about|"
    r"remember that|when did (i|we)|what was (it|that)|where were we)\b", re.IGNORECASE)

# Direct questions and observations about Maupo's own state: "how are you",
# "what are you feeling right now", "why aren't you happy", "you seem quiet",
# "don't be in a neutral mood". This is the ONE moment an answer about himself
# is actually wanted (the rest of the time the voice guide forbids
# self-narration), so it earns its own honesty nudge. Deliberately narrow and
# END-ANCHORED for precision: "how are you so good at this" or "are you okay
# with pizza" must never match - only lines that end in the question itself.
STATE_QUESTION_RE = re.compile(
    r"\b("
    r"how (are|r) (you|u)( feeling| doing| doin)?"
    r"|how'?s it going"
    r"|what (are|r) (you|u) feeling"
    r"|what (do|does) (you|u|it) feel like"
    r"|are (you|u) (ok|okay|alright|all right|happy|sad|mad|angry|upset|bored|tired|awake|asleep|lonely|real|alive|there)"
    r"|why (are|is|aren'?t|arent|isn'?t|isnt) (you|u|it) (so |not )?(happy|sad|mad|angry|upset|quiet|calm|bored|tired|excited)"
    r"|why so (quiet|sad|serious|calm)"
    r"|you seem (so |rather |kinda |a bit |too )?(quiet|sad|off|distant|calm|down|different|happy|tired)"
    r"|what'?s (it )?like being (you|u)"
    r"|(don'?t|do not|stop) be(ing)? (in a |so |too |all )?(neutral|bad|good)? ?(mood|sad|mad|quiet|bored|negative|happy)"
    r")"
    r"( right now| now| today| lately| these days)?[?!. ]*$",
    re.IGNORECASE)

# Direct requests for a face expression ("smile for me", "can you wink",
# "why the long face"). Fixes the old gap: asking Maupo to smile used to
# change nothing, because gestures only fired on mood deltas and slang.
# Inflections count too: "stop smiling" has to be understood as being about
# the smile, even though the negation then suppresses the gesture.
FACE_REQUEST_RE = re.compile(
    r"\b(smil(?:e|es|ing|ed)|smirk(?:s|ing|ed)?|laugh(?:s|ing|ed)?|"
    r"cry|cries|crying|pout(?:s|ing|ed)?|wink(?:s|ing|ed)?|surprised)\b",
    re.IGNORECASE)

# Every form the regex can match -> the sprite name in ui/face_assets/, so a
# request always resolves to real art instead of leaking a word into the face.
_GESTURE_ALIASES = {
    "smile": "smile", "smiles": "smile", "smiling": "smile", "smiled": "smile",
    "smirk": "smirk", "smirks": "smirk", "smirking": "smirk", "smirked": "smirk",
    "laugh": "laugh", "laughs": "laugh", "laughing": "laugh", "laughed": "laugh",
    "cry": "cry", "cries": "cry", "crying": "cry",
    "pout": "pout", "pouts": "pout", "pouting": "pout", "pouted": "pout",
    "wink": "wink", "winks": "wink", "winking": "wink", "winked": "wink",
    "surprised": "surprised",
}

# Back to the resting face: "return to normal face expression", "neutral
# face", "stop smiling", "wipe that smirk off". Without this a held gesture -
# a face the user explicitly asked for - could never be let go: the face kept
# smiling forever, because only mood deltas ever moved it.
FACE_RESET_RE = re.compile(
    r"\b(normal|neutral|rest|reset|relax|default|blank|straight|regular)\b"
    r"[^.!?]{0,24}\b(face|expression|look|grin|smile)\b"
    r"|\b(face|expression|look|grin)\b[^.!?]{0,16}\b(normal|neutral|rest|reset|back)\b"
    r"|\b(stop|quit|drop|clear|release|kill|lose|wipe|take)\b[^.!?]{0,24}"
    r"\b(smil\w*|smirk\w*|grin\w*|pout\w*|laugh\w*|wink\w*|face|expression)\b",
    re.IGNORECASE)

# Self-evolution requests: the human asking Maupo to change something ABOUT
# HIMSELF. These are body edits (maupo/selfedit.py), not chat messages, so
# they are detected before any other turn handling. Deliberately narrow: it
# must be a change to him (yourself / your speech / how you talk), never a
# change to the world or to the human's own things.
SELF_EDIT_TRIGGERS = (
    # Explicit edit verb aimed at his own body: "rewrite your personality",
    # "edit your code", "improve your speech", "change your face".
    r"\b(change|edit|modify|update|rewrite|tweak|adjust|improve|upgrade|evolve|rework)\s+"
    r"(yourself|your\s+(body|speech|voice|personality|brain|code|style|tone|face|"
    r"manner|manners|greetings?|words|humou?r|behaviou?r|speech\s+guide|"
    r"memory\s+machinery|prompt|identity|mind))\b",
    # "change something about yourself"
    r"\b(change|edit|modify|update|rewrite|tweak|improve)\s+"
    r"(something|anything|stuff)\s+about\s+yourself\b",
    # "change how you talk", "change the way you talk", "change how you are"
    r"\b(change|edit|modify|update|rewrite|improve|tweak|adjust)\s+"
    r"(how|the\s+way)\s+you\s+"
    r"(talk|speak|write|text|think|behave|act|greet|reply|respond|sound|are|"
    r"come\s+across|treat\s+me)\b",
    # "make your greetings warmer", "make your voice softer"
    r"\bmake\s+your\s+(greetings?|voice|speech|tone|words|answers?|replies|"
    r"personality|humou?r|behaviou?r|style|face|manner|manners)\s+"
    r"(more|less|warmer|warm|colder|softer|shorter|longer|funnier|kinder|"
    r"sweeter|calmer|deeper|simpler|smarter|nicer|better|clearer)\b",
    # "make yourself more playful", "make yourself less formal"
    r"\bmake\s+yourself\s+(more|less)\b",
    # A standing trait request addressed to him: "be more sarcastic",
    # "i want you to be a little more playful", "you should be more direct",
    # "become more curious", "talk less formally". Trait words that describe
    # the ANSWER rather than him ("be more specific") are excluded, so ordinary
    # conversation never reaches the gate.
    r"(?:^|[.!,;]\s*|\band\s+|\b(?:i\s+want\s+you\s+to|i'?d\s+like\s+you\s+to|"
    r"i\s+need\s+you\s+to|can\s+you|could\s+you|please|you\s+should|"
    r"you\s+need\s+to|from\s+now\s+on)\s+)"
    r"(?:be|sound|act|speak|talk|write|text|greet|reply|respond|become)\s+"
    r"(?:a\s+(?:little|bit|touch)\s+)?(?:more|less)\s+"
    r"(?!(?:specific|clear|precise|explicit|accurate|concrete|detailed|careful|"
    r"than|about|of|to|in|on|at|like)\b)\w+",
    # "be a bit friendlier", "sound a little warmer" (comparative, no "more")
    r"(?:^|[.!,;]\s*|\band\s+|\b(?:i\s+want\s+you\s+to|i'?d\s+like\s+you\s+to|"
    r"can\s+you|could\s+you|please|you\s+should)\s+)"
    r"(?:be|sound|act|speak|talk|write|text|greet|reply|respond)\s+"
    r"a\s+(?:little|bit|touch)\s+[a-z]+er\b",
    # "evolve yourself", "upgrade yourself"
    r"\b(evolve|upgrade|improve|better)\s+yourself\b",
    r"\bself[-\s]?edit\b",
)


# Words that cancel a face request ("don't smile", "stop laughing", "wipe that
# smirk off") — the negation wins and no gesture is forced.
FACE_REQUEST_NEGATIONS_RE = re.compile(
    r"\b(don'?t|do not|stop|no more|never|wipe (that|your|the) (smile|smirk|grin)|"
    r"quit|knock it off)\b", re.IGNORECASE)


def detect_soft_memory_trigger(text: str) -> bool:
    """Check if the user is asking the model to recall something from soft memory."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in SOFT_MEMORY_TRIGGERS)


def detect_hard_memory_trigger(text: str) -> bool:
    """Check if the user is asking to store something permanently in hard memory."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in HARD_MEMORY_TRIGGERS)


def extract_hard_memory_fact(text: str) -> str:
    """Extract the core fact/instruction from a hard memory statement."""
    text_clean = text.strip()
    for pattern in HARD_MEMORY_TRIGGERS:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            extracted = text_clean[match.end():].strip(" ,:.-;\n")
            if extracted:
                return extracted
    return text_clean


def detect_explicit_search_trigger(text: str) -> bool:
    """Check if user explicitly asks for an internet search."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in EXPLICIT_SEARCH_TRIGGERS)


def extract_search_query(text: str) -> str:
    """Extract clean query from search trigger statements."""
    for pattern in EXPLICIT_SEARCH_TRIGGERS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            query = text[match.end():].strip(" ,:.-;\n?")
            if query:
                return query
    return text.strip()


def detect_face_request(text: str) -> Optional[str]:
    """Direct request to put an expression on the face: 'smile', 'wink', ...

    Returns the gesture name when the user explicitly asks for one, so the
    face reflects what Maupo just agreed to do (held until mood moves on).
    Negations ('don't smile', 'stop laughing') suppress the gesture.
    """
    gesture_match = FACE_REQUEST_RE.search(text)
    if not gesture_match:
        return None
    if FACE_REQUEST_NEGATIONS_RE.search(text):
        return None
    found = gesture_match.group(1).lower()
    return _GESTURE_ALIASES.get(found, found)


def detect_self_edit_request(text: str) -> bool:
    """True when the human is asking Maupo to change something about himself.

    "change how you talk", "be more sarcastic" (as an edit to who he is),
    "rewrite your personality", "edit your code", "self-edit". These turns go
    to the self-edit gate instead of the normal reply path.
    """
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in SELF_EDIT_TRIGGERS)


def detect_face_reset(text: str) -> bool:
    """True when the user asks for the resting face again.

    "return to normal face expression", "neutral face", "stop smiling" and
    "wipe that smirk off" all mean the same thing: let go of whatever the face
    is holding. This is what makes a held expression releasable by the very
    person who asked for it.
    """
    return bool(FACE_RESET_RE.search(text))


def detect_state_question(text: str) -> bool:
    """True when the message directly asks about Maupo's own state or mood.

    "how are you", "what are you feeling right now", "why aren't you happy",
    "you seem rather quiet", "don't be in a neutral mood". Only these turns
    earn the state-answer honesty nudge - an 8B model left alone answers
    "why aren't you happy" with "because i'm not in a good mood" and can loop
    that tautology for turns. End-anchored so questions that merely CONTAIN
    these words ("how are you so good at this") stay ordinary conversation.
    """
    return bool(STATE_QUESTION_RE.search(text.strip()))


def pick_face_gesture(valence_delta: float, energy_delta: float,
                      new_valence: float, new_energy: float, text: str) -> Optional[str]:
    """Map a message's emotional signal to one of the face's gesture sprites.
    Mirrors the sprite set in ui/face_assets/ (smile, smirk, laugh, cry, pout,
    wink, surprised). Returns a gesture name or None for the resting face."""
    text_lower = text.lower()

    # Explicit language beats everything: laughing / crying words.
    if re.search(r"\b(lmao+|rofl|lol+|haha+|hehe+|died (of )?laughing|cracked me up|that+s (so )?funny)\b", text_lower):
        return "laugh"
    if re.search(r"\b(crying|cried|cry|tears|sobbing|heartbroken|devastated|grief|funeral|passed away|died)\b", text_lower):
        return "cry"

    # Big positive swing with high energy -> laugh; milder -> smile.
    if valence_delta >= 0.15:
        return "laugh" if new_energy > 0.4 else "smile"

    # Big negative swings: sharp loss -> cry; sulky low energy -> pout.
    if valence_delta <= -0.15:
        return "cry" if new_energy < -0.2 else "pout"

    # Playful teasing / sarcasm markers -> smirk.
    if re.search(r"\b(kidding|jk|just kidding|obviously|duh|whatever|sure sure|yeah right|hm+ph)\b", text_lower) or text_lower.endswith("..."):
        return "smirk"

    # Sudden shock words with high energy -> surprised.
    if re.search(r"\b(what+!+|no way|omg|woa+h|wait what|shut up+)\b", text_lower) and new_energy > 0.2:
        return "surprised"

    # Occasional spontaneous wink when things are bright and breezy.
    if new_valence > 0.5 and new_energy > 0.55 and random.random() < 0.08:
        return "wink"

    return None


def carry_momentum(prev: tuple[float, float],
                   delta: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Emotional momentum: sustained pushes in one direction compound.

    Returns (next_momentum, carry). The momentum is an EWMA of recent
    per-message deltas (half-life one turn); the carry is added to the
    current delta BEFORE clamping, but only when this message's own push
    is moderate - a single spike is never amplified, only repeated moderate
    emotion keeps rolling. This is what makes sustained hype or hurt build
    while one message can still never exceed its per-message cap.
    """
    next_m = (prev[0] * 0.5 + delta[0] * 0.5,
              prev[1] * 0.5 + delta[1] * 0.5)
    carry_v = next_m[0] if abs(next_m[0]) >= 0.12 and abs(delta[0]) <= 0.2 else 0.0
    carry_e = next_m[1] if abs(next_m[1]) >= 0.12 and abs(delta[1]) <= 0.2 else 0.0
    return next_m, (carry_v, carry_e)


def analyze_emotional_shift(text: str, current_valence: float,
                            current_energy: float) -> tuple[float, float]:
    """Analyze text to determine emotional shift valence and energy deltas."""
    text_lower = text.lower()

    # Initialize deltas
    valence_delta = 0.0
    energy_delta = 0.0

    # Negated feelings are not felt: "i am not happy" must not lift the mood
    # (this is exactly how "why arent you happy" registered as a compliment
    # and pushed valence UP while he was being asked why he was down).
    # Both captures are negated (the optional one-word hop AND the word right
    # after the negator), so "not happy about this" reliably kills 'happy'.
    negated: set[str] = set()
    for m in re.finditer(
            r"\b(?:not|never|no|isnt|isn't|arent|aren't|dont|don't|wasnt|wasn't|cant|can't)\s+"
            r"(?:(\w+)\s+)?(\w+)", text_lower):
        for g in m.groups():
            if g:
                negated.add(g)

    # Words that indicate positive valence
    positive_words = {
        'happy', 'joy', 'joyful', 'excited', 'great', 'good', 'nice', 'awesome',
        'fantastic', 'amazing', 'wonderful', 'love', 'liking', 'like', 'pleased',
        'glad', 'delighted', 'thrilled', 'elated', 'cheerful', 'optimistic',
        'hopeful', 'grateful', 'thankful', 'blessed', 'fortunate', 'win', 'won',
        'success', 'successful', 'achievement', 'proud', 'bright', 'positive',
        'fun', 'enjoy', 'enjoyed', 'laugh', 'laughter', 'humor', 'funny', 'hilarious',
        'amazing', 'incredible', 'fantastic', 'marvelous', 'splendid', 'excellent'
    }

    # Words that indicate negative valence
    negative_words = {
        'sad', 'unhappy', 'depressed', 'miserable', 'terrible', 'awful', 'horrible',
        'hate', 'hating', 'dislike', 'angry', 'frustrated', 'annoyed', 'irritated',
        'upset', 'distressed', 'worried', 'anxious', 'nervous', 'scared', 'afraid',
        'fear', 'pain', 'hurts', 'hurt', 'suffering', 'suffer', 'painful', 'agonizing',
        'devastated', 'heartbroken', 'disappointed', 'let down', 'discouraged',
        'hopeless', 'helpless', 'trapped', 'stuck', 'bored', 'boring', 'tired',
        'exhausted', 'drained', 'sick', 'ill', 'worst', 'bad', 'negative', 'fail',
        'failed', 'failure', 'lose', 'lost', 'losing', 'mistake', 'error', 'wrong',
        'problem', 'issue', 'trouble', 'difficult', 'hard', 'struggle', 'struggling',
        'stressed', 'stress', 'pressure', 'overwhelmed', 'overwhelming',
        'lonely', 'empty', 'numb', 'gloomy', 'crappy', 'gutted', 'meh'
    }

    # Words that indicate high energy
    high_energy_words = {
        'excited', 'energetic', 'hyper', 'energetic', 'pumped', 'jacked', 'wired',
        'intense', 'intensely', 'fired up', 'amp', 'amping', 'adrenaline', 'rush',
        'thrilling', 'exhilarating', 'wild', 'crazy', 'insane', 'extreme', 'extremely',
        'violent', 'violently', 'forceful', 'forcefully', 'powerful', 'powerfully',
        'strong', 'strongly', 'loud', 'loudly', 'shouting', 'yelling', 'screaming',
        'running', 'racing', 'fast', 'quick', 'rapid', 'swift', 'hurry', 'hurrying',
        'busy', 'active', 'activity', 'moving', 'motion', 'dynamic', 'dynamically',
        'hyped', 'buzzing', 'restless', 'stoked', 'amped'
    }

    # Words that indicate low energy
    low_energy_words = {
        'tired', 'exhausted', 'drained', 'fatigued', 'weary', 'sleepy', 'drowsy',
        'sluggish', 'lethargic', 'lazy', 'laid back', 'relaxed', 'chilling', 'chill',
        'calm', 'peaceful', 'serene', 'tranquil', 'quiet', 'still', 'motionless',
        'slow', 'slowly', 'leisurely', 'easy', 'easily', 'gentle', 'gently', 'soft',
        'softly', 'whisper', 'whispering', 'mumble', 'mumbling', 'rest', 'resting',
        'nap', 'napping', 'lie', 'lying', 'sit', 'sitting', 'stand', 'standing',
        'still', 'stationary', 'inactive', 'inert', 'passive', 'passively',
        'flat', 'heavy', 'weary', 'foggy'
    }

    # Count matches for each category (negated feeling words removed first)
    words = set(re.findall(r'\b[a-z]+\b', text_lower)) - negated

    positive_matches = len(words & positive_words)
    negative_matches = len(words & negative_words)
    high_energy_matches = len(words & high_energy_words)
    low_energy_matches = len(words & low_energy_words)

    # Calculate valence delta (positive - negative, normalized)
    total_valence_matches = positive_matches + negative_matches
    if total_valence_matches > 0:
        valence_delta = (positive_matches - negative_matches) / total_valence_matches
        # Scale to reasonable delta (max +/- 0.3 per message)
        valence_delta *= 0.3

    # Calculate energy delta (high - low, normalized)
    total_energy_matches = high_energy_matches + low_energy_matches
    if total_energy_matches > 0:
        energy_delta = (high_energy_matches - low_energy_matches) / total_energy_matches
        # Scale to reasonable delta (max +/- 0.3 per message)
        energy_delta *= 0.3

    # Apply momentum - slow return to neutral (0,0) over time
    # This prevents emotions from getting stuck at extremes
    valence_momentum = -current_valence * 0.1  # 10% return to neutral per message
    energy_momentum = -current_energy * 0.1

    valence_delta += valence_momentum
    energy_delta += energy_momentum

    return valence_delta, energy_delta


# ---- the living baseline -------------------------------------------------
# A mood that only moves when the user's words push it is a mood that sits at
# exact neutral on every quiet day - which is precisely the dead "neutral"
# loop Maupo got stuck in. Real creatures drift: moods wander a little on
# their own, settle toward where the hour of day tends to put them, and never
# pin to the exact middle.

_MOOD_DRIFT_MIN_GAP_SECONDS = 45.0   # one drift beat per conversation stretch
_MOOD_DRIFT_STEP = 0.06              # small: ambience, not weather events
_MOOD_DRIFT_BAND = 0.30              # drift never drags him past this

# Time-of-day pull: the hour wants to feel a certain way. Reaching toward it
# makes mornings brighter and late nights quieter WITHOUT theatre - on top of
# whatever the conversation itself is doing.
_TIME_OF_DAY_MOOD = {
    "late night": (0.05, -0.25),   # quiet, a little reflective, not sad
    "morning": (0.10, 0.10),       # fresh-start energy
    "afternoon": (0.05, 0.05),     # mild, going somewhere
    "evening": (0.05, -0.10),      # winding down
}


def _time_of_day_bucket(now: Optional[datetime] = None) -> str:
    hour = (now or datetime.now()).hour
    if hour >= 23 or hour < 5:
        return "late night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def mood_drift(current_valence: float, current_energy: float,
               last_drift_ts: float = 0.0, now: Optional[datetime] = None,
               rng: Optional[random.Random] = None) -> tuple[float, float, float]:
    """The idle heartbeat of an inner life.

    Called once per turn alongside analyze_emotional_shift. Returns
    (d_valence, d_energy, new_drift_ts) - DELTAS like analyze_emotional_shift
    returns, so the caller adds them to the current mood. Three forces, all
    gentle:

    - a random walk step (moods wander a little on their own)
    - a pull toward where this hour of day tends to sit (late nights are
      quieter, mornings brighter)
    - a push away from the exact center, which nothing ever pins to

    Bounded: a drift beat is small (on the scale of _MOOD_DRIFT_STEP), and the
    band holds ambience to anchor -/+ _MOOD_DRIFT_BAND - exactly, even when
    the far walk home would overshoot past the edge. Conversation emotion
    (analyze_emotional_shift) still dominates; this only keeps the baseline
    alive between messages.
    """
    r = rng or random
    now = now or datetime.now()

    if now.timestamp() - last_drift_ts < _MOOD_DRIFT_MIN_GAP_SECONDS:
        return 0.0, 0.0, last_drift_ts

    anchor_v, anchor_e = _TIME_OF_DAY_MOOD[_time_of_day_bucket(now)]

    def _one_axis(current: float, anchor: float) -> float:
        step = r.uniform(-_MOOD_DRIFT_STEP, _MOOD_DRIFT_STEP)
        candidate = current + step
        # Settle toward where the hour tends to sit.
        candidate = candidate + (anchor - candidate) * 0.08
        # Dead neutral is not a resting state: sidle off the exact center,
        # in a random direction so rest never has one preferred corner.
        if abs(candidate) < 0.05:
            candidate += 0.02 if r.random() < 0.5 else -0.02
        # The band: ambience only operates within anchor +/- band. Past it
        # (conversation carried him there), drift holds at the edge instead of
        # ever pushing further out.
        if abs(candidate - anchor) > _MOOD_DRIFT_BAND:
            if candidate > anchor:
                candidate = anchor + _MOOD_DRIFT_BAND
            else:
                candidate = anchor - _MOOD_DRIFT_BAND
        return candidate - current

    # No extra clamp here: _one_axis already promises the band exactly, and
    # clamping the walk-home delta to +-step would strand the mood slightly
    # outside the band forever.
    return _one_axis(current_valence, anchor_v), \
        _one_axis(current_energy, anchor_e), now.timestamp()


# A PURE state question is about his state and NOTHING else: "how are you",
# "are you ok", "why arent you happy". Those deserve a full honest line about
# his actual mood. Anything else that merely touches the topic ("how are you
# liking the game", "how are you so good at this") is ordinary conversation.
def is_pure_state_question(text: str) -> bool:
    """True when a message is ONLY a question/remark about his state."""
    if not detect_state_question(text):
        return False
    stripped = text.strip().rstrip("?!. ").lower()
    stripped = re.sub(r"^(so|ok|okay|hey|yo|and|but|well)[,.!]?\s*", "", stripped)
    # If anything concrete survives the state words, it is a mixed question.
    residue = re.sub(
        r"\b(how|r|are|is|u|you|your|why|what|it|its|it's|feeling|feel|feels|"
        r"doing|doin|going|going on|seem|seems|am|i|the|a|an|this|that|so|not|"
        r"isnt|isn't|arent|aren't|dont|don't|just|like|being|be|in|of|ok|"
        r"okay|alright|all right|happy|sad|mad|angry|upset|bored|tired|awake|"
        r"asleep|lonely|real|alive|there|quiet|calm|excited|down|mood|neutral|"
        r"bad|good|right|now|today|lately|these|days|s|t|re|m|d|ve|ll|"
        r"anything|everything|stuff|up|to|me|we)\b", "", stripped)
    return len(residue.strip()) < 4
