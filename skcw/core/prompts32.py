"""32-class prompt set for D1 clean-latent generation (M1) and E_theta training.

Classes 0-3 are identical to p0_channel_capacity.PROMPTS (continuity with diag).
Classes 28-31 are the held-out evaluation classes (never seen in training).
"""
from __future__ import annotations

PROMPTS32 = {
    # --- carried over from P0 (0-3) ---
    0: "A fluffy orange cat slowly walks across a sunlit wooden floor, tail swaying, soft afternoon light.",
    1: "A red vintage car drives down a quiet city street lined with trees, smooth camera tracking shot.",
    2: "A tall waterfall cascades into a clear mountain pool surrounded by green pine forest, gentle mist.",
    3: "A dancer in a flowing blue dress spins gracefully on an empty stage under a single warm spotlight.",
    # --- animals (4-9) ---
    4: "A golden retriever runs along a sandy beach at sunset, waves rolling in softly behind it.",
    5: "A hummingbird hovers beside a red tropical flower, wings a blur, lush green background.",
    6: "A white horse gallops across a foggy green meadow in the early morning light.",
    7: "A sea turtle glides over a colorful coral reef in clear turquoise water, small fish around.",
    8: "A panda sits in a bamboo forest calmly chewing bamboo leaves, soft diffused daylight.",
    9: "An eagle soars over snow-capped mountain peaks under a clear blue sky, slow circling flight.",
    # --- nature / landscape (10-15) ---
    10: "Ocean waves crash against dark coastal rocks at dusk, spray rising, orange sky on the horizon.",
    11: "A field of sunflowers sways gently in the wind under a bright summer sky with white clouds.",
    12: "Snow falls slowly over a quiet pine forest, branches collecting white powder, grey winter sky.",
    13: "A river winds through a golden autumn valley, red and yellow trees reflected in the water.",
    14: "Northern lights ripple in green curtains over a frozen lake and silhouetted mountains at night.",
    15: "Morning fog drifts through terraced rice fields on a hillside, soft golden sunrise light.",
    # --- urban / objects (16-21) ---
    16: "A steaming cup of coffee sits on a wooden cafe table by a rainy window, blurred city lights outside.",
    17: "A modern city skyline at night with glowing skyscrapers, light trails of traffic on the highway below.",
    18: "A street market at dusk with colorful fruit stalls and warm hanging lanterns, people browsing slowly.",
    19: "A cable car ascends over a green mountain valley, cabins swaying gently, distant peaks in haze.",
    20: "A lighthouse stands on a cliff as its beam sweeps across a dark stormy sea at night.",
    21: "A vintage steam train crosses a tall stone bridge through misty mountains, white smoke trailing.",
    # --- people / activity (22-27) ---
    22: "A chef tosses vegetables in a flaming wok in a busy restaurant kitchen, sparks of flame rising.",
    23: "A child flies a red kite on a grassy hill, the kite dancing against a bright cloudy sky.",
    24: "A violinist plays on a rainy street corner at night, neon reflections shimmering on wet pavement.",
    25: "A surfer rides a large curling wave, spray trailing off the crest, bright tropical sunlight.",
    26: "An elderly painter works at an easel in a sunlit studio, brush moving over a half-finished canvas.",
    27: "A cyclist rides along a tree-lined country road in autumn, fallen leaves swirling behind the wheels.",
    # --- HELD-OUT eval classes (28-31), never used in training ---
    28: "A hot air balloon drifts over a patchwork of green farmland at sunrise, long soft shadows.",
    29: "Koi fish swim slowly in a clear garden pond, ripples distorting the reflection of a red maple tree.",
    30: "A potter shapes a clay bowl on a spinning wheel, hands wet with slip, warm workshop light.",
    31: "Rain falls on a quiet temple courtyard, water dripping from curved roof tiles onto stone steps.",
}

TRAIN_CLASSES = list(range(28))        # 0-27
HELDOUT_CLASSES = [28, 29, 30, 31]     # never in training

assert len(PROMPTS32) == 32
