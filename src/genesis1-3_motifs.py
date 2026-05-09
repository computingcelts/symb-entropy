"""
Genesis 1-3 (KJV) Motif Dictionary
====================================
19 categories, 253 words, 93 phrases.

Used in: "Dancing the Semantic Shuffle: Symbolic Entropy and
Order-Sensitive Computational Semantics" (Monroy & Kurian, 2025)
"""

motif_dict = {
    # ========================================================================
    # TREE / AXIS MUNDI
    # Central axis connecting heaven and earth; knowledge and life
    # ========================================================================
    'Tree/Axis-Mundi': {
        'phrases': [
            'tree of life',           # Theological concept (appears 2x)
            'tree of knowledge',      # Central to Fall narrative (appears 2x)
            'tree of the knowledge',  # Full phrase variant
            'fruit of the tree',      # Action phrase
        ],
        'words': [
            'tree', 'trees', 'fruit', 'seed', 'yielding', 
            'herb', 'grass', 'plant', 'grow', 'grew', 
            'knowledge', 'bearing', 'green', 'ground', 
            'bring', 'brought', 'kind'
        ]
    },
    
    # ========================================================================
    # WATERS / FLOOD / SEA
    # Primordial waters, chaos, and life-giving fluid
    # ========================================================================
    'Waters/Sea': {
        'phrases': [
            'face of the deep',       # Primordial chaos (Genesis 1:2)
            'face of the waters',     # Spirit moving on waters
            'gathering together of the waters', # Separation act
            'divided the waters',     # Creation act
        ],
        'words': [
            'waters', 'water', 'seas', 'sea', 'river', 'deep', 
            'mist', 'rain', 'watered', 'gathered', 'parted', 
            'divided', 'face', 'pison', 'gihon', 'hiddekel', 
            'euphrates', 'havilah', 'ethiopia', 'assyria'
        ]
    },
    
    # ========================================================================
    # LIGHT
    # Divine light, celestial order, and temporal cycles
    # ========================================================================
    'Light': {
        'phrases': [
            'let there be light',     # Fiat lux - first creation (appears 1x)
            'there was light',        # Divine fulfillment
            'the evening and the morning', # Day formula (appears 6x)
            'divided the light',      # Separation act
        ],
        'words': [
            'light', 'lights', 'darkness', 'day', 'night', 
            'evening', 'morning', 'firmament', 'heaven', 'heavens', 
            'stars', 'rule', 'divide', 
            'signs', 'seasons', 'years', 'greater', 'lesser', 
            'first', 'second', 'third', 'fourth', 'fifth', 
            'sixth', 'seventh'
        ]
    },
    
    # ========================================================================
    # SHARPNESS / SWORD / THORN
    # Weapons, painful vegetation, barriers, and divine judgment
    # ========================================================================
    'Sharpness-Sword-Thorn': {
        'phrases': [
            'flaming sword',          # Guardian weapon at Eden's gate (Gen 3:24)
            'thorns also and thistles', # Curse on ground (Gen 3:18)
        ],
        'words': [
            'sword', 'flaming', 'thorns', 'thistles'
        ]
    },
    
    # ========================================================================
    # DEATH / MORTALITY
    # Death, dust, mortality, and return to earth
    # ========================================================================
    'Death/Mortality': {
        'phrases': [
            'shalt surely die',       # Divine warning (Gen 2:17, 3:4)
            'surely die',             # Serpent's contradiction variant
            'unto dust shalt thou return', # Mortality sentence (Gen 3:19)
            'dust shalt thou return', # Shorter variant
            'dust thou art',          # Mortality declaration
            'return unto the ground', # Return to earth
        ],
        'words': [
            'die', 'dust', 'return'
        ]
    },
    
    # ========================================================================
    # EATING / FORBIDDEN FRUIT
    # Consumption, eating, and the forbidden act
    # ========================================================================
    'Eating/Forbidden-Fruit': {
        'phrases': [
            'thou shalt not eat',     # Divine prohibition (Gen 2:17)
            'shalt not eat',          # Prohibition variant
            'ye shall not',           # Prohibition to Eve (Gen 3:3)
            'did eat',                # Fall action (Gen 3:6, 3:12)
            'i did eat',              # Adam's confession
            'she gave me',            # Adam blaming Eve
        ],
        'words': [
            'eat', 'eaten', 'eatest', 'fruit', 'food', 'meat'
        ]
    },
    
    # ========================================================================
    # CURSE / TOIL
    # Divine curse, sorrow, painful labor, and sweat
    # ========================================================================
    'Curse/Toil': {
        'phrases': [
            'cursed is the ground',   # Ground curse (Gen 3:17)
            'because thou hast',      # Judgment formula (Gen 3:14, 3:17)
            'in sorrow',              # Pain/sorrow formula (Gen 3:16, 3:17)
            'sweat of thy face',      # Toil consequence (Gen 3:19)
        ],
        'words': [
            'cursed', 'sorrow', 'sweat', 'till'
        ]
    },
    
    # ========================================================================
    # TRANSGRESSION / DECEPTION
    # Breaking commands, deception, and serpent's lies
    # ========================================================================
    'Transgression/Deception': {
        'phrases': [
            'ye shall be as gods',    # Serpent's promise (Gen 3:5)
            'shall be as gods',       # Variant
            'knowing good and evil',  # Temptation of knowledge (Gen 3:5)
            'the serpent beguiled',   # Eve's explanation (Gen 3:13)
            'serpent beguiled me',    # Confession variant
        ],
        'words': [
            'beguiled', 'subtil'
        ]
    },
    
    # ========================================================================
    # DESIRE / WISDOM
    # Visual temptation, desire for wisdom, and epistemological lust
    # ========================================================================
    'Desire/Wisdom': {
        'phrases': [
            'pleasant to the eyes',   # Visual temptation (Gen 3:6)
            'desired to make one wise', # Desire for wisdom (Gen 3:6)
            'to make one wise',       # Wisdom temptation variant
            'make one wise',          # Shorter variant
            'a tree to be desired',   # Desirability of tree (Gen 3:6)
            'good for food',          # Sensory appeal (Gen 3:6)
        ],
        'words': [
            'desired', 'desire', 'pleasant', 'wise'
        ]
    },
    
    # ========================================================================
    # FEAR / HIDING
    # Post-Fall fear, hiding from God, and psychological shame
    # ========================================================================
    'Fear/Hiding': {
        'phrases': [
            'i was afraid',           # Adam's fear (Gen 3:10)
            'i heard thy voice',      # Hearing with fear (Gen 3:10)
            'heard thy voice',        # Variant
            'i hid myself',           # Adam's hiding (Gen 3:10)
            'hid themselves',         # Couple hiding (Gen 3:8)
            'hid themselves from',    # Full hiding phrase (Gen 3:8)
        ],
        'words': [
            'afraid', 'hid'
        ]
    },
    
    # ========================================================================
    # EXILE / EXPULSION
    # Driving out from Eden, sending forth
    # ========================================================================
    'Exile/Expulsion': {
        'phrases': [
            'drove out the man',      # Expulsion act (Gen 3:24)
            'sent him forth',         # Sending from garden (Gen 3:23)
            'from the garden of eden', # Departure location
            'from the garden',        # Shorter variant
        ],
        'words': [
            'drove', 'sent', 'forth'
        ]
    },
    
    # ========================================================================
    # SERPENT / BEAST
    # Cunning beasts, chaos creatures, and animal life
    # ========================================================================
    'Serpent/Beast': {
        'phrases': [
            'beast of the field',     # Serpent's domain (appears 7x)
            'beast of the earth',     # Creation category
            'fowl of the air',        # Flying creatures (appears 5x)
            'fish of the sea',        # Aquatic creatures
            'living creature',        # General life
            'every living thing',     # Comprehensive life
        ],
        'words': [
            'serpent', 'beast', 'creature', 'creeping', 
            'creepeth', 'cattle', 'fowl', 'living', 'subtil', 
            'field', 'winged', 'moveth', 'whales', 'fly', 
            'fish', 'air', 'life', 'abundantly', 'multiply', 
            'enmity', 'bruise', 'belly', 'cursed', 'beguiled'
        ]
    },
    
    # ========================================================================
    # BRIDEGROOM-BRIDE
    # Marriage, union, human relationship and sexuality
    # ========================================================================
    'Bridegroom-Bride': {
        'phrases': [
            'male and female',        # Creation duality (appears 2x)
            'bone of my bones',       # Adam's recognition (appears 1x)
            'flesh of my flesh',      # Union phrase
            'one flesh',              # Marriage union (appears 1x)
            'help meet',              # Eve's role (appears 2x)
            'man and his wife',       # Marital pair
            'a living soul',          # Human essence (appears 1x)
        ],
        'words': [
            'man', 'woman', 'wife', 'husband', 'adam', 'eve', 
            'male', 'female', 'bone', 'bones', 'flesh', 'cleave', 
            'leave', 'father', 'mother', 'alone', 'help', 'meet', 
            'together', 'sleep', 'slept', 'ribs', 'rib', 'desire', 
            'conception', 'children', 'seed', 'living', 'hearkened', 
            
        ]
    },
    
    # ========================================================================
    # GARDEN / PARADISE
    # Sacred space, Eden, and geographical landmarks
    # ========================================================================
    'Garden/Paradise': {
        'phrases': [
            'garden of eden',         # Paradise location (appears 4x)
            'midst of the garden',    # Central location (appears 2x)
            'out of eden',            # Exile phrase
            'east of the garden',     # Post-exile location
        ],
        'words': [
            'garden', 'eden', 'pleasant', 'midst', 'east', 
            'eastward', 'planted', 'dress', 'keep', 'place', 
            'land', 'gold', 'bdellium', 'onyx', 'stone', 
            'compasseth', 'river', 'heads', 'parted', 'cherubims', 
            'way', 'whole', 'good', 'sight', 'food'
        ]
    },
    
    # ========================================================================
    # CLOTHING / NAKEDNESS
    # Shame, covering, and transformed consciousness
    # ========================================================================
    'Clothing/Nakedness': {
        'phrases': [
            'they were both naked',   # Pre-Fall description
            'the eyes of them both were opened', # Consciousness change
            'were not ashamed',       # Pre-Fall state
            'coats of skins',         # Divine covering
        ],
        'words': [
            'naked', 'clothed', 'coats', 'skins', 'aprons', 
            'fig', 'leaves', 'sewed', 'ashamed', 'opened', 
            'eyes', 'knew', 'hid', 'afraid', 'closed'
        ]
    },
    
    # ========================================================================
    # SACRED NAME / WORD
    # Divine speech, naming, and creative command
    # ========================================================================
    'Sacred-Name/Word': {
        'phrases': [
            'the lord god',           # Divine name (appears 20x)
            'and god said',           # Creation formula (appears 9x)
            'god said, let',          # Command structure (appears 8x)
            'and god saw',            # Divine approval (appears 7x)
            'god saw that',           # Evaluation phrase
            'it was good',            # Divine verdict (appears 7x)
            'in the beginning',       # Opening phrase (appears 1x)
            'image of god',           # Imago Dei (appears 3x)
            'spirit of god',          # Divine presence
            'breath of life',         # Life-giving act (appears 2x)
            'let us make',            # Divine counsel
        ],
        'words': [
            'god', 'lord', 'name', 'names', 'called', 'call', 
            'voice', 'commanded', 'saying', 'spirit', 'blessed', 
            'sanctified', 'created', 'made', 'rested', 'generations', 
            'breathed', 'formed', 'nostrils', 'breath', 'soul', 
            'image', 'likeness', 'dominion', 'replenish', 'subdue', 
            'behold', 'beginning', 'finished', 'host', 'work'
        ]
    },
    
    # ========================================================================
    # GOOD-EVIL / KNOWLEDGE
    # Moral knowledge, epistemological themes, and ethical consciousness
    # ========================================================================
    'Good-Evil/Knowledge': {
        'phrases': [
            'good and evil',          # Central moral concept (appears 4x)
            'knowledge of good',      # Epistemological phrase
        ],
        'words': [
            'good', 'evil', 'knowing', 'knowledge', 'wise', 
            'wisdom', 'eyes', 'opened', 'gods', 'die', 
            'death', 'surely', 'eat', 'eaten', 'touch'
        ]
    },
    
    # ========================================================================
    # COMMAND / OBEDIENCE
    # Divine imperatives, prohibitions, and speech formulas
    # ========================================================================
    'Command/Obedience': {
        'phrases': [
            'god commanded',          # Divine imperative
            'thou shalt not',         # Prohibition formula (appears 3x)
            'thou shalt surely',      # Emphasis formula
            'let there be',           # Creation command (appears 6x)
            'let them have',          # Dominion grant
            'said unto',              # Speech formula (appears 7x)
        ],
        'words': [
            'commanded', 'saying', 'thou', 'shalt', 'mayest', 
            'freely', 'whereof', 'shouldest',
        ]
    },
    
    # ========================================================================
    # TIME / ORDER
    # Temporal structure, cosmological sequence, and creation days
    # ========================================================================
    'Time/Order': {
        'phrases': [
            'in the beginning',       # Temporal origin
            'the first day',          # Creation sequence (appears 6x)
            'the second day',         # Day 2
            'the third day',          # Day 3
            'the fourth day',         # Day 4
            'the fifth day',          # Day 5
            'the sixth day',          # Day 6
            'the seventh day',        # Sabbath (appears 3x)
            'evening and morning',    # Day formula (appears 6x)
        ],
        'words': [
            'day', 'days', 'beginning', 'first', 'second', 
            'third', 'fourth', 'fifth', 'sixth', 'seventh', 
            'evening', 'morning', 'seasons', 'years', 
            'generations', 'finished'
        ]
    }
}
