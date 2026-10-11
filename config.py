"""ALL tunable settings live here. After changing one, re-run the stages listed in the README table."""

# ---------------- Model (change => re-run stages 3,4,5,6) ----------------
MODEL_NAME = "openai/clip-vit-base-patch32"    # any Hugging Face CLIP checkpoint, e.g.
#   "openai/clip-vit-base-patch16"  (more accurate, ~4x slower on CPU)
#   "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
BATCH_SIZE = 32                                 # lower to 8-16 if RAM is tight
TEXT_PROMPT = "a photo of {}"                   # prompt used for category text prototypes

# ---------------- Stage 1: data collection ----------------
CONTACT_EMAIL = "prithampsr@gmail.com"        # REQUIRED by Wikimedia's API policy
PER_CAT = 60                                   # files pulled per seed category
SEEDS = ["Taj Mahal", "Charminar", "Eiffel Tower", "Cats", "Dogs", "Sunsets", "Mountains",
         "Beaches", "Trains", "Bicycles", "Roses", "Butterflies", "Cathedrals", "Bridges",
         "Lighthouses", "Windmills", "Fountains", "Airplanes", "Tractors", "Pizza", "Sushi",
         "Cricket", "Mosques", "Hindu temples", "Castles", "Waterfalls", "Lakes", "Owls",
         "Elephants", "Horses", "Cows", "Sailing ships", "Motorcycles", "Buses", "Trams",
         "Street food", "Markets", "Murals", "Statues", "Clouds", "Snow", "Deserts",
         "Forests", "Volcanoes", "Mushrooms", "Spiders", "Football", "Libraries"]

# ---------------- Stage 2: label cleaning + splits (change => re-run 2..6) ----------------
N_MIN = 20                       # min TRAIN images for a category to enter the vocabulary
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10   # the remaining 20% (latest uploads) is the TEST set
DEDUP_THRESH = 0.97             # cosine vs. train above this => near-duplicate, removed from val/test
MAINT_PATTERNS = [              # regexes (case-insensitive) for process/licence/author categories
    r"uploaded", r"self-published", r"^files? (from|by)\b", r"media (needing|missing|with)",
    r"^pages? with", r"^cc-", r"^gfdl", r"^pd-", r"photographs? by", r"images? by",
    r"pictures? by", r"photos? by", r"quality images", r"featured", r"valued image",
    r"pictures of the day", r"backlog", r"cleanup", r"^wikimedia", r"flickr", r"^template",
    r"^licen[sc]e", r"^taken with", r"^exif", r"^unidentified", r"^media contributed",
    r"^images from", r"^photographs taken on",
]

# ---------------- Scorers / fusion (change => re-run 5,6; no re-embedding) ----------------
KNN_K = 20                      # neighbours that vote
KNN_TAU = 0.05                  # vote temperature (lower = trust closest neighbours more)
CANDIDATES_PER_SCORER = 50      # fusion only looks at the union of each scorer's top-N
FALLBACK_WEIGHTS = (0.6, 0.2, 0.2)   # knn, cls, txt weights used when fusion.joblib is absent

# ---------------- Stage 4: supervised head (change => re-run 4,5,6) ----------------
HEAD_EPOCHS, HEAD_LR, HEAD_WD, HEAD_BS = 40, 1e-3, 1e-4, 128
ASL_GAMMA_POS, ASL_GAMMA_NEG, ASL_MARGIN = 0.0, 4.0, 0.05
HEAD_INIT_SCALE, HEAD_INIT_BIAS = 20.0, -6.0   # text-initialised classifier: w = scale * t_c

# ---------------- Stage 6: evaluation ----------------
EVAL_KS = (1, 3, 5)

# ---------------- Stage 1 (v2): walk sub-categories ----------------
SUBCAT_DEPTH = 2          # how many levels of sub-folders to open below each seed
SUBCATS_PER_NODE = 20     # how many sub-folders to open per category
PER_LEAF = 30             # images taken from each category visited
MAX_PER_SEED = 500        # cap so one seed can't dominate the dataset

KNN_PRIOR_ALPHA = 0.5   # damps kNN votes of user-taught categories that have more examples than a typical one (0 = off, 1 = full)

# ---- extra seeds (v5): new topics for more data and more categories ----
SEEDS += ["Birds", "Fish", "Flowers", "Fruit", "Vegetables", "Bread", "Coffee", "Cakes",
          "Rivers", "Glaciers", "Islands", "Cliffs", "Caves", "Canyons", "Parks", "Gardens",
          "Skyscrapers", "Museums", "Churches", "Palaces", "Ruins", "Monasteries", "Stadiums",
          "Railway stations", "Airports", "Cars", "Trucks", "Helicopters", "Boats", "Tennis",
          "Basketball", "Cycling", "Guitars", "Paintings", "Sculptures", "Fireworks",
          "Festivals", "Dances", "Traditional clothing", "Insects", "Frogs", "Snakes",
          "Monkeys", "Bears", "Deer", "Tigers", "Lions", "Trees", "Rainbows", "Fog",
          "Qutub Minar", "Golconda Fort", "Red Fort"]

# ---- extra seeds (v6): different domains ----
_NEW = ["Samosas", "Biryani", "Tea", "Rice", "Spices", "Chocolate", "Ice cream", "Cheese",
        "Wine", "Beer", "Noodles", "Salads", "Soups", "Burgers", "Cricket stadiums", "Chess",
        "Swimming", "Skiing", "Boxing", "Wrestling", "Golf", "Rugby", "Baseball", "Horse racing",
        "Violins", "Pianos", "Drums", "Flutes", "Clocks", "Watches", "Chairs", "Lamps",
        "Telescopes", "Microscopes", "Computers", "Mobile phones", "Robots", "Satellites",
        "Rockets", "Ships", "Submarines", "Ferries", "Rickshaws", "Scooters", "Ambulances",
        "Fire engines", "Tanks", "Bridges in India", "Gates", "Towers", "Domes", "Arches",
        "Stained glass", "Mosaics", "Graffiti", "Pottery", "Textiles", "Jewellery", "Masks",
        "Weddings", "Parades", "Protests", "Markets in India", "Temples in Hyderabad",
        "Rangoli", "Kites", "Balloons", "Ponds", "Swamps", "Hot springs", "Geysers",
        "Rice fields", "Vineyards", "Tea plantations", "Orchards", "Meadows", "Autumn leaves",
        "Cherry blossoms", "Sunflowers", "Orchids", "Lotus", "Cacti", "Ferns", "Moss",
        "Crabs", "Jellyfish", "Sharks", "Dolphins", "Whales", "Penguins", "Peacocks",
        "Parrots", "Eagles", "Ducks", "Goats", "Sheep", "Camels", "Giraffes", "Zebras",
        "Pandas", "Wolves", "Foxes", "Rabbits", "Squirrels", "Bees", "Ants"]
SEEDS += [s for s in _NEW if s not in SEEDS]

# ---- extra seeds (v6): different domains ----
_NEW = ["Samosas", "Biryani", "Tea", "Rice", "Spices", "Chocolate", "Ice cream", "Cheese",
        "Wine", "Beer", "Noodles", "Salads", "Soups", "Burgers", "Cricket stadiums", "Chess",
        "Swimming", "Skiing", "Boxing", "Wrestling", "Golf", "Rugby", "Baseball", "Horse racing",
        "Violins", "Pianos", "Drums", "Flutes", "Clocks", "Watches", "Chairs", "Lamps",
        "Telescopes", "Microscopes", "Computers", "Mobile phones", "Robots", "Satellites",
        "Rockets", "Ships", "Submarines", "Ferries", "Rickshaws", "Scooters", "Ambulances",
        "Fire engines", "Tanks", "Bridges in India", "Gates", "Towers", "Domes", "Arches",
        "Stained glass", "Mosaics", "Graffiti", "Pottery", "Textiles", "Jewellery", "Masks",
        "Weddings", "Parades", "Protests", "Markets in India", "Temples in Hyderabad",
        "Rangoli", "Kites", "Balloons", "Ponds", "Swamps", "Hot springs", "Geysers",
        "Rice fields", "Vineyards", "Tea plantations", "Orchards", "Meadows", "Autumn leaves",
        "Cherry blossoms", "Sunflowers", "Orchids", "Lotus", "Cacti", "Ferns", "Moss",
        "Crabs", "Jellyfish", "Sharks", "Dolphins", "Whales", "Penguins", "Peacocks",
        "Parrots", "Eagles", "Ducks", "Goats", "Sheep", "Camels", "Giraffes", "Zebras",
        "Pandas", "Wolves", "Foxes", "Rabbits", "Squirrels", "Bees", "Ants"]
SEEDS += [s for s in _NEW if s not in SEEDS]

# ---- extra seeds (v6): different domains ----
_NEW = ["Samosas", "Biryani", "Tea", "Rice", "Spices", "Chocolate", "Ice cream", "Cheese",
        "Wine", "Beer", "Noodles", "Salads", "Soups", "Burgers", "Cricket stadiums", "Chess",
        "Swimming", "Skiing", "Boxing", "Wrestling", "Golf", "Rugby", "Baseball", "Horse racing",
        "Violins", "Pianos", "Drums", "Flutes", "Clocks", "Watches", "Chairs", "Lamps",
        "Telescopes", "Microscopes", "Computers", "Mobile phones", "Robots", "Satellites",
        "Rockets", "Ships", "Submarines", "Ferries", "Rickshaws", "Scooters", "Ambulances",
        "Fire engines", "Tanks", "Bridges in India", "Gates", "Towers", "Domes", "Arches",
        "Stained glass", "Mosaics", "Graffiti", "Pottery", "Textiles", "Jewellery", "Masks",
        "Weddings", "Parades", "Protests", "Markets in India", "Temples in Hyderabad",
        "Rangoli", "Kites", "Balloons", "Ponds", "Swamps", "Hot springs", "Geysers",
        "Rice fields", "Vineyards", "Tea plantations", "Orchards", "Meadows", "Autumn leaves",
        "Cherry blossoms", "Sunflowers", "Orchids", "Lotus", "Cacti", "Ferns", "Moss",
        "Crabs", "Jellyfish", "Sharks", "Dolphins", "Whales", "Penguins", "Peacocks",
        "Parrots", "Eagles", "Ducks", "Goats", "Sheep", "Camels", "Giraffes", "Zebras",
        "Pandas", "Wolves", "Foxes", "Rabbits", "Squirrels", "Bees", "Ants"]
SEEDS += [s for s in _NEW if s not in SEEDS]

# ---- extra seeds (v6): different domains ----
_NEW = ["Samosas", "Biryani", "Tea", "Rice", "Spices", "Chocolate", "Ice cream", "Cheese",
        "Wine", "Beer", "Noodles", "Salads", "Soups", "Burgers", "Cricket stadiums", "Chess",
        "Swimming", "Skiing", "Boxing", "Wrestling", "Golf", "Rugby", "Baseball", "Horse racing",
        "Violins", "Pianos", "Drums", "Flutes", "Clocks", "Watches", "Chairs", "Lamps",
        "Telescopes", "Microscopes", "Computers", "Mobile phones", "Robots", "Satellites",
        "Rockets", "Ships", "Submarines", "Ferries", "Rickshaws", "Scooters", "Ambulances",
        "Fire engines", "Tanks", "Bridges in India", "Gates", "Towers", "Domes", "Arches",
        "Stained glass", "Mosaics", "Graffiti", "Pottery", "Textiles", "Jewellery", "Masks",
        "Weddings", "Parades", "Protests", "Markets in India", "Temples in Hyderabad",
        "Rangoli", "Kites", "Balloons", "Ponds", "Swamps", "Hot springs", "Geysers",
        "Rice fields", "Vineyards", "Tea plantations", "Orchards", "Meadows", "Autumn leaves",
        "Cherry blossoms", "Sunflowers", "Orchids", "Lotus", "Cacti", "Ferns", "Moss",
        "Crabs", "Jellyfish", "Sharks", "Dolphins", "Whales", "Penguins", "Peacocks",
        "Parrots", "Eagles", "Ducks", "Goats", "Sheep", "Camels", "Giraffes", "Zebras",
        "Pandas", "Wolves", "Foxes", "Rabbits", "Squirrels", "Bees", "Ants"]
SEEDS += [s for s in _NEW if s not in SEEDS]
