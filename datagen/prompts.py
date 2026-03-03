OBJECT_PROMPTS = {
    "pillow":     "pillow",
    "bed_sheet":  "bed sheet",
    "blanket":    "blanket",
    "floor":      "floor",
    "carpet":     "carpet",
    "chair":      "chair",
    "desk":       "desk",
    "mirror":     "mirror",
    "sofa":       "sofa",
    "bath_towel": "bath towel",
    "bin":        "rubbish bin",
    "window":     "window",
}

DEFECT_PROMPTS = {
    "pillow": [
        ("a dark strand of hair lying across the white pillow",               "hair"),
        ("a visible yellow stain on the white pillow",                        "stain"),
        ("a pillow with a large deep sunken hollow pressed into the center, foam visibly compressed, pillowcase heavily creased and rumpled around the depression", "indentation"),
        ("a blood stain on the white pillow",                                 "stain"),
    ],
    "bed_sheet": [
        ("dark hairs scattered across the white bed sheet",                   "hair"),
        ("a brown stain on the white bed sheet",                              "stain"),
        ("a blood stain on the white bed sheet",                              "stain"),
        ("food crumbs scattered across the bed sheet",                        "debris"),
        ("a coffee stain spreading across the white bed sheet",               "stain"),
    ],
    "blanket": [
        ("dark hair strands on the white blanket cover",                      "hair"),
        ("lint scattered across the blanket surface",                         "debris"),
        ("pet hair scattered across the blanket cover",                       "debris"),
        ("a visible stain on the blanket cover",                              "stain"),
    ],
    "floor": [
        ("a muddy shoe print on the tiled floor",                             "stain"),
        ("tissue paper left on the floor",                                    "litter"),
        ("food crumbs scattered on the floor",                                "debris"),
        ("a used plastic bag left on the floor",                              "litter"),
        ("a sauce spill stain on the tiled floor",                            "stain"),
        ("a drink spill stain on the tiled floor",                            "stain"),
    ],
    "carpet": [
        ("a visible stain on the carpet",                                     "stain"),
        ("food crumbs ground into the carpet",                                "debris"),
    ],
    "chair": [
        ("food crumbs scattered on the seat cushion",                         "debris"),
        ("a visible stain on the fabric chair seat",                          "stain"),
        ("a visible grease stain on the chair armrest",                       "stain"),
    ],
    "desk": [
        ("food takeaway containers left on the desk",                         "litter"),
        ("a single used disposable coffee cup left on the desk",              "litter"),
        ("crumpled food wrappers left on the desk",                           "litter"),
        ("a sticky spill stain on the desk surface",                          "stain"),
        ("a drink spill on the desk surface",                                 "stain"),
    ],
    "mirror": [
        ("a red lipstick smear across the mirror",                            "stain"),
        ("dried soap scum coating the mirror surface",                        "dirty"),
        ("mascara smeared across the mirror",                                 "stain"),
    ],
    "sofa": [
        ("a visible drink spill stain on the sofa cushion",                   "stain"),
        ("food crumbs scattered on the sofa cushions",                        "debris"),
    ],
    "bath_towel": [
        ("a dirty stained towel with visible brown marks",                    "stain"),
        ("a dirty stained towel hanging on the rack",                         "stain"),
        ("dark mildew spots on the white towel",                              "stain"),
    ],
    "bin": [
        ("a rubbish bin with crumpled tissues and wrappers visible inside",   "not_emptied"),
        ("a rubbish bin not emptied with visible waste inside",               "not_emptied"),
    ],
    "window": [
        ("dead insects on the window ledge",                                  "debris"),
        ("crumpled tissue paper left on the window sill",                     "litter"),
    ],
}
