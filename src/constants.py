# constants.py
# Central place for key column names and domain groupings (avoids repetition across notebooks/src)

ECD_ITEMS = [
    "phys_walk_uneven_surface","phys_jump_two_feet","phys_dress_self","phys_buttons",
    "lang_words_10plus","lang_sentence_3plus","lang_sentence_5plus","lang_use_pronouns","lang_name_objects",
    "lit_letters_5plus","lit_write_name","lit_numbers_1_5","lit_count_3_objects","lit_count_10_objects",
    "soc_independent_activity","soc_name_familiar_people","soc_help_others","soc_get_along_children",
    "soc_often_sad","soc_aggressive_behavior",
]

DOMAIN_ITEMS = {
    "physical": ["phys_walk_uneven_surface","phys_jump_two_feet","phys_dress_self","phys_buttons"],
    "language": ["lang_words_10plus","lang_sentence_3plus","lang_sentence_5plus","lang_use_pronouns","lang_name_objects"],
    "literacy": ["lit_letters_5plus","lit_write_name","lit_numbers_1_5","lit_count_3_objects","lit_count_10_objects"],
    "socio_emotional": ["soc_independent_activity","soc_name_familiar_people","soc_help_others","soc_get_along_children","soc_often_sad","soc_aggressive_behavior"],
}
