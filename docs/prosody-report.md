# F0 temporal-structure report

> **Do not cite the human-vs-synthetic direction.** The human baseline is silence-segmented reference audio (multi-sentence blocks, few speakers), not humans reading the bench prompts, so it is not comparable to the synthetic clips. The per-model tables are descriptive only: no direction is established as more human-like, and nothing here has been validated against arena votes.

Descriptive aggregate (not a finding): MIXED: English=no, French=mixed for less-negative declination plus weaker absolute terminal movement.

This is a descriptive corpus result, not a replication of the paper's experiment. “Weaker terminal movement” is operationalized as a smaller median of `|terminal_slope|`; declination is compared as a signed slope (higher means less negative).

The human side has 102 metric-valid segments, but only 18 unique source clips; segments from a source are not independent. Human-vs-synthetic claims therefore remain weak.

## Human baseline and aggregate synthetic comparison

| language | set | n | declination median | terminal median | median abs terminal | paper pattern? |
|---|---|---:|---:|---:|---:|---|
| English (prompts 1-4) | human | 11 | -2.507 | -5.058 | 13.404 | baseline |
| English (prompts 1-4) | synthetic | 1558 | -3.871 | -13.455 | 15.802 | no |
| French (prompt 5) | human | 2 | -9.138 | -3.870 | 3.870 | baseline |
| French (prompt 5) | synthetic | 203 | -3.818 | -7.927 | 10.480 | mixed |

English prompts 1-4 and French prompt 5 are never pooled. Human language labels come only from named repository references: `juliette.wav` is French; `chris_hemsworth*`, `dafoe*`, and `jo.wav` are English. Distinct `_reference.wav` content without a named match is labeled unknown and excluded from language comparisons.

## Per-model ranking (primary result)

One row per model and mode (preset and cloning are never pooled). Terminal rank orders the smallest median of `|terminal_slope|` first (weakest movement); the signed median is shown alongside but is not ranked, because rises and falls cancel in it. Declination rank orders the highest signed median first (least negative/most positive). `unscored n` counts published clips with no scores.csv row yet (no health/WER gate).

### English (prompts 1-4)

9/148 model medians meet both descriptive directions relative to the language-matched human baseline.

| terminal rank | model | mode | n | median abs terminal | terminal median (signed) | declination median | declination rank | unscored n |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | sanotts_heart_nano | default | 6 | 2.134 | 2.134 | -2.334 | 18 | 0 |
| 2 | sanotts_amy | default | 8 | 3.149 | -2.343 | -1.843 | 9 | 0 |
| 3 | woof | default | 6 | 3.390 | -2.103 | -2.579 | 25 | 0 |
| 4 | audio8_01b | cloning | 16 | 4.288 | -1.971 | -3.586 | 64 | 0 |
| 5 | miotts_06b | cloning | 4 | 4.531 | -3.764 | -6.139 | 132 | 0 |
| 6 | coqui | default | 13 | 4.922 | -0.786 | -3.791 | 70 | 0 |
| 7 | steertts_noflow | default | 16 | 5.078 | -5.078 | -2.514 | 23 | 0 |
| 8 | steertts | cloning | 16 | 5.123 | -2.186 | -5.066 | 115 | 0 |
| 9 | luxtts | default | 5 | 5.304 | 2.810 | -3.600 | 65 | 0 |
| 10 | steertts | default | 16 | 5.349 | -5.349 | -2.274 | 16 | 0 |
| 11 | mars5 | default | 7 | 5.617 | -5.617 | -9.897 | 148 | 0 |
| 12 | miratts | cloning | 8 | 5.619 | -2.540 | -6.368 | 133 | 0 |
| 13 | orpheus | default | 3 | 5.939 | -5.939 | -6.083 | 131 | 0 |
| 14 | dots_tts | cloning | 3 | 6.233 | -6.233 | -2.507 | 22 | 0 |
| 15 | vibevoice | default | 16 | 6.316 | -4.990 | -4.354 | 85 | 0 |
| 16 | steertts_noflow | cloning | 16 | 6.346 | -5.760 | -4.639 | 97 | 0 |
| 17 | indextts25 | cloning | 13 | 6.673 | -4.707 | -6.442 | 136 | 0 |
| 17 | indextts25 | default | 13 | 6.673 | -4.707 | -6.442 | 136 | 0 |
| 19 | qwentts | default | 16 | 6.863 | -4.293 | -2.460 | 21 | 0 |
| 20 | miso | cloning | 7 | 6.920 | -6.196 | -6.473 | 138 | 0 |
| 21 | chatterbox | cloning | 15 | 7.004 | -5.782 | -4.011 | 76 | 0 |
| 22 | openvoice | cloning | 16 | 7.286 | -7.286 | -4.924 | 111 | 0 |
| 23 | miratts | default | 4 | 7.499 | 1.880 | -3.882 | 72 | 0 |
| 24 | indextts | cloning | 11 | 7.537 | 1.393 | -6.487 | 140 | 0 |
| 25 | cosyvoice | cloning | 3 | 7.712 | -7.712 | -3.554 | 61 | 0 |
| 26 | outetts | cloning | 8 | 7.969 | -7.354 | -2.735 | 30 | 0 |
| 27 | f5tts | cloning | 15 | 7.983 | -7.983 | -3.131 | 46 | 0 |
| 28 | moss_tts_v15 | default | 8 | 8.305 | -8.305 | -4.486 | 92 | 0 |
| 29 | omnivoice | cloning | 12 | 8.314 | -7.429 | -2.719 | 29 | 0 |
| 30 | openvoice | default | 16 | 8.440 | -8.440 | -5.025 | 112 | 0 |
| 31 | qwentts_fast | default | 7 | 8.562 | -3.558 | -5.734 | 126 | 0 |
| 32 | fish_15 | cloning | 15 | 8.782 | -8.782 | -5.594 | 124 | 0 |
| 33 | melotts | default | 16 | 8.805 | -8.805 | -3.962 | 75 | 0 |
| 34 | f5tts | default | 16 | 8.980 | -7.234 | -2.082 | 13 | 0 |
| 35 | pocket | cloning | 8 | 9.196 | -9.196 | -3.464 | 58 | 0 |
| 36 | firered3 | cloning | 8 | 9.402 | -8.473 | -2.112 | 14 | 0 |
| 36 | firered3 | default | 8 | 9.402 | -8.473 | -2.112 | 14 | 0 |
| 38 | audio8_fast | cloning | 8 | 9.445 | -9.445 | -4.814 | 103 | 0 |
| 39 | neutts_air | default | 14 | 9.743 | -9.743 | -2.376 | 20 | 0 |
| 40 | phonon_31m | default | 4 | 10.284 | -10.284 | -5.540 | 121 | 0 |
| 41 | phonon_40m | default | 4 | 10.489 | -10.489 | -5.274 | 118 | 0 |
| 42 | outetts | default | 11 | 10.584 | -10.584 | -3.086 | 41 | 0 |
| 43 | coqui | cloning | 12 | 10.683 | -9.804 | -3.236 | 50 | 0 |
| 44 | fish_15 | default | 15 | 10.907 | -9.711 | -5.967 | 128 | 0 |
| 45 | higgs_v3 | cloning | 4 | 10.982 | -10.982 | -3.523 | 59 | 0 |
| 46 | audio8_01b | default | 16 | 11.075 | -11.075 | -4.562 | 94 | 0 |
| 47 | fish_s2 | cloning | 4 | 11.222 | -11.222 | -4.025 | 77 | 0 |
| 47 | fish_s2 | default | 4 | 11.222 | -11.222 | -4.025 | 77 | 0 |
| 49 | indextts | default | 12 | 11.228 | -11.228 | -5.686 | 125 | 0 |
| 50 | audio8 | cloning | 16 | 11.311 | -11.311 | -4.702 | 100 | 0 |
| 51 | longcat_1b | cloning | 8 | 11.317 | -6.533 | -3.278 | 53 | 0 |
| 52 | ming_tiny | cloning | 2 | 11.697 | -10.484 | -3.628 | 68 | 0 |
| 53 | wavtts | cloning | 4 | 11.791 | -7.536 | -4.597 | 95 | 0 |
| 54 | neutts_nano | default | 14 | 11.907 | -10.900 | -2.845 | 33 | 0 |
| 55 | auroratts | default | 16 | 12.151 | -12.151 | -2.345 | 19 | 0 |
| 56 | sesame | cloning | 13 | 12.203 | -3.887 | -4.353 | 84 | 0 |
| 57 | audio8_fast | default | 8 | 12.374 | -12.374 | -2.890 | 36 | 0 |
| 58 | luxtts | cloning | 8 | 12.409 | -12.409 | -4.840 | 107 | 0 |
| 59 | sesame | default | 13 | 13.090 | -13.090 | -4.425 | 88 | 0 |
| 60 | qwentts_fast | cloning | 8 | 13.216 | -11.763 | -6.610 | 141 | 0 |
| 61 | echo | cloning | 8 | 13.365 | -13.365 | -3.569 | 62 | 0 |
| 61 | echo | default | 8 | 13.365 | -13.365 | -3.569 | 62 | 0 |
| 63 | neutts_air | cloning | 14 | 13.530 | -5.963 | -1.666 | 5 | 0 |
| 64 | zonos3_mini | cloning | 16 | 13.817 | -13.817 | -2.811 | 32 | 0 |
| 65 | ribt | cloning | 12 | 13.989 | -13.989 | -3.945 | 74 | 0 |
| 66 | lfm3_audio | default | 16 | 14.247 | -14.247 | -3.878 | 71 | 0 |
| 67 | dia | default | 6 | 14.293 | 1.829 | -1.307 | 3 | 0 |
| 68 | cosyvoice | default | 4 | 14.298 | -14.298 | -3.604 | 66 | 0 |
| 69 | audio8_realistic | cloning | 16 | 14.408 | -12.459 | -2.317 | 17 | 0 |
| 70 | chatterbox_turbo | cloning | 15 | 14.615 | -13.224 | -1.770 | 6 | 0 |
| 71 | vibevoice_7b | cloning | 6 | 14.620 | -14.620 | -5.322 | 119 | 0 |
| 72 | chatterbox_turbo | default | 16 | 14.706 | -3.217 | -4.482 | 90 | 0 |
| 73 | sopro_streaming | cloning | 15 | 14.724 | -14.724 | -3.221 | 48 | 0 |
| 74 | audio8 | default | 16 | 14.961 | -14.961 | -2.981 | 40 | 0 |
| 75 | lfm2_audio | default | 13 | 15.248 | -15.248 | -5.852 | 127 | 0 |
| 76 | sopro_streaming | default | 15 | 15.353 | -15.353 | -3.106 | 44 | 0 |
| 77 | zonos2 | cloning | 4 | 15.722 | -15.722 | -4.652 | 98 | 0 |
| 78 | dramabox | cloning | 8 | 15.734 | -15.734 | -6.768 | 142 | 0 |
| 79 | moss_tts_v15 | cloning | 7 | 15.858 | -12.406 | -4.750 | 101 | 0 |
| 80 | styletts2 | cloning | 16 | 15.907 | -15.907 | -4.753 | 102 | 0 |
| 81 | metavoice | cloning | 2 | 16.246 | -16.246 | -5.549 | 122 | 0 |
| 81 | metavoice | default | 2 | 16.246 | -16.246 | -5.549 | 122 | 0 |
| 83 | ming_tiny | default | 4 | 16.449 | 13.301 | -1.877 | 11 | 0 |
| 84 | moss_tts_nano | cloning | 16 | 16.752 | -14.839 | -4.163 | 80 | 0 |
| 85 | moss_tts_nano | default | 15 | 17.325 | -10.950 | -5.241 | 117 | 0 |
| 86 | audio8_realistic | default | 16 | 18.030 | -18.030 | -1.847 | 10 | 0 |
| 87 | kittentts | default | 8 | 18.412 | -18.412 | -5.438 | 120 | 0 |
| 88 | soprano | default | 16 | 18.437 | -18.437 | -6.394 | 135 | 0 |
| 89 | ming | cloning | 6 | 18.491 | -18.491 | -4.844 | 108 | 0 |
| 90 | miso | default | 4 | 18.686 | 12.109 | -4.301 | 83 | 0 |
| 91 | longcat_3p5b | cloning | 8 | 18.713 | -18.713 | -2.906 | 37 | 0 |
| 92 | zipvoice | default | 14 | 19.114 | -19.114 | -3.093 | 42 | 0 |
| 93 | neutts_nano | cloning | 15 | 19.269 | -18.883 | -3.233 | 49 | 0 |
| 94 | moss_tts | default | 7 | 19.272 | -19.272 | -6.375 | 134 | 0 |
| 95 | auk_flash | cloning | 8 | 19.323 | -19.323 | -5.052 | 113 | 0 |
| 95 | auk_flash | default | 8 | 19.323 | -19.323 | -5.052 | 113 | 0 |
| 97 | vibevoice_7b | default | 8 | 19.707 | -18.720 | -6.856 | 143 | 0 |
| 98 | zipvoice | cloning | 16 | 19.739 | -19.061 | -3.462 | 57 | 0 |
| 99 | step_editx | cloning | 4 | 19.849 | -19.849 | -1.801 | 7 | 0 |
| 99 | step_editx | default | 4 | 19.849 | -19.849 | -1.801 | 7 | 0 |
| 101 | dramabox | default | 8 | 19.867 | -19.867 | -3.523 | 60 | 0 |
| 102 | sopro | cloning | 16 | 19.947 | -19.947 | -2.855 | 34 | 0 |
| 102 | sopro | default | 16 | 19.947 | -19.947 | -2.855 | 34 | 0 |
| 104 | breeze_tts2 | cloning | 8 | 20.293 | -20.293 | -4.896 | 110 | 0 |
| 105 | zonos | cloning | 12 | 20.298 | -12.243 | -2.761 | 31 | 0 |
| 106 | vibevoice_15b | default | 13 | 21.291 | -7.259 | -4.485 | 91 | 0 |
| 107 | zonos3_mini | default | 16 | 21.363 | -21.363 | -2.518 | 24 | 0 |
| 108 | scyllasband | default | 8 | 21.614 | -21.614 | -4.060 | 79 | 0 |
| 109 | miotts_01b | cloning | 3 | 22.003 | -22.003 | -4.815 | 104 | 0 |
| 110 | zonos | default | 13 | 22.079 | -7.946 | -3.373 | 55 | 0 |
| 111 | chatterbox | default | 13 | 22.116 | -10.096 | -4.524 | 93 | 0 |
| 112 | maya1 | default | 13 | 22.724 | -22.724 | -3.616 | 67 | 0 |
| 113 | linnet | default | 8 | 22.884 | -22.884 | -3.248 | 51 | 0 |
| 114 | styletts2 | default | 13 | 23.042 | -23.042 | -4.435 | 89 | 0 |
| 115 | voxtral | default | 4 | 23.250 | -18.115 | -1.390 | 4 | 0 |
| 116 | voxcpm | cloning | 13 | 23.317 | -12.108 | -3.410 | 56 | 0 |
| 117 | kittentts2 | cloning | 15 | 24.528 | -24.528 | -3.176 | 47 | 0 |
| 118 | moss_tts | cloning | 7 | 26.036 | -26.036 | -4.414 | 87 | 0 |
| 119 | longcat_1b | default | 8 | 26.253 | -8.757 | -6.958 | 145 | 0 |
| 120 | dia | cloning | 8 | 26.600 | 3.564 | -1.236 | 2 | 0 |
| 121 | vaniq | default | 16 | 26.724 | -6.349 | -2.939 | 38 | 0 |
| 122 | auk_base | cloning | 8 | 26.772 | -24.540 | -7.256 | 146 | 0 |
| 122 | auk_base | default | 8 | 26.772 | -24.540 | -7.256 | 146 | 0 |
| 124 | ribt | default | 16 | 27.690 | -27.690 | -4.295 | 82 | 0 |
| 125 | omnivoice | default | 13 | 27.983 | -25.334 | -3.304 | 54 | 0 |
| 126 | kittentts2_cpp | default | 8 | 28.864 | -28.864 | -4.239 | 81 | 0 |
| 127 | drama3 | cloning | 8 | 28.877 | -24.837 | -6.030 | 129 | 0 |
| 127 | drama3 | default | 8 | 28.877 | -24.837 | -6.030 | 129 | 0 |
| 129 | longcat_3p5b | default | 8 | 30.291 | -7.581 | -5.123 | 116 | 0 |
| 130 | voxcpm | default | 14 | 31.086 | -3.833 | -2.667 | 26 | 0 |
| 131 | parler | default | 13 | 32.765 | -28.491 | -4.377 | 86 | 0 |
| 132 | hojo_80m | cloning | 16 | 35.284 | -35.284 | -2.699 | 27 | 0 |
| 132 | hojo_80m | default | 16 | 35.284 | -35.284 | -2.699 | 27 | 0 |
| 134 | kittentts2 | default | 16 | 36.287 | -36.287 | -3.909 | 73 | 0 |
| 135 | piper | default | 8 | 37.920 | -37.920 | -4.632 | 96 | 0 |
| 136 | magpie | default | 16 | 38.352 | -35.899 | -3.098 | 43 | 0 |
| 137 | qwentts_06b_custom | default | 16 | 38.848 | -38.848 | -1.943 | 12 | 0 |
| 138 | dots_tts | default | 3 | 40.569 | -12.300 | -3.112 | 45 | 0 |
| 139 | ming | default | 8 | 41.473 | -33.384 | -6.479 | 139 | 0 |
| 140 | hojo_40m | default | 16 | 41.798 | -41.798 | -4.816 | 105 | 0 |
| 141 | higgs_v3 | default | 3 | 43.089 | -43.089 | -4.691 | 99 | 0 |
| 142 | supertonic | default | 8 | 44.128 | -44.128 | -6.874 | 144 | 0 |
| 143 | supratts | default | 14 | 45.450 | -45.450 | -1.117 | 1 | 0 |
| 144 | inflect_nano | default | 16 | 50.429 | -50.429 | -4.820 | 106 | 0 |
| 145 | kokoro | default | 12 | 52.352 | -52.352 | -3.777 | 69 | 0 |
| 146 | inflect_micro | default | 16 | 52.930 | -52.930 | -2.968 | 39 | 0 |
| 147 | pocket | default | 7 | 55.758 | -46.402 | -3.254 | 52 | 0 |
| 148 | kittentts08 | default | 8 | 61.778 | -61.778 | -4.879 | 109 | 0 |

### French (prompt 5)

10/74 model medians meet both descriptive directions relative to the language-matched human baseline.

| terminal rank | model | mode | n | median abs terminal | terminal median (signed) | declination median | declination rank | unscored n |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fish_s2 | cloning | 1 | 0.312 | 0.312 | -2.535 | 17 | 0 |
| 1 | fish_s2 | default | 1 | 0.312 | 0.312 | -2.535 | 17 | 0 |
| 3 | outetts | default | 2 | 0.791 | 0.791 | -6.198 | 57 | 0 |
| 4 | cosyvoice | cloning | 1 | 1.358 | 1.358 | 2.486 | 1 | 0 |
| 4 | cosyvoice | default | 1 | 1.358 | 1.358 | 2.486 | 1 | 0 |
| 6 | audio8 | cloning | 4 | 2.748 | -2.748 | -6.755 | 60 | 0 |
| 7 | voxtral | default | 1 | 3.142 | -3.142 | -2.545 | 20 | 0 |
| 8 | audio8_fast | cloning | 2 | 3.171 | 3.171 | -5.794 | 54 | 0 |
| 9 | moss_tts_nano | default | 4 | 3.544 | 3.544 | -7.600 | 63 | 0 |
| 10 | audio8 | default | 4 | 3.862 | -3.862 | -3.230 | 33 | 0 |
| 11 | fish_15 | default | 4 | 4.303 | -4.303 | -12.172 | 72 | 0 |
| 12 | ming | cloning | 2 | 4.408 | -4.408 | -5.514 | 52 | 0 |
| 13 | zonos | cloning | 4 | 4.858 | -1.095 | -0.209 | 4 | 0 |
| 14 | dots_tts | default | 1 | 5.138 | -5.138 | -5.284 | 51 | 0 |
| 15 | audio8_01b | default | 4 | 5.566 | -5.560 | -4.406 | 42 | 0 |
| 16 | drama3 | cloning | 2 | 5.905 | -5.905 | -9.728 | 67 | 0 |
| 16 | drama3 | default | 2 | 5.905 | -5.905 | -9.728 | 67 | 0 |
| 18 | moss_tts_v15 | cloning | 2 | 6.220 | -6.220 | -7.565 | 62 | 0 |
| 19 | kittentts2 | default | 4 | 6.265 | -6.265 | -4.307 | 40 | 0 |
| 20 | coqui | cloning | 3 | 6.637 | -6.175 | -7.991 | 66 | 0 |
| 21 | kittentts2_cpp | default | 2 | 6.643 | -6.643 | -2.804 | 28 | 0 |
| 22 | moss_tts_v15 | default | 2 | 7.237 | -5.129 | -6.355 | 59 | 0 |
| 23 | audio8_01b | cloning | 4 | 7.355 | -7.355 | -1.832 | 10 | 0 |
| 24 | audio8_fast | default | 2 | 7.471 | -7.471 | -2.691 | 27 | 0 |
| 25 | fish_15 | cloning | 4 | 7.707 | -7.707 | -4.554 | 43 | 0 |
| 26 | lfm3_audio | default | 4 | 7.772 | -7.772 | -3.578 | 35 | 0 |
| 27 | auk_flash | cloning | 2 | 7.801 | -7.801 | -11.377 | 70 | 0 |
| 27 | auk_flash | default | 2 | 7.801 | -7.801 | -11.377 | 70 | 0 |
| 29 | pocket | cloning | 2 | 7.802 | -7.802 | -5.149 | 48 | 0 |
| 30 | pocket | default | 2 | 7.942 | -7.942 | -2.997 | 30 | 0 |
| 31 | qwentts | default | 4 | 8.766 | -5.858 | -5.183 | 50 | 0 |
| 32 | zonos3_mini | default | 3 | 8.941 | -8.941 | -0.764 | 5 | 0 |
| 33 | moss_tts | default | 2 | 9.071 | -9.071 | -1.024 | 6 | 0 |
| 34 | moss_tts_nano | cloning | 3 | 9.758 | -9.758 | -6.135 | 56 | 0 |
| 35 | coqui | default | 4 | 9.869 | -9.869 | 0.509 | 3 | 0 |
| 36 | outetts | cloning | 1 | 9.901 | -9.901 | -6.262 | 58 | 0 |
| 37 | sopro_streaming | default | 4 | 10.137 | -10.137 | -2.508 | 16 | 0 |
| 38 | audio8_realistic | default | 4 | 10.138 | -10.138 | -1.518 | 9 | 0 |
| 39 | sopro_streaming | cloning | 4 | 10.265 | -10.265 | -2.399 | 15 | 0 |
| 40 | zonos3_mini | cloning | 3 | 10.785 | -10.785 | -7.724 | 64 | 0 |
| 41 | phonon_40m | default | 1 | 11.435 | -11.435 | -5.668 | 53 | 0 |
| 42 | qwentts_06b_custom | default | 4 | 11.755 | -7.902 | -1.943 | 11 | 0 |
| 43 | neutts_nano | default | 3 | 12.596 | 2.353 | -1.153 | 7 | 0 |
| 44 | sopro | cloning | 4 | 13.605 | -13.605 | -2.554 | 21 | 0 |
| 44 | sopro | default | 4 | 13.605 | -13.605 | -2.554 | 21 | 0 |
| 46 | omnivoice | cloning | 4 | 14.165 | -11.021 | -4.600 | 44 | 0 |
| 47 | neutts_nano | cloning | 4 | 14.700 | 14.700 | -2.017 | 12 | 0 |
| 48 | scyllasband | default | 2 | 14.946 | -14.946 | -2.581 | 23 | 0 |
| 49 | higgs_v3 | cloning | 1 | 14.955 | -14.955 | -2.678 | 25 | 0 |
| 50 | zonos | default | 4 | 14.976 | 14.976 | -2.584 | 24 | 0 |
| 51 | voxcpm | default | 4 | 15.377 | -15.377 | -4.823 | 47 | 0 |
| 52 | kokoro | default | 4 | 15.695 | -15.695 | -3.898 | 37 | 0 |
| 53 | melotts | default | 2 | 16.451 | -16.451 | -2.537 | 19 | 0 |
| 54 | kittentts2 | cloning | 4 | 16.900 | -16.900 | -4.633 | 45 | 0 |
| 55 | zipvoice | cloning | 4 | 17.309 | -17.309 | -3.211 | 32 | 0 |
| 56 | qwentts_fast | default | 2 | 17.970 | -17.970 | -10.203 | 69 | 0 |
| 57 | dots_tts | cloning | 1 | 18.251 | -18.251 | -7.215 | 61 | 0 |
| 58 | qwentts_fast | cloning | 2 | 18.511 | -18.511 | -5.156 | 49 | 0 |
| 59 | zipvoice | default | 3 | 19.425 | -5.851 | -2.258 | 14 | 0 |
| 60 | piper | default | 2 | 19.869 | -19.869 | -2.962 | 29 | 0 |
| 61 | audio8_realistic | cloning | 4 | 21.380 | 21.380 | -3.954 | 38 | 0 |
| 62 | openvoice | cloning | 4 | 21.520 | -21.520 | -3.872 | 36 | 0 |
| 63 | omnivoice | default | 4 | 22.413 | -22.413 | -2.683 | 26 | 0 |
| 64 | openvoice | default | 4 | 24.231 | -24.231 | -4.048 | 39 | 0 |
| 65 | moss_tts | cloning | 2 | 24.304 | 21.487 | -4.385 | 41 | 0 |
| 66 | supertonic | default | 2 | 26.508 | -26.508 | -5.839 | 55 | 0 |
| 67 | higgs_v3 | default | 1 | 30.232 | -30.232 | -1.476 | 8 | 0 |
| 68 | ming | default | 2 | 30.664 | -30.664 | -7.726 | 65 | 0 |
| 69 | voxcpm | cloning | 4 | 36.241 | -8.893 | -3.500 | 34 | 0 |
| 70 | magpie | default | 4 | 38.106 | -38.106 | -2.029 | 13 | 0 |
| 71 | ming_tiny | default | 1 | 43.192 | 43.192 | -4.650 | 46 | 0 |
| 72 | auk_base | cloning | 2 | 46.433 | 46.433 | -15.345 | 73 | 0 |
| 72 | auk_base | default | 2 | 46.433 | 46.433 | -15.345 | 73 | 0 |
| 74 | ming_tiny | cloning | 1 | 63.250 | 63.250 | -3.053 | 31 | 0 |

### Spanish (prompt 6)

0/36 model medians meet both descriptive directions relative to the language-matched human baseline.

| terminal rank | model | mode | n | median abs terminal | terminal median (signed) | declination median | declination rank | unscored n |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | kittentts2_cpp | default | 1 | 2.665 | -2.665 | -1.118 | 5 | 0 |
| 2 | zonos3_mini | default | 4 | 2.900 | -2.900 | -4.276 | 24 | 0 |
| 3 | piper | default | 1 | 2.934 | 2.934 | -1.112 | 4 | 0 |
| 4 | audio8_realistic | default | 2 | 3.699 | -2.309 | -1.456 | 7 | 0 |
| 5 | audio8_01b | cloning | 2 | 4.149 | 4.149 | -2.651 | 13 | 0 |
| 6 | audio8_realistic | cloning | 2 | 4.267 | -4.267 | -2.777 | 15 | 0 |
| 7 | audio8 | cloning | 2 | 4.986 | 2.180 | -4.257 | 23 | 0 |
| 8 | audio8_01b | default | 2 | 6.040 | -6.040 | -5.303 | 29 | 0 |
| 9 | ming | cloning | 1 | 6.759 | -6.759 | -12.676 | 35 | 0 |
| 10 | lfm3_audio | default | 4 | 7.484 | -3.584 | -2.224 | 11 | 0 |
| 11 | qwentts | default | 2 | 7.694 | -7.694 | -7.699 | 31 | 0 |
| 12 | scyllasband | default | 2 | 8.227 | -8.227 | -4.704 | 28 | 0 |
| 13 | melotts | default | 2 | 9.533 | -9.533 | -2.723 | 14 | 0 |
| 14 | kittentts2 | default | 2 | 9.891 | -9.891 | -0.249 | 2 | 0 |
| 15 | openvoice | cloning | 2 | 10.706 | -10.706 | -3.320 | 18 | 0 |
| 16 | audio8_fast | cloning | 1 | 11.069 | -11.069 | -3.511 | 19 | 0 |
| 17 | pocket | cloning | 1 | 11.264 | -11.264 | 3.321 | 1 | 0 |
| 18 | kittentts2 | cloning | 2 | 11.647 | -11.647 | -0.906 | 3 | 0 |
| 19 | qwentts_fast | cloning | 1 | 11.693 | -11.693 | -8.625 | 34 | 0 |
| 20 | qwentts_06b | default | 2 | 13.804 | -13.804 | -2.593 | 12 | 0 |
| 21 | openvoice | default | 2 | 13.873 | -13.873 | -3.514 | 20 | 0 |
| 22 | audio8 | default | 2 | 13.927 | -13.927 | -1.711 | 9 | 0 |
| 23 | indextts25 | cloning | 4 | 14.619 | -14.619 | -4.378 | 26 | 0 |
| 23 | indextts25 | default | 4 | 14.619 | -14.619 | -4.378 | 26 | 0 |
| 25 | audio8_fast | default | 1 | 15.736 | -15.736 | -1.368 | 6 | 0 |
| 26 | kokoro | default | 2 | 17.709 | -17.709 | -5.524 | 30 | 0 |
| 27 | supertonic | default | 1 | 18.615 | -18.615 | -1.622 | 8 | 0 |
| 28 | qwentts_fast | default | 1 | 18.702 | -18.702 | -12.780 | 36 | 0 |
| 29 | zonos3_mini | cloning | 4 | 18.770 | -18.770 | -4.352 | 25 | 0 |
| 30 | drama3 | cloning | 1 | 21.862 | -21.862 | -7.885 | 32 | 0 |
| 30 | drama3 | default | 1 | 21.862 | -21.862 | -7.885 | 32 | 0 |
| 32 | phonon_40m | default | 1 | 23.639 | -23.639 | -2.857 | 16 | 0 |
| 33 | pocket | default | 1 | 29.920 | -29.920 | -1.915 | 10 | 0 |
| 34 | coqui | default | 2 | 33.082 | -33.082 | -3.047 | 17 | 0 |
| 35 | ming | default | 1 | 69.724 | 69.724 | -3.872 | 21 | 0 |
| 36 | coqui | cloning | 2 | 94.247 | 94.247 | -4.190 | 22 | 0 |

## Dispersion control

Overlap is defined before inspection as overlap between human and synthetic interquartile intervals. Metrics are computed from cleaned voiced frames before time interpolation.

| language | metric | human n | human median [IQR] | synthetic n | synthetic median [IQR] | IQR overlap? |
|---|---|---:|---:|---:|---:|---|
| English (prompts 1-4) | f0_range_st | 11 | 10.616 [6.086, 12.682] | 1558 | 9.039 [7.021, 12.076] | yes |
| English (prompts 1-4) | f0_sd_st | 11 | 3.764 [1.979, 3.787] | 1558 | 2.928 [2.239, 3.757] | yes |
| French (prompt 5) | f0_range_st | 2 | 10.145 [9.621, 10.670] | 203 | 9.090 [7.153, 11.347] | yes |
| French (prompt 5) | f0_sd_st | 2 | 3.245 [3.207, 3.282] | 203 | 2.895 [2.334, 3.494] | yes |

Dispersion-control result: English=reproduced; French=reproduced. This criterion requires IQR overlap for both `f0_range_st` and `f0_sd_st`.

## Duration control

Spearman correlations are utterance-level and descriptive. They do not account for repeated prompts, voices, models, or multiple segments from one human source.

| language | set | n | declination vs duration rho | p | terminal vs duration rho | p |
|---|---|---:|---:|---:|---:|---:|
| English (prompts 1-4) | human | 11 | 0.182 | 0.593 | -0.100 | 0.770 |
| English (prompts 1-4) | synthetic | 1558 | 0.381 | <0.001 | -0.054 | 0.033 |
| French (prompt 5) | human | 2 | NA | NA | NA | NA |
| French (prompt 5) | synthetic | 203 | 0.203 | 0.004 | -0.048 | 0.496 |

## Counts and exclusions

- Published synthetic clips discovered: 1978.
- Joined to `scores.csv` on `(dir, wav)`: 1978; published but unscored: 0.
- Score rows with no published clip: 0 of 1978.
- Failed generations removed before F0 analysis: 59 unique WAVs; health flags=gap=27, head=1, silent=1 (total 29), WER > 0.5=39, both=9.
- Synthetic retained/analyzed: 1919; metric-valid=1829; too few voiced frames=0; analysis errors=0.
- Human source WAV candidates: 64; unique decoded-audio hashes=18; duplicate copies=46; source load errors=0.
- Human segments >= 1.2s analyzed: 123; metric-valid=102; too few voiced frames=0; analysis errors=0.
- Rows with unknown language (excluded from EN/FR comparisons): synthetic=0, human=109.

## Method

Praat autocorrelation F0 uses a broad 50-800 Hz pass to estimate each clip/source voice, followed by an adaptive median/2.5 to median*2.5 floor/ceiling. Unvoiced frames are dropped. Contiguous voiced runs receive a five-frame median filter; frames more than 9 semitones from a 31-frame local median are rejected as octave errors. Each valid contour is converted to semitones relative to its voiced-frame median. Declination is the OLS slope of that contour linearly interpolated to 100 points over normalized time 0-1 (semitones per utterance). Terminal slope is an OLS fit in semitones per second over the real, uninterpolated voiced frames in the last 0.30 s of the voiced span, and is NaN when fewer than 10 such frames exist. Dispersion uses the cleaned, uninterpolated frames.

Human sources are deduplicated by SHA-256 over decoded float32 audio content and segmented with 25 ms energy frames/10 ms hops, an adaptive max(-45 dBFS, p90-30 dB) threshold, and 0.35s minimum silence. Segments shorter than 1.2s are ignored.

## Limitations

- The human baseline is a handful of source recordings. Silence-derived segments from the same clip are correlated and do not create independent speakers.
- Human segments are not text- or duration-matched to the five synthetic prompts. The duration correlations diagnose, but do not remove, this confound.
- Synthetic clips are the published board clips only; the same model can appear from more than one rig/device, and those rows are pooled within a model and mode.
- Per-model medians with small n are unstable. Rows are not statistically independent across devices, modes, prompts, repeated runs, or shared cloning references.
- Terminal slope remains sensitive to endpoint voicing and creak; clips whose final 0.30 s has too little voicing are NaN and drop out of the medians.
- Automatic pitch bounds and median filtering reduce octave errors but cannot guarantee that every tracker error is removed, especially for creaky voice, music, noise, or non-speech artifacts.
- The result is descriptive medians without uncertainty intervals, mixed-effects modeling, or a prompt-matched inferential test. It cannot establish that model architecture causes over-smoothing.
