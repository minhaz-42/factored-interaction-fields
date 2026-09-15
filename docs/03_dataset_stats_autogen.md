# 03b. Dataset statistics (auto-generated from derived labels at 10 fps, training subjects)

## Validity funnel

| stage | frames |
|---|---|
| frames_sampled | 138,579 |
| frames_tracking_valid | 126,321 |
| frames_object_conf>0 | 103,340 |
| frames_object_conf>0.5 | 100,010 |
| frames_left_conf>0.5 | 95,340 |
| frames_right_conf>0.5 | 110,106 |
| frames_labelled | 87,366 |
| frames_both_hands | 60,176 |
| fields (left / right) | 68,270 / 79,272 |
| recordings | 468 |

## Labelled frames per subject

| subject | frames |
|---|---|
| ASC023 | 5,080 |
| LWA828 | 13,341 |
| LYA722 | 6,027 |
| MHA016 | 8,904 |
| MMO925 | 8,183 |
| PCW023 | 4,490 |
| SPI102 | 16,562 |
| XXI103 | 8,537 |
| XYZ109 | 5,921 |
| YZH016 | 10,321 |

## Labelled frames per object

| object | frames | median magnitude (mm) | mean | frac < 15 mm |
|---|---|---|---|---|
| birdhousetoy | 8,489 | 31.3 | 61.0 | 0.25 |
| dinotoy | 8,435 | 26.3 | 68.9 | 0.32 |
| milk | 8,330 | 29.9 | 68.5 | 0.30 |
| keyboard | 8,034 | 35.3 | 84.7 | 0.20 |
| orangejuice | 6,818 | 29.9 | 80.7 | 0.31 |
| vase | 6,572 | 26.4 | 84.3 | 0.33 |
| brushholder | 6,183 | 41.2 | 109.1 | 0.24 |
| dumbbell | 5,639 | 29.0 | 135.9 | 0.34 |
| mug | 5,328 | 31.1 | 80.2 | 0.30 |
| aria | 4,357 | 36.2 | 85.2 | 0.23 |
| mustard | 3,976 | 27.4 | 100.7 | 0.32 |
| balandabowl | 3,608 | 24.5 | 61.7 | 0.35 |
| ranch | 3,196 | 29.8 | 66.0 | 0.30 |
| bbq | 3,082 | 25.1 | 52.8 | 0.33 |
| waffles | 1,062 | 29.1 | 52.0 | 0.29 |
| canparmesan | 1,001 | 22.9 | 66.8 | 0.36 |
| cantomatosauce | 968 | 22.4 | 68.7 | 0.37 |
| cansoup | 668 | 26.1 | 77.0 | 0.33 |
| vegetables | 627 | 31.3 | 70.3 | 0.28 |
| mouse | 535 | 101.0 | 193.6 | 0.14 |
| mug2 | 458 | 58.1 | 162.7 | 0.19 |

## Field magnitude (mm)

| group | mean | median | p90 | frac < 10 | frac < 15 | frac > 60 | n |
|---|---|---|---|---|---|---|---|
| all | 82.8 | 30.6 | 243.6 | 0.186 | 0.289 | 0.302 | 3,098,382 |
| fingertips | 70.6 | 13.8 | 233.8 | 0.401 | 0.523 | 0.232 | 737,710 |
| wrist | 135.4 | 96.5 | 285.2 | 0.000 | 0.000 | 0.870 | 147,542 |
| palm | 85.7 | 37.4 | 248.8 | 0.115 | 0.194 | 0.294 | 147,542 |
| thumb | 78.9 | 30.7 | 232.0 | 0.127 | 0.255 | 0.269 | 295,084 |
| proximal | 95.4 | 47.6 | 253.8 | 0.011 | 0.038 | 0.367 | 590,168 |
| intermediate | 82.4 | 29.6 | 243.0 | 0.077 | 0.196 | 0.266 | 590,168 |
| distal | 73.8 | 17.4 | 238.5 | 0.298 | 0.452 | 0.238 | 590,168 |

## Visibility and observability (view 0)

* joints visible (inside image and not occluded by the object): 0.760; inside image: 0.951
* magnitude of visible joints: median 39.8 mm; occluded/outside: median 11.8 mm
* object field-of-view fraction (view 0): mean 0.968; in view (>=0.9) 0.939; partial 0.048; out (<0.1) 0.013; view 1 out: 0.013
* hand field-of-view fraction (view 0): mean 0.951; in view 0.931; out 0.030

## Stereo geometry

* baseline: mean 63.9 mm (median 63.8)
* joint depth (camera 0): median 308 mm, p90 441 mm
* fx = 454.5 px; median disparity 94 px; median depth per pixel of disparity 3.26 mm
