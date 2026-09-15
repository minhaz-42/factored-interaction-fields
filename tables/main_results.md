| run | model | views | fold | ADE | L | R | acc@10 | acc@50 | acc@100 | recall | SD (mm) | params (M) | train h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1_interfield_A | InterField ResNet-50 (official) | h0 | A | 60.47 | 60.27 | 60.68 | 0.150 | 0.682 | 0.861 | 1.000 | - | 24.9 | official |
| B2_direct_mono_A | direct/precision/rays=False | h0 | A | 49.58 | 47.71 | 51.45 | 0.213 | 0.742 | 0.898 | 1.000 | 35.75 | 14.9 | 1.1 |
| B3_direct_stereo_rays_A | direct/precision/rays=True | h0+h1 | A | 46.15 | 45.76 | 46.55 | 0.227 | 0.756 | 0.912 | 1.000 | 33.22 | 15.1 | 2.2 |
| B3_direct_stereo_rays_B | direct/precision/rays=True | h0+h1 | B | 35.68 | 39.77 | 31.59 | 0.255 | 0.794 | 0.920 | 1.000 | 23.61 | 15.1 | 2.1 |
| B3_direct_stereo_rays_s1_A | direct/precision/rays=True | h0+h1 | A | 45.73 | 44.46 | 47.01 | 0.222 | 0.776 | 0.918 | 1.000 | 33.40 | 15.1 | 1.8 |
| B3b_direct_stereo_norays_A | direct/precision/rays=False | h0+h1 | A | 47.47 | 46.53 | 48.41 | 0.205 | 0.754 | 0.908 | 1.000 | 34.78 | 14.9 | 1.7 |
| FIF_geo_stereo_A | geo/precision/rays=True | h0+h1 | A | 55.11 | 55.31 | 54.90 | 0.097 | 0.689 | 0.896 | 1.000 | 44.11 | 15.1 | 1.9 |
| FIF_hybrid_gatefusion_A | hybrid/gate/rays=True | h0+h1 | A | 44.49 | 43.55 | 45.42 | 0.211 | 0.782 | 0.924 | 1.000 | 32.32 | 15.4 | 2.3 |
| FIF_hybrid_meanfusion_A | hybrid/mean/rays=True | h0+h1 | A | 48.38 | 47.29 | 49.47 | 0.173 | 0.747 | 0.911 | 1.000 | 32.48 | 15.1 | 2.4 |
| FIF_hybrid_mono_A | hybrid/precision/rays=True | h0 | A | 45.19 | 41.69 | 48.69 | 0.219 | 0.783 | 0.918 | 1.000 | 33.32 | 15.1 | 1.3 |
| FIF_hybrid_noaux_A | hybrid/precision/rays=True | h0+h1 | A | 45.71 | 45.30 | 46.12 | 0.207 | 0.774 | 0.920 | 1.000 | 34.09 | 15.1 | 2.2 |
| FIF_hybrid_nonll_A | hybrid/mean/rays=True | h0+h1 | A | 46.48 | 44.85 | 48.11 | 0.169 | 0.757 | 0.921 | 1.000 | 32.41 | 15.1 | 2.3 |
| FIF_hybrid_stereo_A | hybrid/precision/rays=True | h0+h1 | A | 44.43 | 43.95 | 44.90 | 0.211 | 0.768 | 0.921 | 1.000 | 31.19 | 15.1 | 2.4 |
| FIF_hybrid_stereo_B | hybrid/precision/rays=True | h0+h1 | B | 34.08 | 37.44 | 30.71 | 0.252 | 0.807 | 0.930 | 1.000 | 22.75 | 15.1 | 2.3 |
| FIF_hybrid_stereo_s1_A | hybrid/precision/rays=True | h0+h1 | A | 43.82 | 43.12 | 44.53 | 0.221 | 0.787 | 0.927 | 1.000 | 31.87 | 15.1 | 2.4 |
| T2_B3_hot3d | direct/precision/rays=True | h0+h1 | A | 42.63 | 39.83 | 45.42 | 0.087 | 0.750 | 0.891 | 1.000 | 37.36 | 15.1 | 0.1 |
| T2_FIF_hot3d | hybrid/precision/rays=True | h0+h1 | A | 43.32 | 40.00 | 46.64 | 0.082 | 0.748 | 0.894 | 1.000 | 36.44 | 15.1 | 0.1 |
| T3_B3_joint | direct/precision/rays=True | h0+h1 | A | 46.20 | 45.97 | 46.42 | 0.213 | 0.760 | 0.911 | 1.000 | 32.51 | 15.1 | 2.3 |
| T3_FIF_joint | hybrid/precision/rays=True | h0+h1 | A | 46.16 | 45.54 | 46.78 | 0.206 | 0.766 | 0.921 | 1.000 | 32.71 | 15.1 | 2.5 |
