# 09. Figure study: conventions in recent hand-object papers

Ten papers were inspected for figure structure (captions and layout read from the
arXiv HTML/PDF): SHOW3D (CVPR 2026), HOT3D (CVPR 2025), ARCTIC (CVPR 2023), GOLF and
JSSR (HANDS 2026 reports), HOISDF (CVPR 2024), HOPformer/EPIC-Contact (ECCV 2026),
EgoPHI (ECCV 2026), HaMeR (CVPR 2024), and UmeTrack (SIGGRAPH Asia 2022; read
earlier). Observations that transfer to our figures:

## Visual hierarchy and composition
* Teasers are composites of *real frames with overlays* plus at most one schematic
  element (SHOW3D: scene diversity grid; HOPformer: problem/solution split panel;
  HaMeR: 2x4 grid of input + mesh). The message is stated in one sentence in the
  caption. We follow the split-panel idea: real stereo frame -> geometry -> field.
* Architecture figures run left to right with three to four stages, each stage a
  labelled block; data shapes are written on the arrows (ARCTIC "d+3", GOLF
  "35x28", "16x16", "2x21x3"). GOLF adds small real crops inside the pipeline; JSSR
  ends with a real prediction inset as "proof". We adopt: stages = tokens ->
  queries -> heads -> geometric operator -> fusion, with tensor shapes on arrows and
  a real-frame inset at the output.
* Geometry explanations (ARCTIC Fig. 6 heatmap, HOISDF Fig. 3 query points, EgoPHI
  Fig. 3 contact types) use *one* colour dimension for the quantity of interest and
  greyscale for everything else.

## Colour and notation
* Hands: red/orange for left and blue for right is the dominant convention (SHOW3D,
  HOT3D visualisations); objects green; GT vs prediction distinguished by green vs
  red contours (HOT3D) or by column position (HOISDF, HaMeR). We adopt: left hand
  orange, right hand blue, object green wireframe, GT vectors grey, predicted
  vectors coloured by error (single sequential colormap), no more than five hues.
* Contour overlays (HOT3D) read better than filled meshes on monochrome frames; we
  use mesh silhouettes/wireframes on the grayscale Quest 3 frames.

## Qualitative comparison grids
* Columns = methods (with GT first or last), rows = examples; each example shows the
  egocentric crop and a 3D view from a second angle (HaMeR "alternate viewing
  angle", HOT3D five-column input/GT/pred). We use: input crop | baseline | ours |
  GT, plus a 3D side view per row; four to six rows; per-row ADE printed in the corner.
* Failure-case figures are separate and annotated with the failure category
  (SHOW3D Figs. 14-16: interpenetration, missed hand, rotational ambiguity).

## Plots
* Dataset statistics as simple bar charts (SHOW3D Fig. 8) or heatmaps (HOT3D
  orientation heatmaps). Trade-off plots: x = cost, y = error, one marker per
  variant (GOLF Tab. 3 is a table; we render it as a staircase plot for our own
  progression).
* Cross-dataset embeddings (SHOW3D Fig. 4 UMAP) are informative; we do not
  reproduce them, but a transfer bar chart per direction is planned.

## Density and typography
* Captions carry the message and the legend; axes labelled with units (mm).
* Figures are designed at column width (3.3 in) or full width (6.9 in) with 7-8 pt
  sans-serif labels; vector output (PDF) for diagrams, raster for frames.

## Consequences for our figure plan (docs/08)
1. Fig. 1 teaser: three panels on one real stereo frame (inputs with hand/object
   boxes; camera-frame geometry with joints, posed mesh and nearest-surface vectors;
   predicted field overlay), caption states the factorisation idea.
2. Fig. 2 architecture: left-to-right, shapes on arrows, real-frame inset.
3. Fig. 3 field geometry: (a) frame with projected skeleton, mesh silhouette and
   vectors; (b) 3D view; (c) soft assignment weights for one joint at two
   temperatures; (d) symmetry invariance sketch.
4. Fig. 4 qualitative grid; Fig. 5 failure cases by category; Fig. 6 transfer;
   Fig. 7 ablation/trade-off and error-vs-uncertainty curves; Fig. 8 dataset stats.
