# Shape-Informed Medical Image Registration — Dissertation

## Todo

### Must-have (content)
- [ ] Write Introduction (~7–9p):
  - Motivation (~2p): clinical importance of registration → intensity ≠ anatomy → shape as stable modality-agnostic property. Open with D'Arcy Thompson hook (theory of transformations as conceptual anchor for deformation-based registration); optionally Gestalt — group convention is to put cross-disciplinary hooks in §1.1, not in a preface (cf. Christian's Ch 1 opening).
  - **Define "shape" broadly here**: dense (segmentations, SDMs/EDMs), sparse (point clouds, landmarks), parametric/implicit (meshes, INRs), statistical (SSMs, atlases). The umbrella has to honestly cover Ch IV (segmentation-driven) and Ch V (point-cloud-based) alongside the explicit shape-model chapters — naming the breadth up front avoids overclaiming a shape *prior* where there is only a structural representation.
  - Two themes (~0.5p): shape representation learning (Ch I–III) and shape-informed registration (Ch IV–V)
  - Objectives (~1.5p): bullet-point research questions per chapter
  - Organization & Contributions (~3p): overview figure (two-column flow: representation → registration) + one paragraph per chapter with citation
  - Central research question: "How can anatomical shape information be represented and leveraged to improve medical image registration?"
- [ ] Write Background (~17–18p) — **literature only, no own results** (group convention from Lasse/Christian; own synthesis lives in Summary, post-publication literature lives in 7.4 Outlook). Soft rule for "new" literature: published after the chapter's underlying paper, or not cited here — judgment call. No umbrella restatement here (kept in Intro §1.1). **Open each subsection with a forward-pointing sentence** ("Chapter X uses Y, therefore this section reviews ...") — Lasse's pattern; justifies every block by what it sets up.
  - 2.1 Medical Image Registration (~5–6p) — **learning-based focus**: open with a brief classical primer (~0.5–1p: energy-minimisation framing, similarity + regularisation, just enough to anchor convexAdam's discrete-optimisation roots), then learning-based registration (supervised/unsupervised, instance optimisation), multi-modal similarity in deep registration (NMI, MIND, semantic features), evaluation (TRE, DSC, SDLogJ, HD95, Learn2Reg). Threading toward §2.2: the multi-modal / semantic-features paragraph is the bridge — "intensity-based similarity fails across modalities ⇒ structural/semantic features as the modality-agnostic bridge", which opens directly into §2.2.
  - 2.2 Shape Representations in Medical Image Analysis (~12–13p):
    - Opening (~1p): modality diversity (CT/MRI/US) as motivation — shape as stable, modality-agnostic bridge. (D'Arcy Thompson / Gestalt hook moved to Intro §1.1; Background opens directly with the technical motivation.)
    - Taxonomy: distinguish two levels explicitly:
      - Shape descriptors (geometric primitives for a single instance) — compare in a table (axes: memory scaling, precision/resolution control, differentiability, topology flexibility):
          - Dense/volumetric: masks, SDMs/EDMs, occupancy functions
          - Parametric surfaces: meshes + B-spline surfaces (bundled — same idea, different basis)
          - Sparse: point clouds, landmarks (distinct from point clouds: landmarks carry inter-instance correspondence)
          - Implicit: zero-level sets
          - Skeletal: medial axis
      - Shape models (frameworks that capture variation across instances/populations): SSMs (PCA-based), atlases (population templates + deformations), INRs as continuous parameterization that can realise both (Ch II = INR-SSM, Ch III = neural atlas) — natural progression SSM → atlas → INR-based; discussed in prose, not table
      - **Flag INR's dual role explicitly**: an INR is a *descriptor* when it parameterises a single instance and a *model* when it parameterises a population. The two appearances in this section are deliberate, not redundant — call this out in prose so the reader doesn't get confused by the second mention.
- [ ] Write Summary/Discussion (~16p) — **own contributions only**, structured to mirror the Introduction's objectives (group convention); new / post-publication literature appears only in 7.4 Outlook:
  - 7.1 Contributions (~4p): one paragraph per chapter, technical, key results; consider "representation type / task" margin tags
  - 7.2 Research Findings (~5p): synthesize across chapters around two themes — (a) shape representation (EDM→SDM→INR progression, Ch I–III, INR thread); (b) shape-informed registration (learned vs. hand-crafted features, optimization design space, Ch IV–V)
  - 7.3 Limitations (~2p): pathological anatomy, INR training cost, convexAdam's implicit shape reliance, HeatReg reference requirement, **transferability / external validity** (which anatomies and datasets were tested vs. claimed), **reproducibility** (code-release status per chapter)
  - 7.4 Research Outlook (~3p): methodological future directions (INR-based joint registration, foundation segmentation models, atlas priors); one paragraph on clinical translation horizon
- [ ] Write Abstract (EN, ~1p) and Zusammenfassung (DE, ~1p)
- [ ] Write chapter framing paragraphs for each paper chapter (I–V) — **5–15 lines / ≤ half a page**, before the numbered Introduction. Group convention (Christian's template, transferable):
  > "This [first/second/third/fourth/last] methodological chapter ... ⟨one-sentence topic⟩. ⟨one-sentence what is done⟩. The method was published in [Cite]. Open source code was released under: https://github.com/..."
  - Required per chapter: (a) topic + connection to thesis narrative (representation arc Ch I–III, or shape-informed-registration arc Ch IV–V), (b) what the chapter does in one sentence, (c) publication citation, (d) code repo link if public. Group convention does *not* include a per-chapter contribution statement; co-first authorship is flagged via `*` in the publication list only.
  - Ch IV (convexAdam) extras: acknowledge that an earlier version appeared in Hanna's dissertation (unpublished at the time); state what the published version adds — complete redesign of hyperparameter search (rule-based → fully data-driven random sampling), runtime reduction (24h → ~2h), AbdomenMRCT fix via TotalSegmentator, added ablations. Link the Learn2Reg Challenge (co-organised) and Open Science Award 2023 (with M. Heinrich, L. Hansen, A. Hering): https://research.uni-luebeck.de/de/prizes/open-science-award-2023/. **Honest shape-link framing**: convexAdam uses segmentation-derived features (nnU-Net primary; TotalSegmentator/AMOS22 supplementing) as a structure-aware alternative to intensity-based similarity; the hyperparameter search also tests MIND features — segmentations win where labels exist, not universally. Cross-reference the broad "shape" definition from Intro §1.1 so the umbrella is honestly earned.
  - Ch V (IJCARS) extras: framing should make the image → segmentation (nnU-Net) → surface point cloud pipeline explicit, so the "shape-informed" umbrella is visibly justified.

### Structural decisions
- [ ] Decide on WBIR (SINA) chapter ordering — recommendation: keep as Ch III (bridge between shape representation and registration; moving it to end breaks the two-part narrative)

### Quick fixes
- [ ] Fix author name macros in `defines.tex` (two placeholders: `\boldname` and `\shared`)
- [ ] LaTeX package harmonization — remove duplicates from chapter `includes.tex` files, consolidate to root
- [ ] Add Limitations section to `sections/7_summary.tex`

### Nice-to-have
- [ ] Figure harmonization (consistent fonts, colors, framing across chapters)
