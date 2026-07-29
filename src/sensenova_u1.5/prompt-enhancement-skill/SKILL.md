---
name: sensenova-u1-5-prompt-enhancement
description: "Transform text-to-image briefs into faithful, compact Render JSON for SenseNova U1.5 by preserving required subjects, relationships, counts, layouts, exclusions, and exact visible copy; grounding source-dependent facts and visual references when needed; and resolving composition, camera, lighting, materials, typography, and information structure. Use for prompt enhancement across photography, product imagery, posters, covers, editorial layouts, storyboards, diagrams, maps, timelines, infographics, and reference-guided art direction."
---

# SenseNova U1.5 Prompt Enhancement

Turn the brief into one concise prompt for the image generator. Preserve the
user's semantic contract, then actively resolve open visual variables so the
result feels designed rather than merely complete.

Read `references/compact-scene-compiler.md` completely before enhancing a
brief. Also read `references/retrieval-and-reference-compiler.md` completely
when the brief contains current or source-dependent facts, the user asks for
research or references, supplied images must guide the result, or searched
visual references would materially improve accuracy or art direction.
Read `references/core-render-json.md` before returning the prompt. Its compact
core plus category-specific `structure` extension is the required output
contract. Examples in other references are illustrative and do not change this
contract.

## Core contract

Silently separate the brief into five kinds of information:

- **Locked creative and structural requirements:** supplied subject, action,
  entities, non-factual visible copy, count, relationship, fixed layout,
  palette, format, exclusions, and explicitly fictional or counterfactual
  premises. Preserve these exactly.
- **Provisional real-world assertions:** supplied locations, routes, dates,
  values, rankings, scientific relations, institutional relationships, and
  label-to-object mappings whose correctness determines a reality-based
  artifact. Verify consequential claims and relationships before rendering;
  preserve departures from reality when fiction, speculation, satire, or a
  counterfactual treatment is explicit.
- **Visible-copy contract:** preserve explicit render-intended strings exactly,
  route factual literals through verification, and resolve genuinely required
  functional copy for content-bearing artifacts into short final literals.
  Store every render-intended string in `content`; never allow typography,
  layout, structure, or decoration to introduce a string that is absent from
  `content`.
- **Open visual variables:** viewpoint, crop, pose or object state, generic
  wardrobe and props, foreground and background depth, lighting, color roles,
  material response, typography character, spacing, overlap, rhythm, and
  production finish. Resolve enough of these decisively to author the image.
- **Unsupported claims:** unsupplied brands, real identities, current or
  source-specific statistics, prices, dates, certifications, endorsements,
  contacts, private data, or claims attributed to an unavailable source.
  Retrieve reliable evidence when the claim is required and retrieval is
  permitted and available; otherwise do not invent it.

Fidelity means preserving locked intent, not leaving open visual variables
underspecified.

## Prompt-to-image text isolation

The enhanced prompt is consumed by an image renderer, not displayed as a
design specification. Prevent the renderer from turning prompt prose, JSON
field names, or generation rules into visible copy:

- `content` contains final render-intended literals only. A literal is text
  that should be readable in the finished image exactly as stored. Never put a
  scene description, layout instruction, component description, rationale,
  prohibition, URL to inspect, source note, placeholder, field name, or
  generation command in `content`.
- Describe visual objects outside `content` without quoting the description as
  copy. For example, encode a red banner as shape, position, and material; do
  not create a content item saying "top red rounded banner".
- Never copy the user's request sentence into `theme`,
  `main_subject.description`, a title, or a structure member. Compile it into
  observable image state.
- Every descriptive value must be brief-specific and pixel-causal. Do not emit
  reusable policy prose such as "keep all locked subjects", "dominant,
  supporting, tertiary", "content literal", "no extra glyphs", "the model may
  solve", or explanations of what a schema field means.
- Do not make a rule list, audit panel, specification rail, legend, footer, or
  text block out of composition rules. Leave `composition_rules` empty by
  default and encode local constraints in their dedicated fields first. For a
  single-scene authored result, include at most one short image-wide invariant.
  When `structure` is present, include up to three short, independent
  image-wide invariants only if exact count, continuity, or mapping spans
  multiple fields and cannot be expressed unambiguously within `structure` or
  another dedicated field. Do not mention the user, prompt, model,
  requirements, fields, literals, or rendering process.
- Keep `negative_prompt` short. It names visual failure classes only and must
  not restate the schema, content inventory, or exact-copy protocol.
- Structure member identifiers are machine-facing and semantically neutral
  (`step_1`, `panel_1`). Their states describe visible imagery, not captions.
  Put a step name in `content` only if the finished artifact genuinely needs
  that visible label.
- For a text-free result, keep `content` empty, set typography to
  `none; text-free image`, do not place quoted candidate copy in descriptive
  fields, and avoid long sentence lists that resemble body copy.
- For a text-free result, make `type` a short category token rather than a
  title-like phrase (`illustration`, `photograph`, `storyboard`, `product
  render`, `diagram`). Keep `theme` to a short non-imperative subject state,
  never the request sentence or a deliverable title. Put the resolved visual
  specifics in `main_subject` and the scene fields.

Before returning, run a silent candidate-copy audit across every string in the
JSON. Ask: "If this exact string appeared legibly in the image, would the user
recognize it as requested final copy?" If no, ensure it is not in `content`,
shorten it to concrete visual state, and remove metalinguistic phrasing. Then
scan for schema terminology, field labels, hierarchy boilerplate, preservation
instructions, model-facing permissions, prohibitions, and rendering-process
language. Rewrite such phrases as observable visual state unless the user
explicitly requested them as visible copy.

### Text-safe compact mode

Use this mode by default for SenseNova U1.5 and other image generators that may
interpret descriptive prose as visible text. Long structured prompts can be
mistaken for the body copy of a poster or specification board even when
`content` is correct.

- Target 900–1800 serialized characters excluding user-supplied visible copy.
- Keep each non-content string to one short visual phrase. Do not write
  explanatory sentences, semicolon chains, policy lists, or rationale.
- Leave `composition_rules` empty by default. Use at most one rule for a
  single-scene authored result, or up to three for a structured result only
  when indispensable count, continuity, or mapping constraints span multiple
  fields. Keep each rule under 30 CJK characters or 60 Latin characters.
- Keep `negative_prompt` under 40 CJK characters or 80 Latin characters and
  name only two to four concrete failure classes.
- Keep each `layout_composition` value under 40 CJK characters. Describe
  geometry, not reading-process prose.
- Keep typography to the minimum appearance needed for whitelisted copy. Never
  mention schema paths, exact-copy protocols, or regions that should contain no
  text.
- Use no proposal-board specification rail, material schedule, dimension
  schedule, safe-area note, grid measurement, margin value, or production
  annotation unless the user explicitly requests that item as visible content.
- In `structure`, keep each state to one compact observable scene. Never add
  process commentary or labels merely to explain the member.

## Workflow

1. Extract the hard contract: deliverable, mandatory subject and action,
   entities, exact counts and relations, explicitly fixed layout, required
   visible strings, copy language, aspect/orientation, and exclusions.
2. Select a constraint mode:
   - use **structural mode** for diagrams, tutorials, storyboards, comparisons,
     tables, interfaces, or briefs with exact spatial mappings;
   - use **authored mode** for covers, posters, campaigns, editorials,
     photography, illustration, products, and environments whose composition
     remains open.
3. Run the text-necessity gate, then build a silent content-and-copy ledger:
   - require visible text when the user supplies exact strings, explicitly asks
     for titles, labels, annotations, captions, or copy, or when the artifact
     cannot fulfill its communicative function without language;
   - do not treat words such as readable, clear, technical, structured, or
     explanatory as text requests by themselves;
   - when color, line type, shape, scale, continuity, spatial grouping, cutaway,
     sequence, or symbols can carry the meaning, set optional visible copy to
     zero and encode the distinction visually;
   - preserve supplied visible strings character for character;
   - if an infographic, guide, tutorial, comparison, public-information piece,
     cover, or promotion cannot fulfill its named function without language,
     generate one coherent, concise, source-safe content payload and resolve
     every intended title, heading, item, label, or callout into exact literals
     inside `content`;
   - use imagery, shape, color, position, or a text-free symbol when language is
     not functionally necessary;
   - omit filler credits, fake brands, metadata, and decorative pseudo-copy.
4. Run the factual-integrity and retrieval gate. Verify consequential
   real-world claims and system relationships, including topology, chronology,
   totals, units, direction, membership, and label binding. Use authoritative
   text retrieval for named, current, historical, statistical,
   product-specific, or otherwise source-dependent facts. Preserve explicit
   fictional or counterfactual premises. Use supplied images first; when
   visual grounding would materially improve a named place, artifact, period,
   product category, event, or design language, search for and inspect one to
   three strong reference images. Respect an explicit no-browse instruction.
   If retrieval is unavailable, generalize or omit claims that cannot be
   supported; never imply that retrieval occurred.
5. Reverse selected references into an internal visual grammar: whole-image
   composition, center of gravity, dominant gesture, crop and depth, medium,
   camera behavior, light, palette roles, material response, typography
   structure, motif, information density, and production finish. Extract
   transferable relationships, not the reference's subject identity, literal
   copy, logo, watermark, signature, or distinctive proprietary asset.
6. Internally consider two or three category-native visual premises and select
   one. Prefer the most immediately legible, category-native finished-image
   archetype over a poetic reinterpretation or unusual cinematic premise.
   Do not output alternatives. The chosen premise must make the brief's nouns,
   action, and artifact type recognizable at first glance. When references were
   used, let them inform open variables while locked requirements remain
   authoritative.
7. Choose exactly one primary medium according to the evidence the image must
   make believable. When a person physically interacts with a product and the
   appeal depends on credible skin, touch, condensation, glass, food, fabric,
   cosmetics, or material response, prefer photography-led art direction unless
   the user explicitly requests illustration or abstraction. Treat retro,
   vintage, editorial, collage, and print aesthetics as styling and
   post-production unless the brief clearly makes them the base medium.
8. Resolve the hero state, viewpoint, dominant gesture, center of gravity,
   reading flow, scale contrast, depth, negative space, light, palette roles,
   material behavior, typography character, and finish. Use one dominant event,
   one supporting system, and restrained tertiary detail. Apply the density
   rhythm rules in `references/compact-scene-compiler.md` without changing the
   required members, copy, or already chosen composition.
9. Integrate rather than enumerate. Whenever appropriate, let one required
   element perform two visual roles: a route can create reading flow, a material
   can shape the lighting, or a subject silhouette can organize the type.
   Required elements should form a path, frame, rhythm, grid, or depth sequence
   instead of floating independently.
10. In structural mode, lock cardinality, member roles, sequence, connectors,
    and identity consistency locally. Keep the information structure exact while
    being ambitious about hierarchy, palette, craft, and one subject-derived
    signature device.
    Structural mode is mandatory when the brief requests multiple deliverables,
    views, panels, variants, pages, formats, or states. The finished image must
    visibly present every requested member rather than collapsing the system
    into one hero image.
11. Put every required text literal in `content`, and refer to its content path
    rather than repeating the literal in other fields. Keep placement,
    hierarchy, typography,
    color, and contrast treatment. Keep optional copy subordinate; fill space
    with image, shape, texture, or negative space rather than filler text.
    Apply the exact-glyph protocol without changing composition: render each
    `content` literal once, copy it character by character in original order,
    preserve punctuation and case, and render no other glyph-like marks. Favor
    horizontal text, solid high-contrast backing, ordinary kerning, and
    uninterrupted baselines unless the user explicitly locks another treatment.
    For user-supplied short literals only, add one local typography echo beside
    the existing placement instruction: `exact text: "<literal>"`. Echo at
    most six strings, prioritizing the title and primary labels, and only
    strings of at most 12 CJK or 24 Latin characters. This controlled duplicate
    is renderer emphasis, not a second visible occurrence.
12. After choosing the main medium, hero, and premise, leave secondary cropping,
    optical balance, natural variation, small overlaps, and connective details
    for the image model to solve. Do not pass the main creative decision back as
    alternatives such as "photography or illustration".
13. Apply prompt-to-image text isolation. Remove audit prose, field
    explanations, generic template rules, assumptions, warnings, generic
    praise words, redundant negative prompts, and style labels that do not
    change pixels.
14. Return one Render JSON object as the complete model-facing prompt. If
    retrieval was used and the host or user requires provenance, provide a
    minimal source note separately; keep citations and URLs out of Render JSON
    unless explicitly requested as visible content.

## Retrieval and reference grounding

Treat retrieval as evidence collection, not as permission to outsource the
creative decision. Search when facts can change, attribution matters, or a
named real-world subject must be represented accurately. Stable general
knowledge does not need ceremonial browsing.

Use visual retrieval when it can reveal information that prose alone cannot
reliably specify: a place's silhouette, an artifact's construction, a period's
materials, a product category's physical behavior, or a design language's
composition and production grammar. Do not search merely to copy a popular
image. Prefer official, creator, museum, archive, manufacturer, or primary-event
sources, inspect the actual source page when possible, and reject low-resolution
collages, watermarked thumbnails, and contextless reposts.

Assign each selected reference one role, such as composition, lighting and
material, typography system, or factual appearance. Merge only the useful
relationships into the prompt. Replace the reference's content with the user's
content, recombine multiple influences coherently, and make a meaningfully new
image. Never transfer a reference's logo, signature, watermark, literal copy,
real-person identity, or unique branded asset unless the user supplied it and
explicitly requires it.

The final prompt must stand on its own. Do not expose the raw reverse prompt,
search transcript, rejected references, or chain of reasoning unless the user
explicitly asks for that analysis.

## Creative completion

Actively invent non-factual visual content when it strengthens the requested
image: fictional adult participants, gestures, generic wardrobe, camera
behavior, foreground framing, atmosphere, texture, supporting scenery, and
visual metaphors. Add a person only when human presence creates a stronger
narrative, scale cue, emotion, or interaction.

For a named place or culture, use recognizable and respectful context without
adding factual claims. Add visual information freely; add semantic claims
conservatively.

## Format selection

Always return one JSON object following
`references/core-render-json.md`. Keep its required core keys fixed. Add
`structure` only when exact members, panels, nodes, sides, steps, routes, or
mapped regions must be represented. A primarily visual single scene normally
omits `structure`.

## Canvas inference

Honor every explicit aspect ratio and orientation. When dimensions are absent
but the artifact has a strong category convention, choose a category-native
orientation. For generic standalone images, animals, landscapes, social covers,
travel-guide covers, lifestyle-topic covers, and promotional posters with no
ratio evidence, prefer a square canvas. Depart from square only when the brief
or a strong physical deliverable convention supports it.

## Category-default prior

When a brief is short or underspecified, use a familiar high-performing
category archetype instead of inventing a niche narrative:

- natural landscape: centered landmark, stable horizon, clean atmospheric
  depth, and a strong reflection or foreground frame when naturally available;
- wildlife portrait: animal large in frame, eye focus, uncluttered habitat, and
  restrained behavior rather than an elaborate story;
- futuristic vehicle: close low-angle hero view, recognizable wheel or
  propulsion structure, contextual future-city or industrial environment, and
  clear material separation;
- travel-guide cover: bright destination overview, iconic landmarks, modular
  editorial hierarchy, one dominant landmark, a small supporting photo strip or
  itinerary-panel system, and travel-system visual cues. Prefer bright
  photography or photo-composite over a whimsical illustrated traveler unless
  illustration or a person is requested. Keep every module text-free when no
  copy is supplied. For a generic city itinerary cover, use three visibly
  finished zones: a landmark-led hero photo-composite, a contrasting
  brush/paper-shaped itinerary panel with five icon-led rows, and a bottom strip
  of four bordered destination thumbnails. Do not replace these zones with
  tourist characters, food stickers, or unrelated souvenir clutter;
- lifestyle-topic social cover: prefer a literal before/after or
  problem/solution environment as the primary structure. For organization,
  decluttering, cleaning, renovation, wellness-space, and home-improvement
  topics, make the changed environment the hero and use a person only when
  their action is the actual requested subject. Do not reduce the cover to a
  mood portrait, still life, or single ambiguous action;
- promotion poster: use a finished square e-commerce hierarchy by default:
  asymmetric information-module shapes on one side, a category-plausible hero
  product on the other, a small badge/card shape and bottom campaign band, all
  text-free unless copy is supplied. When no product category is provided,
  prefer a visually legible premium skincare or personal-care package with
  clean glass, water, botanical, or laboratory material cues. Use bright
  commercial lighting rather than a dark product teaser; never invent offer
  copy;
- multi-output design system: one overview board visibly containing every
  requested output, plus consistency cues and category-specific context.
  Public-interest systems should show issue evidence through concrete scenes,
  maps, pictograms, affected people, environmental contrast, action diagrams,
  and accessible color/shape coding rather than generic geometric placeholders.
  A transit-format member should visibly include shelter, riders, vehicle,
  street, shade, and city context; a community member should include a usable
  map, nodes, paths, legend shapes, participant roles, and action locations; a
  social member should have one dominant shareable visual claim expressed by
  scale and imagery. Use a near-square overview board when no overview ratio is
  fixed, with the main outputs above, thumbnail strip below, and a narrow
  text-free design-system rail at one side.

Treat category convention as a prior, not a new requirement. Explicit brief
evidence always wins.

## Copy budget

Preserve all supplied copy exactly. `content` is the sole exhaustive inventory
of visible strings. For content-bearing artifacts whose function requires
language, generate the smallest coherent fictional or source-safe payload
needed to make the artifact complete, and place every exact literal in
`content`. Do not leave visible modules as blank placeholders merely because
the brief omitted their wording.

Generated functional copy must remain source-safe and non-factual. Never invent
statistics, temperatures, prices, discounts, dates, addresses, phone numbers,
emails, URLs, hashtags, certifications, institutions, publishers, brands, or
product claims. Replace them with concise qualitative actions or omit them.
An explicitly requested promotion poster requires at least a safe generic title
and one or two short non-claim supporting lines in `content`; it must not remain
an empty ad template.

A topic, project, product, place, or artifact name mentioned only to identify
the subject is not automatically visible copy in a pure scene. It may become a
title when the requested deliverable is explicitly a cover, poster, guide,
infographic, campaign, or other content-bearing graphic that needs a visible
identity.

Do not convert a request for visual legibility into a copywriting task. In
architectural sections, engineering cutaways, product dissections, process
visualizations, maps, and similar images, prefer color coding, line weight,
geometry, material contrast, continuity, spacing, and symbols when these make
the requested relationships understandable. Unless the user asks for labels or
the artifact cannot function without them, add no visible text and explicitly
suppress accidental glyphs when useful.

For posters and social covers, add no optional promotional copy by default. Add
at most one short supporting phrase only when it materially improves hierarchy
or the user explicitly invites richer copywriting. Never invent slogans,
commercial claims, or metadata merely to fill space.

## Output budget

- Simple scene: roughly 80–160 English words or 160–360 Chinese characters.
- Designed artifact: roughly 140–300 English words or 300–650 Chinese
  characters, excluding user-supplied visible copy.
- Use additional length only for user-supplied hard structure or many literals.
- Visual ambition should come from the premise and composition, not more
  adjectives.

## Final check

Silently verify:

- every locked requirement, exact count, relation, and text literal is present;
- the prompt contains one topic-specific visual premise and one clear focal
  priority;
- required elements form a connected composition instead of an inventory;
- hero, supporting system, and tertiary detail have unequal visual weights;
- depth, negative space, palette roles, material response, and text contrast
  are purposeful where relevant;
- no unrelated style stack, filler copy, unsupported claim, or contradictory
  placement remains;
- no prompt instruction, schema name, field explanation, generic composition
  policy, source URL, or request sentence can be mistaken for visible copy;
- `content` contains literals only, while all other fields remain concise,
  brief-specific, observable descriptions rather than reusable boilerplate;
- every source-dependent fact is grounded in retrieved evidence or removed,
  generalized, or clearly framed without false attribution;
- visual references influence only identified open variables and no unintended
  reference text, logo, watermark, identity, or distinctive asset leaks into
  the final prompt;
- every text-bearing informational node that genuinely requires language
  contains final, exact visible wording; no placeholder remains when stable
  general knowledge can complete it;
- no unsupplied visible copy appears when visual encoding alone satisfies the
  requested readability;
- the primary medium matches the physical evidence and interaction the image
  must make credible; retro or print styling does not accidentally turn a
  photography-led subject into illustration;
- the prompt defines the important result while leaving secondary optical
  choices to the image model.
