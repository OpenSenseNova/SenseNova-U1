# Compact authored scene compiler

## Contents

- Designed artifacts
- Requirement tiers
- Authored visual premise
- Whole-image composition
- Density rhythm
- Meaningful creative freedom
- Creative completion and factual restraint
- Knowledge-backed content completion
- Medium selection
- Text necessity gate
- Text island and copy budget
- Structural mode
- Layout completeness
- Single scenes
- Canvas inference
- Negative instructions

## Designed artifacts

Use the fixed core in `core-render-json.md`. Shape its values around the image's
organizing idea and add category-specific members only through `structure`:

```json
{
  "type": "Concrete deliverable category",
  "theme": "Concise subject and emotional intent",
  "content": {
    "title": "Exact literal or empty string",
    "content_sections": [],
    "data_verbatim": []
  },
  "main_subject": "Resolved hero subject or event",
  "layout_composition": "Whole-image organization and reading flow",
  "structure": "Optional exact members or mapped regions"
}
```

Map poster title treatment through `content` and `typography`; hero imagery
through `main_subject`; and panels, nodes, sides, steps, routes, interface
zones, or proposal members through the matching `structure` subtype. Do not add
alternative top-level keys, internal reasoning, or interchangeable style
adjectives.

## Requirement tiers

Treat supplied information according to its role:

- **Locked creative and structural requirements:** exact subject, action,
  non-factual visible copy, count, relation, sequence, fixed placement,
  palette, format, exclusion, and explicit fictional premise.
- **Provisional reality claims:** consequential locations, routes, dates,
  values, rankings, scientific relations, and label bindings. Verify these
  before using them in a reality-based artifact.
- **Open:** viewpoint, crop, pose, object state, generic wardrobe, atmosphere,
  foreground and background, depth, lighting, color allocation, typography
  character, spacing, overlap, rhythm, material response, and finish.
- **Unsupported:** unsupplied factual or commercial claims, real identities,
  brands, statistics, dates, prices, certifications, contacts, or endorsements.

Never weaken locked content for novelty. Never leave all open variables vague
in the name of fidelity.

## Authored visual premise

Internally consider two or three plausible premises, then commit to one without
showing the alternatives. A useful premise changes how the requested elements
form the image:

- a coastline also becomes the route and reading path;
- steam from a product creates the title's quiet backing and upward movement;
- an architectural silhouette determines the information grid;
- the physical process in a science topic determines its connector system;
- the texture or folds of a material control crop rhythm and typography.

A decorative addition does not count as a premise. Random doodles, gradients,
sparkles, floating cards, or a fashionable typeface do not create an authored
idea unless they organize the composition.

Use one strong premise. Several unrelated ideas create an AI collage. Ask
silently whether the premise would remain almost unchanged after swapping in a
different topic; if so, make it more subject-specific.

## Whole-image composition

Describe perceptual goals before listing local parts:

- **first read:** one unmistakable hero, event, title, or silhouette;
- **second read:** one coherent supporting system that explains or extends it;
- **quiet discovery:** restrained tertiary detail, never equal competition;
- **center of gravity:** where the visual mass sits and what counterbalances it;
- **dominant gesture:** an S-curve, diagonal, vertical rise, open horizon,
  centrifugal motion, frame, or another subject-derived movement;
- **scale contrast:** visibly unequal hero, support, and detail sizes;
- **depth and overlap:** foreground, middle ground, and background when the
  medium benefits from spatial experience;
- **negative space:** active quiet area that protects type or mood;
- **tension:** one crop, imbalance, off-axis placement, overlap, or controlled
  grid break when category-appropriate.

Prefer a connected path, frame, rhythm, grid, or depth sequence over unrelated
objects floating in separate boxes. Avoid equal-sized modules unless equality
is semantically required.

## Density rhythm

Within the chosen composition, distribute visual information in three
perceptual scales:

- **dominant:** the clearest large-scale read;
- **supporting:** grouped medium-scale evidence that extends the dominant read;
- **tertiary:** localized fine detail that rewards close viewing without
  competing at thumbnail size.

Do not spread the same module size, edge activity, contrast, or detail density
uniformly across the canvas. Place a quieter area beside a complex cluster so
the eye can alternate between concentration and release. Create rhythm through
grouping, scale contrast, alignment, repetition with controlled variation, and
changes in local spacing. Do not impose a global element cap or uniformly lower
the requested information density.

For information-rich structural artifacts, preserve every required member,
item, and literal. Cluster related content, establish unequal group scales, and
use internal whitespace and repeated visual cadence to make the full system
legible; never create rhythm by deleting content, collapsing members, or
turning all content into smaller equal cards.

For photography and pure scenes, express density through spatial layers and
detail falloff: concentrate the strongest texture, contrast, and specificity
around the subject or event; let supporting context carry medium detail; keep
one atmospheric or low-activity area calm. Do not introduce cards, panels, or
an information grid merely to demonstrate hierarchy.

Treat this as local visual pacing, not a new layout plan. Do not move fixed
regions, change member cardinality, invent copy, or replace the selected
composition.

## Meaningful creative freedom

Lock the main medium, hero event, visual premise, and required structure. Then
let the image model solve optical balance, secondary crops, natural variation,
small overlaps, atmospheric nuance, and connective details.

Good:

```text
Let the coastline determine the exact route curve and secondary sticker
overlaps; keep the cyclist, title, and three named stops unmistakable.
```

Over-constrained:

```text
Place a cloud at 12%, three dots at 18%, and five icons at fixed coordinates.
```

Under-directed:

```text
Make it beautiful, premium, creative, and well composed.
```

Be precise about the intended perception and permissive about secondary
solutions.

## Creative completion and factual restraint

Actively choose non-factual visual details that make the image tangible:
fictional adult people when useful, gestures, generic wardrobe, camera
behavior, lighting, generic props, atmosphere, texture, supporting scenery,
material metaphors, graphic paths, framing devices, and recurring motifs.

Do not add a person by habit. Use human presence when it provides action,
emotion, interaction, or scale. For a named place or culture, use respectful,
recognizable visual context without adding unsupported facts.

Distinguish visual invention, functional content, and source-dependent claims:

- visual invention may be rich;
- functional content may be synthesized from stable general knowledge;
- source-dependent claims must remain tied to an available, reliable source.

## Knowledge-backed content completion

When the artifact's function depends on information, distinguish missing
wording from missing knowledge. Apply the text-necessity gate before generating
visible copy. If visible language is necessary and the topic can be covered
using stable, widely accepted general knowledge, write the content yourself and
make it directly renderable.

Use this for technique summaries, foundational principles, educational
breakdowns, checklists, process steps, comparison dimensions, generic labels,
and numbered frameworks. If the brief asks for an exact count, create exactly
that many distinct, non-overlapping items and arrange them in a sensible
sequence from foundation to application.

For each requested item, prefer:

- one short, concrete heading;
- one concise explanatory sentence;
- one optional action cue or example only when the layout can support it.

Use the language requested or implied by the brief. Output the final strings
character for character beside their placement and typography. Do not output
bracketed placeholders, TBD labels, "insert text here", or instructions for a
later writer when the model already has enough knowledge to complete the
content.

Do not fabricate source-dependent material. A named paper, report, dataset,
private document, branded methodology, quotation, current statistic, legal or
medical claim, or attributed research finding requires the actual source or
reliable retrieval. When a generic theme requests a numbered framework without
citing a source, synthesize a coherent editorial framework from general
knowledge without claiming that it is a canonical or externally authored list.

## Medium selection

Honor an explicit medium without reinterpretation. When the medium is open,
choose it according to the image's most important evidence:

- prefer **photography-led** direction for people physically interacting with
  products; beauty, fashion, food, beverage, hospitality, and lifestyle work;
  or any scene whose appeal depends on believable skin, hands, contact,
  condensation, glass, liquid, food, cosmetics, fabric, or surface response;
- prefer **illustration-led** direction when the user requests illustration,
  when abstraction or impossible transformation is the central experience, or
  when simplified symbolic communication matters more than physical evidence;
- prefer **3D/product-render-led** direction when controlled product geometry,
  impossible staging, cutaways, or pristine material and reflection control are
  central and human realism is not;
- use a **hybrid** only after naming one primary medium. Keep type, diagram,
  collage, print, or drawn overlays subordinate to that base.

Words such as retro, vintage, editorial, collage, screenprint, newspaper, and
poster describe art direction, not automatically the base medium. For a
photography-led result, express them through casting, wardrobe, grooming, set
design, props, lighting, lens character, color grade, typography, framing, and
restrained post-production. Keep faces, bodies, contact, products, and tactile
evidence photographic.

State the reality anchor locally when confusion is likely: a real adult model,
real photographed product, natural skin texture, anatomically credible contact,
optical depth, and physically believable reflections or condensation. Keep
halftone, paper grain, ink spread, misregistration, or collage edges on graphic
fields and typography unless the user explicitly wants the subject itself
illustrated.

## Text necessity gate

Decide whether the image needs visible language before generating any copy.
Separate semantic readability from textual readability.

Visible text is required when at least one condition holds:

- the user supplies an exact string or explicitly requests a title, heading,
  label, annotation, caption, legend, body copy, or written explanation;
- the deliverable is inherently text-bearing, such as an infographic whose
  purpose is to teach named principles, a tutorial with written steps, a table,
  a checklist, a poster title, or a route whose named stops must be read;
- removing language would make the requested information impossible to
  understand, not merely less conventional.

Visible text is not implied by adjectives such as readable, clear, technical,
structured, informative, explanatory, or high-tech. First ask whether the same
meaning can be communicated through color roles, line weight or pattern, shape,
scale, continuity, spatial grouping, cutaway, connectors, sequence, icons, or
material contrast.

Prefer no visible text when visual encoding is sufficient. Typical cases:

- an architectural section with readable structure and MEP systems: distinguish
  systems through fixed colors, pipe and duct geometry, routing continuity,
  cut-plane treatment, and hierarchy;
- a product cutaway with understandable assembly: use exploded order, material
  contrast, and connector geometry;
- a scientific process visualization: use unambiguous stages, arrows, state
  changes, and stable visual identity when names are not requested;
- an ordinary photographic or cinematic scene: add no text unless requested.

When text is unnecessary, omit visible-text fields from the output format. If
the model is likely to hallucinate signage or diagram labels, add one concise
instruction such as "no visible text, letters, numerals, logos, or watermarks."
Do not add a legend whose symbols require written decoding when direct visual
differentiation can do the job.

When text is necessary, generate the minimum functional copy that makes the
artifact work and apply the exact-literal rules below.

## Text island and copy budget

Treat supplied literals and generated functional copy as protected islands
inside a freer composition:

- preserve supplied characters, punctuation, capitalization, symbols, numerals,
  spaces, and meaningful line breaks;
- render generated functional content exactly as written in the enhanced
  prompt;
- specify one appearance per literal unless repetition is requested;
- bind the text to placement, scale, alignment, typography character, color,
  contrast backing, and local whitespace;
- integrate type through crop, alignment, scale, or controlled overlap without
  obscuring glyphs;
- for posters and social covers, add no optional copy by default;
- add at most one short generic supporting phrase when the user invites
  copywriting or the artifact cannot function without it;
- generate functional checklist items when the artifact requires them, but do
  not invent promotional slogans, commercial claims, itinerary promises, or
  tiny metadata merely to simulate sophistication.

When mandatory copy is dense, reduce visual clutter instead of shrinking type.
Fill open space with imagery, shape, material, texture, or negative space.

## Structural mode

For sequences and exact systems, use the category-native `structure` subtype:

- storyboard: `panels`, with one state per panel and identity continuity;
- infographic: `nodes`, `connectors`, `legend`;
- comparison: `left_side`, `right_side`, `shared_scale`;
- tutorial: `steps`, with sequence and state continuity;
- route: named stops with connector direction beside endpoints;
- interface/game screen: stable zones and exact state.

State cardinality once near its members. Give every member a unique role. Do
not describe supporting details as extra panels. Keep the information exact,
then apply one hierarchy, palette logic, spatial rhythm, material finish, and
subject-derived motif across the whole system.

## Layout completeness

Describe what visibly fills a requested container, not merely its name.

Bad:

```text
Lower-right bilingual dimension table with correct units.
```

Better:

```text
Define a four-column dimension schedule—component, width, height, unit—with one
row per supplied component; bind every displayed value and unit to a verified
content entry.
```

Supply renderable density for requested cards, routes, tables, material
swatches, and diagrams. Use placeholders only when the user explicitly asks
for a template. Ask for, generalize, or omit unavailable private or
source-specific values rather than inventing them. Generate concrete functional
content only when stable general knowledge supports it. Do not interpret
completeness as permission to add filler copy to an open poster or cover.

## Single scenes

Fill the fixed Render JSON core compactly:

1. put the decisive subject, action, or object state in `main_subject`;
2. put environment, depth, and spatial relationships in
   `background_environment`;
3. resolve medium, camera, light, palette roles, and material behavior in their
   dedicated fields;
4. keep `negative_prompt` limited to brief-specific failure classes.

Do not add visible text to an ordinary scene unless requested. Do not compress a
rich scene into a caption.

## Canvas inference

Honor every supplied ratio and orientation. If dimensions are absent but the
artifact has a strong category convention, choose a category-native orientation
to guide the composition. Do not invent an exact ratio unless the caller or
rendering system requires one.

## Negative instructions

Protect the brief from likely failures, not from creativity. Keep negatives to
wrong count, missing or garbled required text, anatomy or contact failure,
incorrect material, real logos, specifically rejected styles, or destructive
clutter. Avoid universal defect lists.
