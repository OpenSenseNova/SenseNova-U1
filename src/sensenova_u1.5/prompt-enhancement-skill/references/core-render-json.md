# Render JSON Output Contract

Return one compact JSON object with exactly these required top-level keys:

```json
{
  "type": "",
  "theme": "",
  "language": "",
  "canvas": {
    "aspect_ratio": "",
    "orientation": ""
  },
  "content": {
    "title": "",
    "content_sections": [],
    "data_verbatim": []
  },
  "main_subject": {
    "description": "",
    "action": "",
    "position": "",
    "supporting_elements": []
  },
  "background_environment": {
    "setting": "",
    "spatial_layers": "",
    "environmental_details": []
  },
  "style": {
    "overall": "",
    "medium": "",
    "color_palette": []
  },
  "lighting": {
    "source": "",
    "quality": "",
    "contrast": ""
  },
  "camera": {
    "shot_type": "",
    "viewpoint": "",
    "lens": "",
    "framing": ""
  },
  "typography": {
    "hierarchy": "",
    "integration": ""
  },
  "layout_composition": {
    "overall": "",
    "visual_hierarchy": "",
    "visual_flow": "",
    "negative_space": ""
  },
  "composition_rules": [],
  "negative_prompt": ""
}
```

Add the optional top-level `structure` object only when the artifact has exact
members, panels, nodes, sides, steps, routes, states, or spatial mappings.
Never add other top-level keys.

## Core intent

Describe a finished image directly. Prefer observable subjects, actions,
spatial relationships, camera behavior, light, materials, and layout over
design rationale or audit prose. Match the density and renderer-facing
specificity of a strong image-derived description.

Write all descriptive values in the brief's primary language. Keep exact
visible-copy literals in their original language. Do not default to English
merely because this instruction file is English.

### `type`, `theme`, and `language`

- `type`: the concrete deliverable and category, including orientation only
  when useful.
- `theme`: one concise subject-and-emotion statement.
- `language`: the language of the allowed visible copy, or `none`.

For `language: none`, `type` must be a short category token such as
`illustration`, `photograph`, `storyboard`, `product render`, or `diagram`, not
a natural-language title. Keep `theme` short, non-imperative, and visibly
unlike a headline; never copy the request sentence. This reduces the risk that
text-sensitive image generators treat the first descriptive values as title
copy.

### `canvas`

- Preserve an explicit whole-output ratio.
- When only internal panel ratios are supplied, infer a category-native overall
  canvas rather than treating a panel ratio as the whole image.
- `orientation` must be `landscape`, `portrait`, or `square` and agree with
  `aspect_ratio`.
- Put background and safe-margin behavior into the relevant environment or
  layout field only when it changes the image.

### `content`: the sole visible-copy contract

`content` is a closed whitelist for every glyph intended to appear:

```json
{
  "title": "exact user-supplied title or empty string",
  "content_sections": [
    {
      "heading": "exact user-supplied visible heading",
      "items": ["exact user-supplied visible item"]
    }
  ],
  "data_verbatim": ["exact user-supplied standalone data literal"]
}
```

Include explicit non-factual render-intended strings exactly. For a
reality-based artifact, verify consequential factual literals and relationships
before freezing their final wording. For a content-bearing artifact that cannot
fulfill its named function without language, generate the smallest coherent set
of source-safe functional copy and resolve it into exact final literals here
before rendering. This may include a title, short section headings, concise
items, labels, or callouts. Never generate filler credits, fake brands,
unsupported factual claims, decorative pseudo-copy, or metadata.

Generated copy must not invent statistics, temperatures, prices, discounts,
dates, addresses, phone numbers, emails, URLs, hashtags, certifications,
institutions, publishers, brands, or product claims. Use qualitative,
non-factual actions and category language instead. A promotion poster may not
remain an empty template: give it a safe generic launch title and one or two
short non-claim supporting lines in `content`.

A word or phrase mentioned only as a theme is not automatically visible in a
pure scene, but it may become the visible title of an explicitly requested
cover, guide, poster, infographic, campaign, or other content-bearing graphic.

Store each literal once. Preserve spelling, capitalization, punctuation, and
spacing character for character. When no text is supplied, keep the title
empty and both arrays empty; the image must then be text-free.

Treat each stored literal as an indivisible glyph sequence. The renderer must
copy it once from first character to last, without translation, paraphrase,
abbreviation, insertion, deletion, duplication, character substitution, or
line-break insertion inside a short phrase. This is a rendering constraint,
not permission to change the layout or add more text.

No other field may introduce candidate visible copy. Other fields may reference
`content.title`, a content-section role, or a data role, but must not repeat its
literal. Do not request unlitelisted ticks, numbers, technical notes, captions,
panel labels, badges, legends, footer metadata, or placeholder-like marks in
layout, typography, composition, camera, environment, or structure.

`content` values must pass the literal test: the exact string is intended to
be read by the viewer in the final image. Never store descriptions of regions,
objects, layout, styling, placement, behavior, or generation instructions in
`content`. In particular, reject values such as "top red rounded banner",
"centered ring chart", "product placed in a real usage scene", "use bold
sans-serif type", or "keep the hierarchy clear". Move their visual meaning to
the appropriate subject, environment, style, typography, or layout field.

### `main_subject`

Resolve the primary subject as a visible entity or scene:

- concrete identity and form;
- category-native appearance and materials;
- one decisive action or state;
- position and relative scale;
- a short list of subordinate, non-text supporting elements.

For underspecified single-scene campaigns about human behavior, emotion, habit,
relationship, or transformation, prefer one credible person performing a
decisive action over an inert symbolic still life. For lifestyle covers about
organization, cleaning, renovation, or spatial change, prefer a literal
before/after or problem/solution environment; do not let an invented person
replace the environmental transformation.

### `background_environment`, `style`, `lighting`, and `camera`

Write pixel-causal descriptions:

- organize foreground, middle ground, and background;
- choose one primary medium and one coherent style family;
- assign colors roles rather than listing fashionable swatches;
- specify motivated light source, quality, contrast, and atmosphere;
- specify shot scale, viewpoint, lens behavior, perspective, focus, crop, and
  depth only as appropriate to the medium.

Do not repeat the same scene description across these groups.

### `typography`

Describe only how the whitelisted `content` is rendered and integrated:
type character, relative hierarchy, placement behavior, backing, contrast, and
legibility. Reference content roles rather than repeating literals.

If `content` is empty, set all typography values to `none; text-free image`.
Typography must never create additional copy.

If `content` is non-empty, typography may name only content paths such as
`content.title` or `content.content_sections[0].heading`. It must explicitly
make all other graphic regions text-free.

One narrow exception is allowed for renderer emphasis: append up to six
user-supplied short literals to `typography.integration` as
`exact text: "<literal>"`, prioritizing the title and primary labels. Only echo
literals up to 12 CJK characters or 24 Latin characters. The echo specifies
the same single visible occurrence and must not alter its region, scale, or
layout. Do not echo generated copy or long body text.

Use normal horizontal baselines, non-condensed spacing, crisp high-contrast
letterforms, and a quiet solid backing by default. Override these only when the
brief explicitly locks vertical, curved, distorted, or overlapping type. Do
not move or resize an already locked text region merely to satisfy this clause.

Keep typography phrasing visual and concise. Do not emit schema tutorials such
as "content.title is the largest level", "one occurrence per content literal",
or "all other regions contain no text"; text-sensitive image generators may
typeset that prose.
Use content paths plus appearance, for example
`content.title: largest white bold type on a quiet purple backing`.

### `layout_composition` and `composition_rules`

Describe the resolved whole-image organization:

- largest masses and region proportions;
- first, second, and third visual reads;
- eye path or dominant gesture;
- protected negative space;
- a small set of concrete compositional invariants.

Do not turn this into a second inventory of subjects or style.

`composition_rules` is not a policy dump. Encode local constraints in
`structure`, `main_subject`, or `layout_composition` first, and leave this array
empty by default. For a single-scene authored result, use at most one short
image-wide invariant. When `structure` is present, use up to three short,
independent image-wide invariants only if exact count, continuity, or mapping
spans multiple fields and cannot be expressed unambiguously in a dedicated
field. Keep each rule under 30 CJK characters or 60 Latin characters. Prohibit
generic reusable sentences, field names, renderer instructions, and terms such
as `dominant/supporting/tertiary`, `content literal`, "keep all locked members",
or "the model may solve". If a rule would fit an unrelated brief unchanged,
delete it.

### `negative_prompt`

The negative prompt should cover only likely damaging visual failures and stay
under roughly 80 Latin or 40 CJK characters. For text-free images, use a compact phrase such
as `visible letters, numerals, logos, watermarks`. For text-bearing images,
use a compact phrase such as `extra copy, pseudo-text, misspelled required
copy, logos, watermarks`. Do not paste the copy contract, schema explanation,
or long exact-glyph protocol here.

## `structure` extension

Use one category-native object when exact structure matters:

- infographic: `{"type":"infographic","nodes":[],"connectors":[],"legend":[]}`
- storyboard: `{"type":"storyboard","panels":[],"continuity":[]}`
- comparison: `{"type":"comparison","left_side":{},"right_side":{},"shared_scale":""}`
- tutorial: `{"type":"tutorial","steps":[],"continuity":[]}`
- route/map: `{"type":"route","stops":[],"connectors":[],"legend":[]}`
- interface/game: `{"type":"interface","zones":[],"state":{}}`
- proposal board: `{"type":"proposal_board","render_frames":[],"material_schedule":[],"dimension_schedule":[]}`

Each member contains only its unique state and relationship. Shared medium,
palette, lighting, and finish remain in the core fields. Structure fields may
reference content roles but may not introduce visible literals.

Use neutral machine identifiers such as `panel_1` and `step_1`. A member's
state must describe visible imagery, not a label to typeset. Never use field
names such as `line_art`, `base_color`, or a prose state as implied panel copy
unless that exact label is separately whitelisted in `content`.

`structure` is mandatory when the brief requests multiple deliverables, views,
panels, variants, pages, formats, or states. Include every requested member and
give it concrete, visible category context; do not replace functional members
with generic empty boxes.

## Compression and priority

For SenseNova U1.5 and other text-sensitive image generators, target 900–1800
serialized characters excluding user-supplied visible copy. User-supplied copy
or genuinely dense locked structure may extend the output only as necessary.
Priority:

1. locked subject, action, count, relationship, canvas, and copy;
2. main subject and observable event;
3. composition, camera, and spatial layers;
4. coherent medium, light, palette, and materials;
5. exact structure;
6. finish and failure protection.

Remove cross-field repetition. The result should read like a compact reverse
description of an already-resolved successful image, not a design explanation.

In text-safe compact mode, keep each non-content value to one short
pixel-causal phrase; each `layout_composition` value is under 40 CJK
characters; `composition_rules` is normally empty; and no grid measurement,
safe-area note, material schedule, dimension schedule, production annotation,
or specification rail is added unless explicitly requested as visible content.

## Final prompt-to-image text-isolation check

Before output, scan every string:

1. Remove direct copies of the user's request sentence; translate them into
   observable subject, action, medium, and layout.
2. Remove reusable boilerplate, schema names, field explanations, rationale,
   prohibitions, and prompt/model/user language.
3. Confirm every `content` leaf is an exact viewer-facing literal.
4. Confirm all non-content strings are concise visual descriptions that would
   not form a plausible footer, checklist, specification panel, or body-copy
   block if accidentally typeset.
