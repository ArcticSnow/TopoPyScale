# Sketches of the documentation

The figures of the science pages (`doc/docs/img/science/*.svg`) are drawn by hand in the clean SVG files of `src/`,
then turned into hand-drawn sketches by `roughen.js`. Edit the files of `src/`, never the generated ones, then run:

```sh
cd doc/sketches
npm install roughjs@4 jsdom@26
node roughen.js
```

- Lines, rectangles, circles, ellipses and paths are redrawn with [rough.js](https://roughjs.com) (MIT licence),
  with a fixed seed per element, so the output does not change between runs.
- Arrowheads (`marker-end`, `marker-start` on `<line>`) become two hand-drawn strokes in the colour of the marker.
- Text is set in Patrick Hand (Patrick Wagesreiter, SIL Open Font License 1.1, see `fonts/OFL.txt`), embedded in
  each SVG as a subset (`fonts/patrick-hand-subset.woff2`: printable ASCII and `°·½×…−±²³`). Greek letters, arrows
  and mathematical symbols, which the font lacks, are set in a math font.
- Text is enlarged by `SCALE` (1.35): the handwriting font is narrower than the sans-serif of the sources. Check that
  labels still fit after an edit.

To check a figure, open it in a browser: the embedded font is a web font, which some command-line renderers ignore.
