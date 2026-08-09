/** The Logs view fills the window instead of sitting in a fixed card.
 *
 * This is a stylesheet test, not a render test, and the reason is worth stating
 * because it bounds what it proves. jsdom has no layout engine: it parses CSS
 * and answers `matches()`, but it computes no boxes, so "the pane is 700px
 * tall" is not observable here. What *is* observable — and what actually broke —
 * is the cascade. A `.logs pre { max-height: 320px }` rule survived the move
 * from a dashboard card to a dedicated view. It has lower specificity than
 * `.panel.logs pre`, but that rule declares no `max-height` of its own, so the
 * cap was never overridden: the pane stayed 320px whatever the window did, and
 * every review that read only the newer rule saw a working full-height layout.
 *
 * So the property pinned here is the one that failed: **no rule that applies to
 * the Logs view's `pre` may cap its height, and the container must be anchored
 * to the viewport.** Both are checked against the real stylesheet, through the
 * real element the real component renders.
 *
 * What this does not establish: that the rendered pane is actually tall. That
 * needs a browser, and it belongs to the manual release matrix.
 */

import { describe, expect, it } from "vitest";

// The application's own stylesheet, imported exactly as `main.tsx` imports it,
// so what is examined below is the parsed cascade the app really gets rather
// than a copy of the text. `test.css` is enabled in `vitest.config.ts` for this;
// without it Vitest stubs the import and every assertion here would pass while
// checking an empty sheet — which is what the counterexample test guards.
import "./styles.css";

/** Declarations that cap a box's height. Any one of them defeats `flex-grow`. */
const CAPPING_PROPERTIES = ["max-height", "height"];

/** The Logs view's own markup, as `App.tsx` renders it. */
function logsPre(): HTMLElement {
  document.body.innerHTML = `
    <main>
      <header></header>
      <section class="panel logs">
        <h2>Server log</h2>
        <pre>a line</pre>
      </section>
    </main>`;
  const pre = document.querySelector("main > section.panel.logs > pre");
  if (!(pre instanceof HTMLElement)) throw new Error("the Logs markup changed shape");
  return pre;
}

/** Every style rule the document actually carries, as (selector, style) pairs.
 *
 * Read through the CSSOM rather than by regex so that a rule written across
 * several lines, or with its properties in another order, is still seen.
 */
function styleRules(): { selector: string; style: CSSStyleDeclaration }[] {
  const rules: { selector: string; style: CSSStyleDeclaration }[] = [];
  for (const sheet of Array.from(document.styleSheets)) {
    for (const rule of Array.from(sheet.cssRules)) {
      if (rule instanceof CSSStyleRule) {
        rules.push({ selector: rule.selectorText, style: rule.style });
      }
    }
  }
  if (rules.length === 0) throw new Error("no stylesheet reached the document");
  return rules;
}

describe("the Logs pane is not capped to a fixed height", () => {
  it("has no rule that caps the height of its pre", () => {
    const pre = logsPre();

    const capping = styleRules().filter(({ selector, style }) => {
      if (!CAPPING_PROPERTIES.some((property) => style.getPropertyValue(property) !== "")) {
        return false;
      }
      // A selector jsdom cannot parse cannot be shown to apply, and a rule that
      // does not apply is not this test's business.
      try {
        return pre.matches(selector);
      } catch {
        return false;
      }
    });

    expect(
      capping.map(({ selector }) => selector),
      "a height cap on the Logs pre silently defeats flex-grow; scope the rule to whatever else wanted it",
    ).toEqual([]);
  });

  it("still caps the setup log, which is a card and is meant to be", () => {
    // The counterexample that keeps the rule above honest: it must be rejecting
    // caps because they apply to *this* pre, not because it never finds any.
    document.body.innerHTML = `<pre class="setup-log"></pre>`;
    const setupLog = document.querySelector("pre.setup-log") as HTMLElement;

    const capped = styleRules().some(({ selector, style }) => {
      if (style.getPropertyValue("max-height") === "") return false;
      try {
        return setupLog.matches(selector);
      } catch {
        return false;
      }
    });

    expect(capped).toBe(true);
  });
});

describe("the Logs container is anchored to the viewport", () => {
  /** A viewport-relative length, in any of the CSS units that mean one. */
  const VIEWPORT_LENGTH = /^\d+(\.\d+)?\s*(vh|dvh|svh|lvh)$/;

  it("gives the pane a definite height, not merely a floor", () => {
    // The distinction this test exists for. `min-height: 100vh` reads like the
    // same thing and is not: the pane's `flex-basis` is `auto`, so its base
    // size is the entire log, the container grows past the window to fit it,
    // and the *page* scrolls while the pane sits at content height. Measured in
    // the packaged app: an 868px viewport holding an 1846px document. Only a
    // definite height gives the flex chain something to distribute.
    const rules = styleRules().filter(({ selector }) => selector.includes(".panel.logs"));

    const definite = rules.filter(({ style }) =>
      VIEWPORT_LENGTH.test(style.getPropertyValue("height").trim()),
    );
    const floorOnly = rules.filter(
      ({ style }) =>
        VIEWPORT_LENGTH.test(style.getPropertyValue("min-height").trim()) &&
        !VIEWPORT_LENGTH.test(style.getPropertyValue("height").trim()) &&
        !VIEWPORT_LENGTH.test(style.getPropertyValue("max-height").trim()),
    );

    expect(
      floorOnly.map(({ selector }) => selector),
      "a viewport `min-height` is a floor: the container still grows to its content and the page scrolls",
    ).toEqual([]);
    expect(
      definite.length,
      "no rule gives the Logs container a definite viewport height, so there is no 'rest of the window' to fill",
    ).toBeGreaterThan(0);
  });

  it("lets the flex children shrink below their content", () => {
    const pre = logsPre();
    const grows = styleRules().some(({ selector, style }) => {
      try {
        return pre.matches(selector) && style.getPropertyValue("flex").startsWith("1 1");
      } catch {
        return false;
      }
    });

    expect(grows).toBe(true);
  });
});
