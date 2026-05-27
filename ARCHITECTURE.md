# Open ML Engineer Architecture

## High-Level Architecture

Open ML Engineer is a static-first educational platform.

- Frontend: Next.js, React, TypeScript.
- Styling: Tailwind CSS with a restrained design system.
- Content: MDX-compatible lesson routes and structured curriculum metadata.
- Visuals: React state, SVG, Canvas, and D3 for scales, layouts, and plots.
- Hosting: GitHub Pages through `next export` or Vercel static hosting.
- Backend: none for v1. Optional services can be added later for accounts,
  discussions, saved progress, and cohort features.

Next.js is the default choice because it gives React, file routing, static
export, MDX support, and a future path to server features without forcing a
backend today.

## Folder Structure

```text
app/
  page.tsx
  layout.tsx
  globals.css
  docs/
    architecture/page.tsx
  learn/
    level-1/
      linear-regression/page.tsx
components/
  interactive/
    LinearRegressionPlayground.tsx
content/
  curriculum.ts
public/
ARCHITECTURE.md
CONTRIBUTING.md
ROADMAP.md
```

## Lesson Page Template

Every topic page must contain:

1. Big Picture
2. Core Intuition
3. Mathematical Foundation
4. From Scratch Implementation
5. Framework Implementation
6. Engineering Perspective
7. Interview Questions
8. Further Reading

This repeated structure makes the site usable as a textbook, interview prep
resource, wiki, and self-study roadmap.

## Component Architecture

- Lesson routes own topic-specific narrative.
- Shared lesson UI should move into `components/lesson`.
- Interactive explainers live in `components/interactive`.
- Curriculum metadata lives in `content/curriculum.ts`.
- Later, MDX files can import the same interactive components directly.

## Contributor Workflow

Contributors add lessons through pull requests. Each lesson should include a
clear conceptual arc, derivations, runnable code, engineering notes, interview
questions, and citations. Interactive components should be small, accessible,
and reusable across lessons.
