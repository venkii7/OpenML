# Contributing to Open ML Engineer

This project is GitHub-first. The goal is to make every lesson deep, visual,
correct, and maintainable.

## Add a New Lesson

1. Choose the curriculum level and topic.
2. Create a lesson route under `app/learn/<level>/<topic>/page.tsx`.
3. Follow the required topic structure:
   - Big Picture
   - Core Intuition
   - Mathematical Foundation
   - From Scratch Implementation
   - Framework Implementation
   - Engineering Perspective
   - Interview Questions
   - Further Reading
4. Add interactive components under `components/interactive`.
5. Update `content/curriculum.ts` if the topic changes the roadmap.
6. Run linting, type-checking, and a local visual review.

## Content Standards

- Explain from first principles before using framework abstractions.
- Include assumptions, edge cases, and engineering tradeoffs.
- Prefer small executable examples over broad pseudo-code.
- Cite papers, official docs, or high-quality open-source codebases.
- Avoid shallow summaries and motivational filler.

## Review Checklist

- The lesson has all required sections.
- Math notation is introduced before it is used.
- Code examples are minimal, correct, and reproducible.
- Interactive visuals have labels, accessible names, and stable layout.
- The topic connects to real systems and interview expectations.

## Future Translation Model

Translations should mirror the canonical English content path and store locale
metadata separately. Do not fork lesson structure per language.
