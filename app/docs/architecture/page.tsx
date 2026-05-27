export default function ArchitecturePage() {
  return (
    <main className="mx-auto max-w-4xl px-6 py-10 lesson-prose">
      <h1 className="text-4xl font-black">System Architecture</h1>
      <p>
        Open ML Engineer is a static-first Next.js and MDX site. Lessons live as
        content files, interactive explanations live as typed React components,
        and the platform can deploy for free to GitHub Pages or Vercel.
      </p>
      <h2>Why Next.js</h2>
      <p>
        Next.js gives React, MDX, static export, file-based routing, strong
        TypeScript support, and a smooth path to optional server features later.
        The first version avoids a backend so the project remains cheap,
        forkable, and easy for contributors.
      </p>
      <h2>Core Layers</h2>
      <ul>
        <li>Content: MDX lessons, curriculum metadata, reading lists, exercises.</li>
        <li>Learning UI: lesson shell, navigation, callouts, equations, code blocks.</li>
        <li>Interactive visuals: D3, SVG, Canvas, and React state.</li>
        <li>Quality gates: linting, type-checking, content review, visual review.</li>
      </ul>
      <h2>Authoring Workflow</h2>
      <p>
        A contributor adds one MDX lesson under content, imports approved
        interactive components, fills the required lesson sections, and opens a
        pull request using the topic checklist.
      </p>
    </main>
  );
}
