import { curriculum } from "@/content/curriculum";

const concepts = [
  "Linear Regression",
  "Logistic Regression",
  "Gradient Descent",
  "Transformers",
  "RAG Systems",
  "MLOps"
];

export default function HomePage() {
  return (
    <main className="min-h-screen bg-paper text-ink">
      <header className="fixed left-4 right-4 top-4 z-50 rounded-md border border-line bg-paper/90 px-5 py-4 backdrop-blur md:left-16 md:right-16">
        <nav className="flex items-center justify-between gap-6" aria-label="Primary navigation">
          <a className="brand-mark flex-shrink-0 text-xl font-black md:text-2xl" href="/">
            OpenML
          </a>
          <div className="hidden items-center gap-8 md:flex">
            <a className="nav-link" href="#roadmap">Roadmap</a>
            <a className="nav-link" href="#concepts">Concepts</a>
            <a className="nav-link" href="/docs/architecture">Docs</a>
          </div>
          <a className="contact-button" href="/docs/architecture">
            Architecture
          </a>
        </nav>
      </header>

      <section className="relative flex min-h-screen overflow-hidden px-5 pt-28 md:px-16">
        <div className="flex min-h-[calc(100vh-7rem)] w-full flex-col justify-between">
          <div className="relative z-10 max-w-3xl pt-10 md:pt-20">
            <p className="mono-label">You are now entering</p>
            <h1 className="mt-4 max-w-5xl text-4xl font-black leading-[0.95] tracking-normal md:text-7xl">
              AI/ML learning with engineering discipline.
            </h1>
          </div>

          <div className="relative min-h-[44vh]">
            <div className="hero-cue absolute right-0 top-3 z-10 hidden grid-cols-4 border-l border-line bg-paper/80 text-xs uppercase leading-5 text-muted backdrop-blur-sm md:grid">
              <div className="border-r border-line px-5">
                <p>You are</p>
                <p>now</p>
                <p>entering</p>
              </div>
              <a className="border-r border-line px-5 transition hover:text-ink" href="#roadmap">
                <p>AI/ML</p>
                <p>Roadmap</p>
              </a>
              <a className="border-r border-line px-5 transition hover:text-ink" href="#concepts">
                <p>AI/ML</p>
                <p>Concepts</p>
              </a>
              <div className="px-5">
                <p>Open-source</p>
                <p>curriculum</p>
              </div>
            </div>
            <div className="grid gap-3 pb-8 md:hidden">
              <a className="mobile-choice" href="#roadmap">AI/ML Roadmap</a>
              <a className="mobile-choice" href="#concepts">AI/ML Concepts</a>
            </div>
            <p className="hero-word pointer-events-none absolute bottom-0 left-0 z-0 whitespace-nowrap font-black leading-none">
              OpenML
            </p>
          </div>
        </div>
      </section>

      <section id="roadmap" className="border-t border-line px-5 py-20 md:px-16">
        <div className="grid gap-12 xl:grid-cols-[420px_1fr] 2xl:grid-cols-[460px_1fr]">
          <div className="max-w-[420px]">
            <p className="mono-label">AI/ML Roadmap</p>
            <h2 className="mt-4 text-4xl font-black leading-[1.05] md:text-5xl xl:text-6xl">
              From prerequisites to production AI.
            </h2>
          </div>
          <div className="grid border-t border-line md:grid-cols-2 xl:grid-cols-3">
            {curriculum.map((level) => (
              <article key={level.slug} className="min-h-72 border-b border-line p-6 md:border-r xl:p-8">
                <p className="mono-label text-muted">{level.id}</p>
                <h3 className="mt-5 text-2xl font-black">{level.title}</h3>
                <p className="mt-4 text-sm leading-7 text-muted">{level.description}</p>
                <ul className="mt-6 space-y-2 text-sm leading-6">
                  {level.modules.slice(0, 5).map((module) => (
                    <li key={module}>{module}</li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="concepts" className="border-t border-line px-5 py-20 md:px-16">
        <div className="mb-10 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mono-label">AI/ML Concepts</p>
            <h2 className="mt-4 text-4xl font-black leading-tight md:text-6xl">
              Learn one idea deeply.
            </h2>
          </div>
          <a className="contact-button w-fit" href="/learn/level-1/linear-regression">
            Start lesson
          </a>
        </div>

        <div className="grid border-t border-line md:grid-cols-2 xl:grid-cols-3">
          {concepts.map((concept, index) => (
            <a
              key={concept}
              href={
                index === 0
                  ? "/learn/level-1/linear-regression"
                  : index === 1
                    ? "/learn/level-1/logistic-regression"
                    : index === 2
                      ? "/learn/level-1/optimization"
                      : "#roadmap"
              }
              className="group min-h-56 border-b border-line p-6 transition hover:bg-ink hover:text-paper md:border-r"
            >
              <p className="mono-label text-muted transition group-hover:text-paper/60">
                Concept {String(index + 1).padStart(2, "0")}
              </p>
              <h3 className="mt-8 text-3xl font-black">{concept}</h3>
            </a>
          ))}
        </div>
      </section>
    </main>
  );
}
