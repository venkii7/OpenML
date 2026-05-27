"use client";

import { useMemo, useState } from "react";
import * as d3 from "d3";

type Point = { x: number; y: number };

const points: Point[] = [
  { x: 1, y: 1.4 },
  { x: 2, y: 1.9 },
  { x: 3, y: 3.2 },
  { x: 4, y: 3.7 },
  { x: 5, y: 5.1 },
  { x: 6, y: 5.6 },
  { x: 7, y: 6.9 },
  { x: 8, y: 7.4 }
];

export function LinearRegressionPlayground() {
  const [slope, setSlope] = useState(0.8);
  const [intercept, setIntercept] = useState(0.2);

  const width = 720;
  const height = 420;
  const margin = { top: 24, right: 32, bottom: 46, left: 54 };

  const xScale = useMemo(
    () => d3.scaleLinear().domain([0, 9]).range([margin.left, width - margin.right]),
    []
  );
  const yScale = useMemo(
    () => d3.scaleLinear().domain([0, 9]).range([height - margin.bottom, margin.top]),
    []
  );

  const mse = useMemo(() => {
    const total = points.reduce((sum, p) => {
      const prediction = slope * p.x + intercept;
      return sum + (p.y - prediction) ** 2;
    }, 0);
    return total / points.length;
  }, [slope, intercept]);

  const lineStart = { x: 0, y: intercept };
  const lineEnd = { x: 9, y: slope * 9 + intercept };

  return (
    <section className="rounded-md border border-line bg-white p-4">
      <div className="grid gap-4 lg:grid-cols-[1fr_220px]">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Linear regression fit visualization">
          <line x1={margin.left} y1={height - margin.bottom} x2={width - margin.right} y2={height - margin.bottom} stroke="#8a8a8a" />
          <line x1={margin.left} y1={margin.top} x2={margin.left} y2={height - margin.bottom} stroke="#8a8a8a" />
          {points.map((p) => {
            const predictedY = slope * p.x + intercept;
            return (
              <g key={`${p.x}-${p.y}`}>
                <line
                  x1={xScale(p.x)}
                  y1={yScale(p.y)}
                  x2={xScale(p.x)}
                  y2={yScale(predictedY)}
                  stroke="#8a8a8a"
                  strokeDasharray="4 4"
                />
                <circle cx={xScale(p.x)} cy={yScale(p.y)} r="5" fill="#242424" />
              </g>
            );
          })}
          <line
            x1={xScale(lineStart.x)}
            y1={yScale(lineStart.y)}
            x2={xScale(lineEnd.x)}
            y2={yScale(lineEnd.y)}
            stroke="#242424"
            strokeWidth="3"
          />
          <text x={margin.left} y={height - 12} fontSize="13" fill="#5f5f5f">feature x</text>
          <text x={12} y={margin.top + 12} fontSize="13" fill="#5f5f5f">target y</text>
        </svg>
        <div className="space-y-5">
          <div>
            <label className="text-sm font-semibold" htmlFor="slope">Slope: {slope.toFixed(2)}</label>
            <input
              id="slope"
              className="mt-2 w-full accent-neutral-900"
              type="range"
              min="-1"
              max="2"
              step="0.05"
              value={slope}
              onChange={(event) => setSlope(Number(event.target.value))}
            />
          </div>
          <div>
            <label className="text-sm font-semibold" htmlFor="intercept">Intercept: {intercept.toFixed(2)}</label>
            <input
              id="intercept"
              className="mt-2 w-full accent-neutral-900"
              type="range"
              min="-2"
              max="4"
              step="0.05"
              value={intercept}
              onChange={(event) => setIntercept(Number(event.target.value))}
            />
          </div>
          <div className="rounded-md border border-line bg-paper p-3 text-sm">
            <p className="font-bold">Mean squared error</p>
            <p className="mt-1 text-2xl font-black text-signal">{mse.toFixed(3)}</p>
            <p className="mt-2 leading-6 text-muted">
              Dotted residuals show each prediction error. Training tries to
              find parameters that make these residuals small together.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
