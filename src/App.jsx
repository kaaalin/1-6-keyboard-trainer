import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

const STORAGE_KEY = "number-grid-trainer-last-run-v2";
const ALPHA = 0.05;
const DIGITS = ["1", "2", "3", "4", "5", "6"];
const GRID_SIZE = 6;
const CELL_COUNT = GRID_SIZE * GRID_SIZE;
const TOTAL = CELL_COUNT;

function randomDigit() {
  return DIGITS[Math.floor(Math.random() * DIGITS.length)];
}

function makeGrid() {
  return Array.from({ length: CELL_COUNT }, () => randomDigit());
}

function formatSeconds(ms) {
  if (!Number.isFinite(ms)) return "—";
  return `${(ms / 1000).toFixed(2)}s`;
}

function formatP(p) {
  if (!Number.isFinite(p)) return "—";
  if (p < 0.001) return "< 0.001";
  return p.toFixed(3);
}

function keyLabel(key) {
  if (key === " ") return "Space";
  if (key === "ArrowUp") return "↑";
  if (key === "ArrowDown") return "↓";
  if (key === "ArrowLeft") return "←";
  if (key === "ArrowRight") return "→";
  if (key === "Enter") return "Enter";
  if (key === "Backspace") return "Backspace";
  return key.length === 1 ? key : key.replace(/^Key/, "");
}

function isIgnorableKey(e) {
  return ["Shift", "Control", "Alt", "Meta", "CapsLock", "NumLock", "ScrollLock", "Tab"].includes(e.key);
}

function calculateSummary(results) {
  const typed = results.length;
  const correct = results.filter((r) => r.correct).length;
  const wrong = typed - correct;
  const speedValues = results.map((r) => r.speedMs).filter(Number.isFinite);
  const averageSpeedMs = speedValues.length
    ? Math.round(speedValues.reduce((sum, value) => sum + value, 0) / speedValues.length)
    : null;
  const accuracy = typed ? Math.round((correct / typed) * 100) : 0;
  return { typed, correct, wrong, averageSpeedMs, accuracy };
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function variance(values) {
  if (values.length < 2) return null;
  const m = mean(values);
  return values.reduce((sum, value) => sum + Math.pow(value - m, 2), 0) / (values.length - 1);
}

function logGamma(z) {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];

  if (z < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
  }

  const adjusted = z - 1;
  let x = 0.99999999999980993;

  for (let i = 0; i < coefficients.length; i += 1) {
    x += coefficients[i] / (adjusted + i + 1);
  }

  const t = adjusted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (adjusted + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaContinuedFraction(x, a, b) {
  const maxIterations = 100;
  const epsilon = 3e-8;
  const fpMin = 1e-30;
  const qab = a + b;
  const qap = a + 1;
  const qam = a - 1;

  let c = 1;
  let d = 1 - (qab * x) / qap;

  if (Math.abs(d) < fpMin) d = fpMin;
  d = 1 / d;

  let h = d;

  for (let m = 1; m <= maxIterations; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));

    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;
    h *= d * c;

    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < fpMin) d = fpMin;
    c = 1 + aa / c;
    if (Math.abs(c) < fpMin) c = fpMin;
    d = 1 / d;

    const delta = d * c;
    h *= delta;

    if (Math.abs(delta - 1) < epsilon) break;
  }

  return h;
}

function regularizedBeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const betaTerm = Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x)
  );

  if (x < (a + 1) / (a + b + 2)) {
    return (betaTerm * betaContinuedFraction(x, a, b)) / a;
  }

  return 1 - (betaTerm * betaContinuedFraction(1 - x, b, a)) / b;
}

function fSurvival(f, df1, df2) {
  if (!Number.isFinite(f) || f < 0 || df1 <= 0 || df2 <= 0) return null;
  const x = (df1 * f) / (df1 * f + df2);
  const cdf = regularizedBeta(x, df1 / 2, df2 / 2);
  return Math.max(0, Math.min(1, 1 - cdf));
}

function oneVsRestWelchP(groupValues, restValues) {
  if (groupValues.length < 2 || restValues.length < 2) return null;

  const groupMean = mean(groupValues);
  const restMean = mean(restValues);
  const groupVariance = variance(groupValues);
  const restVariance = variance(restValues);

  if (!Number.isFinite(groupMean) || !Number.isFinite(restMean)) return null;
  if (!Number.isFinite(groupVariance) || !Number.isFinite(restVariance)) return null;

  const standardErrorSquared = groupVariance / groupValues.length + restVariance / restValues.length;
  if (!Number.isFinite(standardErrorSquared) || standardErrorSquared <= 0) return null;

  const t = (groupMean - restMean) / Math.sqrt(standardErrorSquared);
  const numerator = Math.pow(standardErrorSquared, 2);
  const denominator =
    Math.pow(groupVariance / groupValues.length, 2) / (groupValues.length - 1) +
    Math.pow(restVariance / restValues.length, 2) / (restValues.length - 1);
  const df = numerator / denominator;

  if (!Number.isFinite(df) || df <= 0) return null;
  return fSurvival(t * t, 1, df);
}

function runAnova(results) {
  const valid = results.filter((r) => Number.isFinite(r.speedMs));
  const baseGroups = DIGITS.map((digit) => {
    const values = valid.filter((r) => r.expected === digit).map((r) => r.speedMs);
    return {
      digit,
      values,
      n: values.length,
      meanMs: mean(values),
      restMeanMs: null,
      posthocP: null,
      isSlower: false,
    };
  });

  const nonEmptyGroups = baseGroups.filter((g) => g.n > 0);
  const allValues = nonEmptyGroups.flatMap((g) => g.values);
  const grandMean = mean(allValues);
  const k = nonEmptyGroups.length;
  const n = allValues.length;

  if (k < 2 || n <= k || !Number.isFinite(grandMean)) {
    return {
      available: false,
      groups: baseGroups,
      message: "Not enough between-keypress data for ANOVA yet.",
    };
  }

  const ssBetween = nonEmptyGroups.reduce((sum, g) => sum + g.n * Math.pow(g.meanMs - grandMean, 2), 0);
  const ssWithin = nonEmptyGroups.reduce(
    (sum, g) => sum + g.values.reduce((inner, value) => inner + Math.pow(value - g.meanMs, 2), 0),
    0
  );
  const dfBetween = k - 1;
  const dfWithin = n - k;
  const msBetween = ssBetween / dfBetween;
  const msWithin = ssWithin / dfWithin;

  if (!Number.isFinite(msWithin) || msWithin <= 0) {
    return {
      available: false,
      groups: baseGroups,
      message: "ANOVA cannot be calculated because there is no measurable variation inside the groups.",
    };
  }

  const f = msBetween / msWithin;
  const p = fSurvival(f, dfBetween, dfWithin);
  const bonferroniAlpha = ALPHA / DIGITS.length;
  const enrichedGroups = baseGroups.map((g) => {
    const restValues = baseGroups.filter((other) => other.digit !== g.digit).flatMap((other) => other.values);
    const restMeanMs = mean(restValues);
    const posthocP = oneVsRestWelchP(g.values, restValues);
    const isSlower =
      Number.isFinite(p) &&
      p < ALPHA &&
      Number.isFinite(posthocP) &&
      posthocP < bonferroniAlpha &&
      Number.isFinite(g.meanMs) &&
      Number.isFinite(restMeanMs) &&
      g.meanMs > restMeanMs;

    return { ...g, restMeanMs, posthocP, isSlower };
  });
  const slowerDigits = enrichedGroups.filter((g) => g.isSlower).map((g) => g.digit);

  return {
    available: true,
    groups: enrichedGroups,
    f,
    p,
    dfBetween,
    dfWithin,
    significant: Number.isFinite(p) && p < ALPHA,
    slowerDigits,
    message:
      Number.isFinite(p) && p < ALPHA
        ? slowerDigits.length
          ? `Significantly slower: ${slowerDigits.join(", ")}.`
          : "Overall ANOVA is significant, but no single number passes the stricter slower-number check."
        : "No statistically significant speed difference between numbers in this run.",
  };
}

function AnovaPanel({ results }) {
  const anova = useMemo(() => (results.length >= TOTAL ? runAnova(results) : null), [results]);

  if (!anova) return null;

  const sortedGroups = [...anova.groups].sort((a, b) => {
    if (!Number.isFinite(a.meanMs)) return 1;
    if (!Number.isFinite(b.meanMs)) return -1;
    return b.meanMs - a.meanMs;
  });

  return (
    <div className="rounded-xl bg-neutral-950 border border-neutral-800 p-4 space-y-4">
      <div>
        <div className="text-sm mb-1">ANOVA after full grid</div>
        <div className="text-lg font-semibold">{anova.message}</div>
        <div className="text-xs mt-1">
          Timing is measured between consecutive keypresses. The first square is excluded from speed averages because it has no previous keypress.
        </div>
      </div>

      {anova.available && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center">
          <div className="rounded-lg bg-neutral-800 p-3">
            <div className="text-xl font-semibold">{anova.f.toFixed(2)}</div>
            <div className="text-xs uppercase tracking-wide">F</div>
          </div>
          <div className="rounded-lg bg-neutral-800 p-3">
            <div className="text-xl font-semibold">{formatP(anova.p)}</div>
            <div className="text-xs uppercase tracking-wide">p-value</div>
          </div>
          <div className="rounded-lg bg-neutral-800 p-3">
            <div className="text-xl font-semibold">
              {anova.dfBetween}, {anova.dfWithin}
            </div>
            <div className="text-xs uppercase tracking-wide">df</div>
          </div>
          <div className="rounded-lg bg-neutral-800 p-3">
            <div className="text-xl font-semibold">{anova.significant ? "Yes" : "No"}</div>
            <div className="text-xs uppercase tracking-wide">significant</div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="text-sm">Speed by number</div>
        {sortedGroups.map((g) => (
          <div
            key={g.digit}
            className={`grid grid-cols-5 gap-2 items-center rounded-lg px-3 py-2 text-sm ${
              g.isSlower ? "bg-neutral-700" : "bg-neutral-800"
            }`}
          >
            <div className="font-bold text-lg tabular-nums">{g.digit}</div>
            <div className="tabular-nums">n={g.n}</div>
            <div className="tabular-nums">avg {formatSeconds(g.meanMs)}</div>
            <div className="tabular-nums">p {formatP(g.posthocP)}</div>
            <div className="text-right">{g.isSlower ? "slower" : ""}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function cellBorderClass(index) {
  const row = Math.floor(index / GRID_SIZE);
  const col = index % GRID_SIZE;
  const classes = ["border-neutral-500"];
  if (col === 2) classes.push("border-r-4", "border-r-neutral-600");
  if (row === 1 || row === 3) classes.push("border-b-4", "border-b-neutral-600");
  return classes.join(" ");
}

function nextPendingIndex(marks, currentIndex) {
  for (let index = currentIndex + 1; index < CELL_COUNT; index += 1) {
    if (!marks[index]) return index;
  }
  return null;
}

export default function NumberGridTrainer() {
  const [running, setRunning] = useState(false);
  const [grid, setGrid] = useState(() => makeGrid());
  const [activeIndex, setActiveIndex] = useState(null);
  const [cellMarks, setCellMarks] = useState({});
  const [round, setRound] = useState(0);
  const [typed, setTyped] = useState("");
  const [results, setResults] = useState([]);
  const [lastRun, setLastRun] = useState(null);
  const [status, setStatus] = useState("Press Start, then fill the grid one square at a time from left to right.");

  const inputRef = useRef(null);
  const gridRef = useRef(grid);
  const activeIndexRef = useRef(activeIndex);
  const runningRef = useRef(false);
  const lastPressAtRef = useRef(null);
  const resultsRef = useRef(results);
  const marksRef = useRef(cellMarks);

  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved) setLastRun(JSON.parse(saved));
    } catch {}
  }, []);

  useEffect(() => {
    gridRef.current = grid;
    activeIndexRef.current = activeIndex;
    runningRef.current = running;
    resultsRef.current = results;
    marksRef.current = cellMarks;
  }, [grid, activeIndex, running, results, cellMarks]);

  const summary = calculateSummary(results);
  const previousAverageMs = lastRun?.summary?.averageSpeedMs ?? null;
  const previousAccuracy = lastRun?.summary?.accuracy ?? null;
  const previousWrong = lastRun?.summary?.wrong ?? null;
  const speedProgressMs =
    Number.isFinite(summary.averageSpeedMs) && Number.isFinite(previousAverageMs)
      ? summary.averageSpeedMs - previousAverageMs
      : null;
  const accuracyProgress = results.length > 0 && Number.isFinite(previousAccuracy) ? summary.accuracy - previousAccuracy : null;
  const wrongProgress = results.length > 0 && Number.isFinite(previousWrong) ? summary.wrong - previousWrong : null;

  function saveCompletedRun(finalResults) {
    const run = {
      completedAt: new Date().toISOString(),
      results: finalResults,
      summary: calculateSummary(finalResults),
    };

    setLastRun(run);

    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(run));
    } catch {}
  }

  function start() {
    const nextGrid = makeGrid();
    gridRef.current = nextGrid;
    activeIndexRef.current = 0;
    runningRef.current = true;
    lastPressAtRef.current = null;
    resultsRef.current = [];
    marksRef.current = {};

    setGrid(nextGrid);
    setActiveIndex(0);
    setCellMarks({});
    setRunning(true);
    setResults([]);
    setRound(0);
    setTyped("");
    setStatus("Started successfully. Fill square 1, then the cursor will move to the next square.");
    window.setTimeout(() => inputRef.current?.focus(), 0);
  }

  function stop() {
    runningRef.current = false;
    activeIndexRef.current = null;
    lastPressAtRef.current = null;
    setRunning(false);
    setActiveIndex(null);
    setStatus("Stopped. Press Start to begin again.");
  }

  function resetPreviousRun() {
    setLastRun(null);
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {}
  }

  function focusTrainer() {
    inputRef.current?.focus();
  }

  function handleKeyDown(e) {
    if (!runningRef.current || isIgnorableKey(e)) return;

    const index = activeIndexRef.current;
    if (index === null || index === undefined) return;

    e.preventDefault();

    const now = performance.now();
    const expected = gridRef.current[index];
    const actual = e.key;
    const actualDisplay = keyLabel(actual);
    const correct = actual === expected;
    const nextRound = resultsRef.current.length + 1;
    const speedMs = lastPressAtRef.current === null ? null : Math.round(now - lastPressAtRef.current);
    const row = Math.floor(index / GRID_SIZE) + 1;
    const col = (index % GRID_SIZE) + 1;
    const newResult = {
      round: nextRound,
      expected,
      actual: actualDisplay,
      correct,
      speedMs,
      cell: index,
      row,
      col,
    };

    lastPressAtRef.current = now;
    setTyped(actualDisplay);

    const updatedMarks = { ...marksRef.current, [index]: correct ? "✓" : "×" };
    const updatedResults = [...resultsRef.current, newResult];
    const nextIndex = nextPendingIndex(updatedMarks, index);

    marksRef.current = updatedMarks;
    resultsRef.current = updatedResults;
    activeIndexRef.current = nextIndex;

    setCellMarks(updatedMarks);
    setResults(updatedResults);
    setRound(nextRound);
    setActiveIndex(nextIndex);

    if (updatedResults.length >= TOTAL || nextIndex === null) {
      runningRef.current = false;
      activeIndexRef.current = null;
      setRunning(false);
      setActiveIndex(null);
      saveCompletedRun(updatedResults);
      setStatus("Finished. You filled the whole grid.");
      return;
    }

    setStatus(
      correct
        ? `Correct ✓${speedMs === null ? "" : ` · ${formatSeconds(speedMs)}`}. Move to square ${nextIndex + 1}.`
        : `Wrong × · expected ${expected}, typed ${actualDisplay}${speedMs === null ? "" : ` · ${formatSeconds(speedMs)}`}. Move to square ${nextIndex + 1}.`
    );
  }

  function progressText() {
    if (!lastRun) return "No previous completed run yet.";

    if (results.length === 0) {
      return `Previous: ${lastRun.summary.accuracy}% accuracy, ${lastRun.summary.wrong} wrong, ${formatSeconds(
        lastRun.summary.averageSpeedMs
      )} avg between keypresses.`;
    }

    const speedPart =
      speedProgressMs === null
        ? "speed —"
        : speedProgressMs < 0
          ? `${formatSeconds(Math.abs(speedProgressMs))} faster`
          : speedProgressMs > 0
            ? `${formatSeconds(speedProgressMs)} slower`
            : "same speed";
    const accuracyPart =
      accuracyProgress === null
        ? "accuracy —"
        : accuracyProgress > 0
          ? `accuracy +${accuracyProgress}%`
          : accuracyProgress < 0
            ? `accuracy ${accuracyProgress}%`
            : "same accuracy";
    const wrongPart =
      wrongProgress === null
        ? "wrong —"
        : wrongProgress < 0
          ? `${Math.abs(wrongProgress)} fewer wrong`
          : wrongProgress > 0
            ? `${wrongProgress} more wrong`
            : "same wrong count";

    return `Vs previous run: ${speedPart}, ${accuracyPart}, ${wrongPart}.`;
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-white flex items-center justify-center p-6 [&_*]:text-white" onClick={focusTrainer}>
      <Card className="w-full max-w-3xl bg-neutral-900 border-neutral-800 shadow-2xl rounded-2xl">
        <CardContent className="p-8 space-y-6">
          <div className="space-y-2 text-center">
            <h1 className="text-3xl font-semibold tracking-tight">1–6 Grid Keyboard Trainer</h1>
            <p>Fill the full grid one by one · type the active light-grey number · timing between keypresses</p>
          </div>

          <input
            ref={inputRef}
            value={typed}
            onChange={() => {}}
            onKeyDown={handleKeyDown}
            inputMode="numeric"
            autoFocus
            className="sr-only"
            aria-label="Type the active square number"
          />

          <div className="flex justify-center">
            <div className="grid grid-cols-6 w-full max-w-[480px] aspect-square rounded-md overflow-hidden border-4 border-neutral-600 bg-neutral-100">
              {grid.map((digit, index) => {
                const active = activeIndex === index;
                const mark = cellMarks[index];
                const done = Boolean(mark);

                return (
                  <button
                    key={`${index}-${digit}`}
                    type="button"
                    onClick={focusTrainer}
                    className={`relative bg-neutral-50 border border-neutral-400 flex items-center justify-center text-3xl md:text-4xl font-semibold tabular-nums transition ${cellBorderClass(index)} ${
                      active ? "ring-4 ring-inset ring-neutral-900 bg-neutral-200" : ""
                    } ${done ? "bg-neutral-100" : ""}`}
                  >
                    <span className="!text-neutral-400">{digit}</span>
                    {mark && (
                      <span className={`absolute inset-0 flex items-center justify-center text-5xl font-black ${mark === "✓" ? "!text-green-700" : "!text-red-700"}`}>
                        {mark}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="text-center space-y-1">
            <p className="text-lg">{status}</p>
            <p>{`Filled ${Math.min(round, TOTAL)} / ${TOTAL}`}</p>
            <p className="text-sm">{progressText()}</p>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-center">
            <div className="rounded-xl bg-neutral-800 p-3">
              <div className="text-2xl font-semibold">{summary.typed}</div>
              <div className="text-xs uppercase tracking-wide">typed</div>
            </div>
            <div className="rounded-xl bg-neutral-800 p-3">
              <div className="text-2xl font-semibold">{summary.correct}</div>
              <div className="text-xs uppercase tracking-wide">correct</div>
            </div>
            <div className="rounded-xl bg-neutral-800 p-3">
              <div className="text-2xl font-semibold">{summary.wrong}</div>
              <div className="text-xs uppercase tracking-wide">wrong</div>
            </div>
            <div className="rounded-xl bg-neutral-800 p-3">
              <div className="text-2xl font-semibold">{summary.accuracy}%</div>
              <div className="text-xs uppercase tracking-wide">accuracy</div>
            </div>
            <div className="rounded-xl bg-neutral-800 p-3 col-span-2 md:col-span-1">
              <div className="text-2xl font-semibold">{formatSeconds(summary.averageSpeedMs)}</div>
              <div className="text-xs uppercase tracking-wide">avg speed</div>
            </div>
          </div>

          <AnovaPanel results={results} />

          <div className="flex flex-wrap gap-3 justify-center">
            <Button onClick={start} className="rounded-xl px-6 text-white">
              Start
            </Button>
            <Button onClick={stop} variant="secondary" className="rounded-xl px-6 text-white">
              Stop
            </Button>
            <Button onClick={resetPreviousRun} variant="outline" className="rounded-xl px-6 border-neutral-700 text-white hover:bg-neutral-800">
              Reset previous
            </Button>
          </div>

          <div className="rounded-xl bg-neutral-950 border border-neutral-800 p-4 min-h-28">
            <div className="flex justify-between gap-3 text-sm mb-3">
              <span>History</span>
              <span>{results.length ? "actual / expected · cell · speed" : ""}</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {results.length === 0 ? (
                <span>No keys typed yet.</span>
              ) : (
                results.map((r) => (
                  <div
                    key={r.round}
                    className={`flex items-center justify-between gap-3 px-3 py-2 rounded-lg text-sm ${
                      r.correct ? "bg-neutral-800" : "bg-neutral-700"
                    }`}
                    title={`Round ${r.round}: row ${r.row}, column ${r.col}, expected ${r.expected}, typed ${r.actual}`}
                  >
                    <span className="tabular-nums">#{String(r.round).padStart(2, "0")}</span>
                    <span className="font-semibold tabular-nums">
                      {r.correct ? "✓" : "×"} {r.actual}/{r.expected}
                    </span>
                    <span className="tabular-nums">r{r.row} c{r.col}</span>
                    <span className="tabular-nums">{formatSeconds(r.speedMs)}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
