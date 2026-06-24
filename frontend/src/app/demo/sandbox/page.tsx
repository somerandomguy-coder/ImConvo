"use client";

import { useEffect, useRef, useState, useMemo, useCallback } from "react";
import { analyzeVideo } from "@/utils/api";

// Constants for GRID-Bot columns
const COLUMNS = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y", "Z"
]; // Note: Skipping 'W' (25 columns total)

const DIGIT_WORDS: Record<string, number> = {
  zero: 0, one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9,
  "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9
};

const COLOR_MAP: Record<string, string> = {
  blue: "#3b82f6",
  green: "#22c55e",
  red: "#ef4444",
  white: "#ffffff",
};

const COLOR_SHADOW_MAP: Record<string, string> = {
  blue: "#1d4ed8",
  green: "#15803d",
  red: "#b91c1c",
  white: "#d1d5db",
};

const COLOR_LIGHT_MAP: Record<string, string> = {
  blue: "#93c5fd",
  green: "#86efac",
  red: "#fca5a5",
  white: "#f9fafb",
};

// Grid definitions
const COLS_COUNT = 25;
const ROWS_COUNT = 10;

interface GridCell {
  type: "empty" | "flat" | "block";
  color: "blue" | "green" | "red" | "white" | null;
}

interface RobotState {
  x: number;
  y: number;
  currentX: number;
  currentY: number;
  heading: number;
  speedMultiplier: number;
}

interface QueueItem {
  id: string;
  verb: "place" | "lay" | "bin";
  color: "blue" | "green" | "red" | "white";
  preposition: "at" | "by" | "in" | "with";
  col: number;
  row: number;
  adverb: "again" | "now" | "please" | "soon";
  originalText: string;
}

interface ToastMessage {
  id: string;
  message: string;
  type: "error" | "success" | "info";
}

interface LeaderboardRecord {
  initials: string;
  time: number; // in seconds
  date: string;
}

// 3 Level blueprints
const BLUEPRINTS = {
  cabin: {
    name: "The Cabin",
    description: "Build a cozy neon micro-cabin with surrounding grass and a red roof.",
    generate: (): GridCell[][] => {
      const grid = Array.from({ length: ROWS_COUNT }, () =>
        Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null } as GridCell))
      );
      // Grass foundation at bottom (Row 8, cols 8-16)
      for (let c = 8; c <= 16; c++) {
        grid[8][c] = { type: "flat", color: "green" };
      }
      // Cabin base walls: place white (Row 5, 6, 7; cols 10, 14, and solid row 7)
      for (let r = 5; r <= 7; r++) {
        grid[r][10] = { type: "block", color: "white" };
        grid[r][14] = { type: "block", color: "white" };
      }
      grid[5][11] = { type: "block", color: "white" };
      grid[5][12] = { type: "block", color: "white" };
      grid[5][13] = { type: "block", color: "white" };
      grid[7][11] = { type: "block", color: "white" };
      grid[7][13] = { type: "block", color: "white" };

      // Blue Door: lay blue (Row 6, 7; col 12)
      grid[6][12] = { type: "flat", color: "blue" };
      grid[7][12] = { type: "flat", color: "blue" };

      // Cabin Roof: place red (Row 4, cols 9-15; Row 3, cols 11-13)
      for (let c = 9; c <= 15; c++) {
        grid[4][c] = { type: "block", color: "red" };
      }
      for (let c = 11; c <= 13; c++) {
        grid[3][c] = { type: "block", color: "red" };
      }
      return grid;
    },
    presets: [
      "lay green with M 8 soon",
      "place white with K 6 soon",
      "place white with O 6 soon",
      "place white at N 5 soon",
      "place red with M 4 soon",
      "place red by M 4 soon",
      "lay blue at M 7 soon"
    ]
  },
  forest: {
    name: "The Forest",
    description: "Plant three standard micro-trees along a blue water stream.",
    generate: (): GridCell[][] => {
      const grid = Array.from({ length: ROWS_COUNT }, () =>
        Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null } as GridCell))
      );
      // Stream running through the bottom: lay blue (Row 9, cols 0-24)
      for (let c = 0; c < COLS_COUNT; c++) {
        grid[9][c] = { type: "flat", color: "blue" };
      }
      // Tree 1 centered at E (col 4)
      grid[7][4] = { type: "block", color: "white" }; // trunk
      grid[8][4] = { type: "block", color: "white" }; // trunk
      // Foliage: place green solid 3x3 centered at E 5 (col 4, row 5)
      for (let r = 4; r <= 6; r++) {
        for (let c = 3; c <= 5; c++) {
          grid[r][c] = { type: "block", color: "green" };
        }
      }
      // Tree 2 centered at M (col 12)
      grid[7][12] = { type: "block", color: "white" }; // trunk
      grid[8][12] = { type: "block", color: "white" }; // trunk
      // Foliage centered at M 5 (col 12, row 5)
      for (let r = 4; r <= 6; r++) {
        for (let c = 11; c <= 13; c++) {
          grid[r][c] = { type: "block", color: "green" };
        }
      }
      // Tree 3 centered at T (col 19)
      grid[7][19] = { type: "block", color: "white" }; // trunk
      grid[8][19] = { type: "block", color: "white" }; // trunk
      // Foliage centered at T 5 (col 19, row 5)
      for (let r = 4; r <= 6; r++) {
        for (let c = 18; c <= 20; c++) {
          grid[r][c] = { type: "block", color: "green" };
        }
      }
      return grid;
    },
    presets: [
      "lay blue with M 9 again",
      "place white at E 8 soon",
      "place white at E 7 soon",
      "place green with E 5 soon",
      "place white at M 8 soon",
      "place white at M 7 soon",
      "place green with M 5 soon",
      "place white at T 8 soon",
      "place white at T 7 soon",
      "place green with T 5 soon",
    ]
  },
  archway: {
    name: "The Archway",
    description: "Construct a massive glowing gateway of cyber-blue columns and red beams.",
    generate: (): GridCell[][] => {
      const grid = Array.from({ length: ROWS_COUNT }, () =>
        Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null } as GridCell))
      );
      // Left pillar: place blue (Row 3 to 8; cols 7, 8)
      for (let r = 3; r <= 8; r++) {
        grid[r][7] = { type: "block", color: "blue" };
        grid[r][8] = { type: "block", color: "blue" };
      }
      // Right pillar: place blue (Row 3 to 8; cols 16, 17)
      for (let r = 3; r <= 8; r++) {
        grid[r][16] = { type: "block", color: "blue" };
        grid[r][17] = { type: "block", color: "blue" };
      }
      // Top arch beam: place red (Row 2; cols 6-18)
      for (let c = 6; c <= 18; c++) {
        grid[2][c] = { type: "block", color: "red" };
      }
      // Inner lights: lay white (Row 5; cols 11-13)
      for (let c = 11; c <= 13; c++) {
        grid[5][c] = { type: "flat", color: "white" };
      }
      return grid;
    },
    presets: [
      "place blue with H 7 soon",
      "place blue with H 4 soon",
      "place blue with Q 7 soon",
      "place blue with Q 4 soon",
      "place red with M 2 soon",
      "place red by M 2 soon",
      "lay white with M 5 soon"
    ]
  }
};

// Seed initial leaderboard if empty
const INITIAL_LEADERBOARDS: Record<string, LeaderboardRecord[]> = {
  cabin: [
    { initials: "BOT", time: 14.2, date: "2026-06-24" },
    { initials: "MLV", time: 19.5, date: "2026-06-24" },
    { initials: "SYS", time: 27.8, date: "2026-06-24" }
  ],
  forest: [
    { initials: "BOT", time: 22.1, date: "2026-06-24" },
    { initials: "DRD", time: 28.4, date: "2026-06-24" },
    { initials: "ROB", time: 35.6, date: "2026-06-24" }
  ],
  archway: [
    { initials: "BOT", time: 31.8, date: "2026-06-24" },
    { initials: "ARC", time: 42.0, date: "2026-06-24" },
    { initials: "NEX", time: 54.3, date: "2026-06-24" }
  ]
};

export default function SandboxGamePage() {
  const [activeLevel, setActiveLevel] = useState<keyof typeof BLUEPRINTS>("cabin");
  const [userGrid, setUserGrid] = useState<GridCell[][]>(() =>
    Array.from({ length: ROWS_COUNT }, () =>
      Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null }))
    )
  );

  const [robot, setRobot] = useState<RobotState>({
    x: 12, // 'M' column
    y: 5,
    currentX: 12,
    currentY: 5,
    heading: 0,
    speedMultiplier: 1.0,
  });

  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [activeQueueItem, setActiveQueueItem] = useState<QueueItem | null>(null);
  const [robotState, setRobotState] = useState<"idle" | "moving" | "busy">("idle");
  const [inputText, setInputText] = useState("");
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [accuracy, setAccuracy] = useState(0);

  // Stopwatch state
  const [elapsedTime, setElapsedTime] = useState(0);
  const [timerStarted, setTimerStarted] = useState(false);
  const timerIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Leaderboard states
  const [leaderboard, setLeaderboard] = useState<LeaderboardRecord[]>([]);
  const [showWinModal, setShowWinModal] = useState(false);
  const [playerInitials, setPlayerInitials] = useState("");

  // Webcam states
  const [webcamActive, setWebcamActive] = useState(false);
  const [recordingState, setRecordingState] = useState<"idle" | "recording" | "analyzing">("idle");
  const [recordingProgress, setRecordingProgress] = useState(0);
  const [aiOfflineFallback, setAiOfflineFallback] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Particle systems
  const particlesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    color: string;
    alpha: number;
    size: number;
  }>>([]);

  // Toast helper
  const showToast = useCallback((message: string, type: "error" | "success" | "info" = "info") => {
    const id = Math.random().toString(36).substr(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  // Set active level and reset variables
  const loadLevel = useCallback((levelKey: keyof typeof BLUEPRINTS) => {
    setActiveLevel(levelKey);
    // Clear user canvas
    setUserGrid(
      Array.from({ length: ROWS_COUNT }, () =>
        Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null }))
      )
    );
    // Reset Robot position to M (12), 5
    setRobot({
      x: 12,
      y: 5,
      currentX: 12,
      currentY: 5,
      heading: 0,
      speedMultiplier: 1.0,
    });
    setQueue([]);
    setActiveQueueItem(null);
    setRobotState("idle");
    setInputText("");
    // Reset Timer
    setTimerStarted(false);
    setElapsedTime(0);
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setShowWinModal(false);
    setPlayerInitials("");

    showToast(`Loaded blueprint: ${BLUEPRINTS[levelKey].name}`, "info");
  }, [showToast]);

  // Initial load
  useEffect(() => {
    loadLevel("cabin");
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (recordIntervalRef.current) clearInterval(recordIntervalRef.current);
      // Clean up media stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, [loadLevel]);

  // Get active blueprint matrix
  const activeBlueprintMatrix = useMemo(() => {
    return BLUEPRINTS[activeLevel].generate();
  }, [activeLevel]);

  // Calculate Accuracy
  const calculateAccuracy = useCallback((grid: GridCell[][]) => {
    let matchCount = 0;
    const totalCells = COLS_COUNT * ROWS_COUNT;
    const blueprint = activeBlueprintMatrix;

    for (let r = 0; r < ROWS_COUNT; r++) {
      for (let c = 0; c < COLS_COUNT; c++) {
        const u = grid[r][c];
        const b = blueprint[r][c];
        if (u.type === b.type && u.color === b.color) {
          matchCount++;
        }
      }
    }
    return Math.round((matchCount / totalCells) * 100);
  }, [activeBlueprintMatrix]);

  // Update accuracy when grid changes
  useEffect(() => {
    const acc = calculateAccuracy(userGrid);
    setAccuracy(acc);

    if (acc === 100 && timerStarted) {
      // STOP stopwatch
      setTimerStarted(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      // Open win ceremony modal
      setShowWinModal(true);
      showToast("Sync Blueprint Complete! 100% Match!", "success");
    }
  }, [userGrid, calculateAccuracy, timerStarted, showToast]);

  // Start Stopwatch Timer on first input
  const startStopwatch = useCallback(() => {
    if (!timerStarted && !showWinModal) {
      setTimerStarted(true);
      const startTime = Date.now() - elapsedTime * 1000;
      timerIntervalRef.current = setInterval(() => {
        setElapsedTime(Math.round((Date.now() - startTime) / 100) / 10);
      }, 100);
    }
  }, [timerStarted, elapsedTime, showWinModal]);



  // Preposition brush helper
  const getBrushOffsets = (prep: "at" | "by" | "in" | "with"): Array<{ dx: number; dy: number }> => {
    if (prep === "in" || prep === "at") {
      return [{ dx: 0, dy: 0 }];
    }
    if (prep === "by") {
      return [
        { dx: -1, dy: -1 }, { dx: 0, dy: -1 }, { dx: 1, dy: -1 },
        { dx: -1, dy: 0 },                    { dx: 1, dy: 0 },
        { dx: -1, dy: 1 },  { dx: 0, dy: 1 },  { dx: 1, dy: 1 }
      ];
    }
    if (prep === "with") {
      return [
        { dx: -1, dy: -1 }, { dx: 0, dy: -1 }, { dx: 1, dy: -1 },
        { dx: -1, dy: 0 },  { dx: 0, dy: 0 },  { dx: 1, dy: 0 },
        { dx: -1, dy: 1 },  { dx: 0, dy: 1 },  { dx: 1, dy: 1 }
      ];
    }
    return [{ dx: 0, dy: 0 }];
  };

  // Command Execution: Apply paint changes to user grid
  const executePaintCommand = useCallback((item: QueueItem) => {
    setUserGrid((prevGrid) => {
      const nextGrid = prevGrid.map((row) => row.map((cell) => ({ ...cell })));
      const offsets = getBrushOffsets(item.preposition);

      offsets.forEach(({ dx, dy }) => {
        const targetCol = item.col + dx;
        const targetRow = item.row + dy;

        // Verify borders
        if (targetCol >= 0 && targetCol < COLS_COUNT && targetRow >= 0 && targetRow < ROWS_COUNT) {
          if (item.verb === "place") {
            nextGrid[targetRow][targetCol] = { type: "block", color: item.color };
          } else if (item.verb === "lay") {
            nextGrid[targetRow][targetCol] = { type: "flat", color: item.color };
          } else if (item.verb === "bin") {
            nextGrid[targetRow][targetCol] = { type: "empty", color: null };
          }
        }
      });
      return nextGrid;
    });

    // Particle effect burst
    const centerPxX = 30 + item.col * 30 + 15;
    const centerPxY = 20 + item.row * 30 + 15;
    const color = item.verb === "bin" ? "#ff3b30" : COLOR_MAP[item.color];

    for (let i = 0; i < 18; i++) {
      particlesRef.current.push({
        x: centerPxX,
        y: centerPxY,
        vx: (Math.random() - 0.5) * 4,
        vy: (Math.random() - 0.5) * 4,
        color,
        alpha: 1.0,
        size: Math.random() * 4 + 2,
      });
    }

    showToast(`Robot executed: ${item.verb} ${item.color} ${item.preposition} ${COLUMNS[item.col]} ${item.row}`, "success");

    // Handle "again" (Shift loop +1 tile to the right)
    if (item.adverb === "again") {
      if (item.col + 1 < COLS_COUNT) {
        const nextCol = item.col + 1;
        const duplicate: QueueItem = {
          id: Math.random().toString(36).substr(2, 9),
          verb: item.verb,
          color: item.color,
          preposition: item.preposition,
          col: nextCol,
          row: item.row,
          adverb: "soon", // Prevent infinite loop chains
          originalText: `${item.verb} ${item.color} ${item.preposition} ${COLUMNS[nextCol]} ${item.row} soon`,
        };
        // Add to the front of queue to run next
        setQueue((prev) => [duplicate, ...prev]);
      } else {
        showToast("Cannot shift 'again' command further right: boundary hit.", "info");
      }
    }
  }, [showToast]);

  // Main Queue Loop: Controls robot state and targets
  useEffect(() => {
    if (robotState === "idle" && queue.length > 0) {
      // Dequeue next command
      const item = queue[0];
      setQueue((prev) => prev.slice(1));
      setActiveQueueItem(item);
      setRobotState("moving");

      setRobot((prev) => {
        // Adjust heading direction towards target
        const dx = item.col - prev.x;
        const dy = item.row - prev.y;
        let heading = prev.heading;
        if (Math.abs(dx) > 0.1 || Math.abs(dy) > 0.1) {
          heading = Math.atan2(dy, dx);
        }
        return {
          ...prev,
          x: item.col,
          y: item.row,
          heading,
          speedMultiplier: item.adverb === "please" ? 3.0 : 1.0, // 3x speed boost
        };
      });
    }
  }, [queue, robotState]);

  // Position interpolation / updates in requestAnimationFrame
  useEffect(() => {
    let animId: number;

    const updateRobot = () => {
      setRobot((prev) => {
        const speed = 0.08 * prev.speedMultiplier; // Speed base per frame
        const dx = prev.x - prev.currentX;
        const dy = prev.y - prev.currentY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > 0.02 && robotState === "moving") {
          // Move closer
          const ratio = Math.min(1, speed / dist);
          return {
            ...prev,
            currentX: prev.currentX + dx * ratio,
            currentY: prev.currentY + dy * ratio,
          };
        } else if (robotState === "moving") {
          // Reached target
          setTimeout(() => {
            if (activeQueueItem) {
              executePaintCommand(activeQueueItem);
            }
            setRobotState("idle");
            setActiveQueueItem(null);
          }, 150); // Small stamp latency delay
          setRobotState("busy");
          return {
            ...prev,
            currentX: prev.x,
            currentY: prev.y,
          };
        }
        return prev;
      });

      // Update particles
      particlesRef.current = particlesRef.current
        .map((p) => ({
          ...p,
          x: p.x + p.vx,
          y: p.y + p.vy,
          alpha: p.alpha - 0.02,
        }))
        .filter((p) => p.alpha > 0);

      animId = requestAnimationFrame(updateRobot);
    };

    animId = requestAnimationFrame(updateRobot);
    return () => cancelAnimationFrame(animId);
  }, [robotState, activeQueueItem, executePaintCommand]);

  // Strict grammar parsing validation
  const validateAndParseCommand = (rawText: string): QueueItem | string => {
    const tokens = rawText.trim().toLowerCase().split(/\s+/);

    if (tokens.length !== 6) {
      return `Syntax Error: Grid commands must contain exactly 6 words. [Got ${tokens.length}]`;
    }

    const [v, c, p, l, d, a] = tokens;

    // 1. Verb
    if (v !== "place" && v !== "lay" && v !== "bin") {
      return `Syntax Error: Unknown Verb '${v}'. Allowed: place | lay | bin`;
    }

    // 2. Color
    if (c !== "blue" && c !== "green" && c !== "red" && c !== "white") {
      return `Syntax Error: Unknown Color '${c}'. Allowed: blue | green | red | white`;
    }

    // 3. Preposition
    if (p !== "at" && p !== "by" && p !== "in" && p !== "with") {
      return `Syntax Error: Unknown Preposition '${p}'. Allowed: at | by | in | with`;
    }

    // 4. Letter
    const upperL = l.toUpperCase();
    const colIdx = COLUMNS.indexOf(upperL);
    if (colIdx === -1) {
      if (upperL === "W") {
        return `Syntax Error: Column 'W' is excluded from grid coordinates.`;
      }
      return `Syntax Error: Unknown Column Letter '${l}'. Allowed: A to Z (excluding W)`;
    }

    // 5. Digit
    if (DIGIT_WORDS[d] === undefined) {
      return `Syntax Error: Unknown Row Digit '${d}'. Allowed: 0 to 9 (or zero to nine)`;
    }
    const rowIdx = DIGIT_WORDS[d];
    if (rowIdx < 0 || rowIdx > 9) {
      return `Syntax Error: Digit '${d}' out of bounds. Must map to row 0-9.`;
    }

    // 6. Adverb
    if (a !== "again" && a !== "now" && a !== "please" && a !== "soon") {
      return `Syntax Error: Unknown Adverb '${a}'. Allowed: again | now | please | soon`;
    }

    return {
      id: Math.random().toString(36).substr(2, 9),
      verb: v as "place" | "lay" | "bin",
      color: c as "blue" | "green" | "red" | "white",
      preposition: p as "at" | "by" | "in" | "with",
      col: colIdx,
      row: rowIdx,
      adverb: a as "again" | "now" | "please" | "soon",
      originalText: rawText.trim()
    };
  };

  // Run a verified command object
  const runCommand = useCallback((cmd: QueueItem) => {
    // Start stopwatch if not already
    startStopwatch();

    if (cmd.adverb === "now") {
      // Immediate hijack
      setQueue([]);
      setActiveQueueItem(null);
      setRobotState("busy");
      setRobot((prev) => ({
        ...prev,
        x: cmd.col,
        y: cmd.row,
        currentX: cmd.col,
        currentY: cmd.row,
        speedMultiplier: 1.0,
      }));
      // Paint immediately
      executePaintCommand(cmd);
      setRobotState("idle");
    } else {
      // Queue soon / please / again
      setQueue((prev) => [...prev, cmd]);
    }
  }, [startStopwatch, executePaintCommand]);

  // Handle text input submission
  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    const result = validateAndParseCommand(inputText);
    if (typeof result === "string") {
      showToast(result, "error");
    } else {
      runCommand(result);
      setInputText("");
    }
  };

  // Render main Canvas Grid
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    // Set actual canvas resolution according to retina/high-dpi display ratio
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const padLeft = 30;
    const padTop = 20;
    const cellW = 30;
    const cellH = 30;

    ctx.clearRect(0, 0, rect.width, rect.height);

    // 1. Draw Grid Cell Backgrounds & Gridlines
    for (let r = 0; r < ROWS_COUNT; r++) {
      for (let c = 0; c < COLS_COUNT; c++) {
        const x = padLeft + c * cellW;
        const y = padTop + r * cellH;

        // Subtle light border
        ctx.strokeStyle = "#e2e8f0";
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cellW, cellH);

        // Light background fill for cells
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(x + 1, y + 1, cellW - 2, cellH - 2);
      }
    }

    // Draw Chess-like Row Numbers (0-9) on Left and Right borders
    ctx.fillStyle = "#64748b"; // slate-500
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (let r = 0; r < ROWS_COUNT; r++) {
      const y = padTop + r * cellH + cellH / 2;
      ctx.fillText(String(r), 15, y); // Left border
      ctx.fillText(String(r), 795, y); // Right border
    }

    // Draw Chess-like Column Letters (A-Z, skipping W) on Top and Bottom borders
    for (let c = 0; c < COLS_COUNT; c++) {
      const x = padLeft + c * cellW + cellW / 2;
      ctx.fillText(COLUMNS[c], x, 10); // Top border
      ctx.fillText(COLUMNS[c], x, 330); // Bottom border
    }

    // 2. Draw ghost blueprint overlay (semi-transparent holographic shapes)
    for (let r = 0; r < ROWS_COUNT; r++) {
      for (let c = 0; c < COLS_COUNT; c++) {
        const b = activeBlueprintMatrix[r][c];
        if (b.type !== "empty" && b.color) {
          const x = padLeft + c * cellW;
          const y = padTop + r * cellH;
          const pad = 2;
          const w = cellW - 2 * pad;
          const h = cellH - 2 * pad;

          ctx.save();
          if (b.type === "flat") {
            // Semi-transparent dotted outline flat tile
            ctx.fillStyle = `${COLOR_MAP[b.color]}1a`; // 10% opacity
            ctx.fillRect(x + pad, y + pad, w, h);
            ctx.strokeStyle = COLOR_MAP[b.color];
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);
            ctx.strokeRect(x + pad, y + pad, w, h);
          } else if (b.type === "block") {
            // Oblique projection layout for ghost block
            const offset = 6;
            ctx.translate(x + pad, y + pad);

            // Draw ghost 3D block
            ctx.strokeStyle = COLOR_MAP[b.color];
            ctx.lineWidth = 1;
            ctx.setLineDash([2, 2]);

            // Front face outline
            ctx.strokeRect(0, 0, w - offset, h - offset);

            // Top face outline
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(offset, -offset);
            ctx.lineTo(w, -offset);
            ctx.lineTo(w - offset, 0);
            ctx.closePath();
            ctx.stroke();

            // Right face outline
            ctx.beginPath();
            ctx.moveTo(w - offset, 0);
            ctx.lineTo(w, -offset);
            ctx.lineTo(w, h - offset);
            ctx.lineTo(w - offset, h);
            ctx.closePath();
            ctx.stroke();
          }
          ctx.restore();
        }
      }
    }

    // 3. Draw User Grid Placed Blocks/Tiles
    for (let r = 0; r < ROWS_COUNT; r++) {
      for (let c = 0; c < COLS_COUNT; c++) {
        const cell = userGrid[r][c];
        if (cell.type !== "empty" && cell.color) {
          const x = padLeft + c * cellW;
          const y = padTop + r * cellH;
          const pad = 2;
          const w = cellW - 2 * pad;
          const h = cellH - 2 * pad;

          if (cell.type === "flat") {
            // Draw flat painted tile
            ctx.fillStyle = COLOR_MAP[cell.color];
            ctx.fillRect(x + pad, y + pad, w, h);

            // Add simple inner glow styling
            ctx.strokeStyle = "#e2e8f0";
            ctx.lineWidth = 1;
            ctx.strokeRect(x + pad, y + pad, w, h);
          } else if (cell.type === "block") {
            // Draw 2.5D solid block using Oblique Cabinet Projection
            const offset = 6;
            const blockColor = COLOR_MAP[cell.color];
            const shadowColor = COLOR_SHADOW_MAP[cell.color];
            const lightColor = COLOR_LIGHT_MAP[cell.color];

            ctx.save();
            ctx.translate(x + pad, y + pad + offset);

            const fw = w - offset;
            const fh = h - offset;

            // Side Right Face
            ctx.fillStyle = shadowColor;
            ctx.beginPath();
            ctx.moveTo(fw, 0);
            ctx.lineTo(fw + offset, -offset);
            ctx.lineTo(fw + offset, fh - offset);
            ctx.lineTo(fw, fh);
            ctx.closePath();
            ctx.fill();

            // Top Face
            ctx.fillStyle = lightColor;
            ctx.beginPath();
            ctx.moveTo(0, 0);
            ctx.lineTo(offset, -offset);
            ctx.lineTo(fw + offset, -offset);
            ctx.lineTo(fw, 0);
            ctx.closePath();
            ctx.fill();

            // Front Face
            ctx.fillStyle = blockColor;
            ctx.fillRect(0, 0, fw, fh);

            // Subtle border outline
            ctx.strokeStyle = "rgba(0, 0, 0, 0.15)";
            ctx.lineWidth = 1;
            ctx.strokeRect(0, 0, fw, fh);

            ctx.restore();
          }
        }
      }
    }

    // 4. Draw Particles
    particlesRef.current.forEach((p) => {
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    // 5. Draw GRID Droid Robot
    const rx = padLeft + robot.currentX * cellW + cellW / 2;
    const ry = padTop + robot.currentY * cellH + cellH / 2;

    ctx.save();
    ctx.translate(rx, ry);

    // Glowing base/pulse
    const pulseRadius = 14 + Math.sin(Date.now() / 120) * 2;
    const grad = ctx.createRadialGradient(0, 0, 4, 0, 0, pulseRadius);
    grad.addColorStop(0, "rgba(255, 170, 0, 0.8)");
    grad.addColorStop(0.5, "rgba(255, 170, 0, 0.3)");
    grad.addColorStop(1, "rgba(255, 170, 0, 0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(0, 0, pulseRadius, 0, Math.PI * 2);
    ctx.fill();

    // Droid Body
    ctx.rotate(robot.heading);
    ctx.fillStyle = "#ffaa00"; // robot primary yellow-orange
    ctx.beginPath();
    ctx.arc(0, 0, 9, 0, Math.PI * 2);
    ctx.fill();

    // Robot visor/heading pointer direction
    ctx.fillStyle = "#000000";
    ctx.beginPath();
    ctx.moveTo(3, -5);
    ctx.lineTo(10, 0);
    ctx.lineTo(3, 5);
    ctx.closePath();
    ctx.fill();

    // Glowing core
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(-2, 0, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }, [userGrid, robot, activeBlueprintMatrix]);

  // START Webcam A.I. stream
  const startWebcam = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240, frameRate: 25 },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setWebcamActive(true);
      setAiOfflineFallback(false);
      showToast("A.I. Camera Tracker connected.", "success");
    } catch (err) {
      console.warn("Could not access camera feed:", err);
      setAiOfflineFallback(true);
      showToast("Camera access rejected. AI simulation fallback activated.", "info");
    }
  };

  // STOP Webcam A.I. stream
  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setWebcamActive(false);
    setRecordingState("idle");
    setRecordingProgress(0);

    if (timerStarted) {
      setTimerStarted(false);
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
      setShowWinModal(true);
      showToast(`GRID-Bot Sandbox session finished with ${accuracy}% match.`, "success");
    }
  };

  // Toggle Webcam Camera Widget
  const toggleWebcam = () => {
    if (webcamActive || aiOfflineFallback) {
      stopWebcam();
      setAiOfflineFallback(false);
    } else {
      startWebcam();
    }
  };

  // State error trackers
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // START recording 3s video clip command
  const startRecording = () => {
    if (recordingState !== "idle") return;

    recordedChunksRef.current = [];
    setRecordingProgress(0);
    setRecordingState("recording");

    if (aiOfflineFallback) {
      // Mock recording behavior
      let progress = 0;
      recordIntervalRef.current = setInterval(() => {
        progress += 10;
        setRecordingProgress(progress);
        if (progress >= 100) {
          clearInterval(recordIntervalRef.current!);
          simulateOfflineInference();
        }
      }, 300);
      return;
    }

    if (!streamRef.current) {
      showToast("Webcam stream is not active.", "error");
      setRecordingState("idle");
      return;
    }

    try {
      let selectedMimeType = "";
      const mimeTypes = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
        "video/mp4;codecs=h264",
        "video/mp4",
      ];
      for (const mime of mimeTypes) {
        if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(mime)) {
          selectedMimeType = mime;
          break;
        }
      }

      let recorder: MediaRecorder;
      if (selectedMimeType) {
        recorder = new MediaRecorder(streamRef.current, { mimeType: selectedMimeType });
      } else {
        recorder = new MediaRecorder(streamRef.current);
      }

      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        setRecordingState("analyzing");
        const mimeType = recorder.mimeType || "video/webm";
        const blob = new Blob(recordedChunksRef.current, { type: mimeType });
        const extension = mimeType.includes("mp4") ? "mp4" : "webm";
        const file = new File([blob], `recorded_command.${extension}`, { type: mimeType });
        await uploadAndInference(file);
      };

      recorder.start();

      let progress = 0;
      recordIntervalRef.current = setInterval(() => {
        progress += 10;
        setRecordingProgress(progress);
        if (progress >= 100) {
          clearInterval(recordIntervalRef.current!);
          if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
            mediaRecorderRef.current.stop();
          }
        }
      }, 300);

    } catch (e) {
      showToast("Failed to initialize MediaRecorder: " + String(e), "error");
      setRecordingState("idle");
    }
  };

  // Upload Recorded Video to FastAPI Offline Inference Model
  const uploadAndInference = async (file: File) => {
    try {
      // Best Model used: gap_proj conformer lite ctc
      const response = await analyzeVideo({
        file,
        modelPath: "checkpoints/best_ctc_model_conformer_lite_gap_proj.keras",
        decoderMode: "greedy_ctc",
        llmPostprocess: false,
      });

      const predictedSentence = response.predicted_text || "";

      if (!predictedSentence.trim()) {
        showToast("A.I. Lip-Reader could not read a command. Please speak clearly.", "error");
        setRecordingState("idle");
        return;
      }

      showToast(`A.I. Decoded: "${predictedSentence}"`, "success");
      setInputText(predictedSentence);

      const parsed = validateAndParseCommand(predictedSentence);
      if (typeof parsed === "string") {
        showToast(`Grammar Match Error: ${parsed}`, "error");
      } else {
        runCommand(parsed);
      }
      setRecordingState("idle");
    } catch (err) {
      console.error("API Lip-Reading inference failed:", err);
      showToast("FastAPI Offline Inference offline. Fallback simulation triggered.", "info");
      simulateOfflineInference();
    }
  };

  // Simulated Lip Reading when Server is offline
  const simulateOfflineInference = () => {
    setRecordingState("analyzing");
    setTimeout(() => {
      // Find what preset matching the blueprint is not built yet to help the user complete it!
      const levelPresets = BLUEPRINTS[activeLevel].presets;
      const randomPreset = levelPresets[Math.floor(Math.random() * levelPresets.length)];

      showToast(`[SIMULATOR] Decoded Lip movements to: "${randomPreset}"`, "success");
      setInputText(randomPreset);

      const parsed = validateAndParseCommand(randomPreset);
      if (typeof parsed !== "string") {
        runCommand(parsed);
      }
      setRecordingState("idle");
    }, 1500);
  };

  // Keyboard shortcut preset execution helper
  const handlePresetClick = (presetText: string) => {
    setInputText(presetText);
    const parsed = validateAndParseCommand(presetText);
    if (typeof parsed !== "string") {
      runCommand(parsed);
      setInputText("");
    } else {
      showToast(parsed, "error");
    }
  };

  return (
    <div className="flex-1 flex flex-col w-full overflow-x-hidden p-6 max-w-6xl mx-auto">
      {/* Header section inside main layout container */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4 w-full">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-foreground flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-accent"></span>
            </span>
            GRID-Bot Lip-Sync Sandbox
          </h1>
          <p className="text-sm text-muted">
            A.I. Lip-Reading Robotic Showcase Terminal
          </p>
        </div>

        {/* Preset Blueprint Selectors */}
        <div className="flex gap-2">
          {(Object.keys(BLUEPRINTS) as Array<keyof typeof BLUEPRINTS>).map((key) => (
            <button
              key={key}
              onClick={() => loadLevel(key)}
              className={`rounded-lg px-4 py-2 text-xs font-bold uppercase tracking-wider transition-all border ${
                activeLevel === key
                  ? "bg-accent text-white border-accent shadow-sm"
                  : "bg-card text-muted border-border hover:border-muted/30 hover:text-foreground"
              }`}
            >
              {BLUEPRINTS[key].name}
            </button>
          ))}
        </div>
      </div>

      {/* Main split dashboard panel */}
      <main className="flex-1 w-full mt-6 grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
        {/* Left Side Sidebar Workspace */}
        <section className="space-y-6">
          {/* AI Lip-Tracker Camera Simulator */}
          <div className="rounded-xl border border-border bg-card p-5 shadow-sm flex flex-col">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wider mb-3 flex items-center justify-between">
              A.I. Lip-Tracker Camera Feed
              <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-mono ${
                webcamActive
                  ? "bg-accent/15 text-accent border border-accent/20"
                  : aiOfflineFallback
                  ? "bg-yellow-100 text-yellow-700 border border-yellow-200"
                  : "bg-red-100 text-red-700 border border-red-200"
              }`}>
                {webcamActive ? "CAMERA LIVE" : aiOfflineFallback ? "SIMULATOR ACTIVE" : "OFFLINE"}
              </span>
            </h3>

            {/* Simulated Tracking Feed Frame */}
            <div className="relative aspect-[4/3] rounded-lg bg-black overflow-hidden border border-border flex items-center justify-center">
              {webcamActive && !aiOfflineFallback ? (
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover scale-x-[-1]"
                />
              ) : aiOfflineFallback ? (
                <div className="text-center p-4">
                  <div className="w-12 h-12 rounded-full border-4 border-yellow-500 border-t-transparent animate-spin mx-auto mb-2" />
                  <p className="text-xs font-mono text-yellow-600">Camera permission denied.</p>
                  <p className="text-[10px] text-muted mt-1">Manual Simulation Mode active.</p>
                </div>
              ) : (
                <div className="text-center p-6 text-muted font-mono text-xs">
                  <p>A.I. Scan Engine Offline</p>
                  <p className="text-[10px] mt-1 text-muted/80">Activate camera feed to speak commands</p>
                </div>
              )}

              {/* Wireframe Scanning HUD Overlay */}
              {(webcamActive || aiOfflineFallback) && (
                <div className="absolute inset-0 pointer-events-none border border-accent/30 flex flex-col justify-between p-3">
                  {/* Glowing Corners */}
                  <div className="flex justify-between w-full">
                    <div className="w-3 h-3 border-t-2 border-l-2 border-accent" />
                    <div className="w-3 h-3 border-t-2 border-r-2 border-accent" />
                  </div>

                  {/* Pulsating Dashed Mouth Outline */}
                  <div className="self-center flex flex-col items-center gap-1.5 opacity-80">
                    <div className="w-24 h-10 border-2 border-dashed border-accent rounded-[50%/70%_70%_30%_30%] animate-pulse" />
                    <span className="text-[9px] font-mono tracking-widest text-accent bg-black/60 px-1 rounded">
                      LIP TARGET LOCKED
                    </span>
                  </div>

                  {/* Scanning Horizontal Line */}
                  <div className="absolute left-0 right-0 h-[1.5px] bg-red-500/50 shadow-[0_0_8px_rgba(239,68,68,0.5)] animate-[bounce_2.5s_infinite_ease-in-out]" />

                  {/* Details Overlay */}
                  <div className="flex justify-between w-full items-end">
                    <div className="w-3 h-3 border-b-2 border-l-2 border-accent" />
                    <span className="text-[8px] font-mono text-accent tracking-tighter bg-black/60 px-1 rounded">
                      {recordingState === "recording"
                        ? "STATUS: RECORDING..."
                        : recordingState === "analyzing"
                        ? "STATUS: READING IN PROGRESS..."
                        : "STATUS: READY"}
                    </span>
                    <div className="w-3 h-3 border-b-2 border-r-2 border-accent" />
                  </div>
                </div>
              )}
            </div>

            {/* Webcam / Recording Actions */}
            <div className="mt-4 flex gap-2">
              <button
                onClick={toggleWebcam}
                className={`flex-1 rounded-lg py-2 text-xs font-bold border transition-colors ${
                  webcamActive || aiOfflineFallback
                    ? "bg-red-500/10 text-red-500 border-red-500/30 hover:bg-red-500/20"
                    : "bg-accent/10 text-accent border-accent/30 hover:bg-accent/20"
                }`}
              >
                {webcamActive || aiOfflineFallback ? "Stop Camera" : "Start Camera"}
              </button>

              <button
                onClick={startRecording}
                disabled={!(webcamActive || aiOfflineFallback) || recordingState !== "idle"}
                className={`flex-1 rounded-lg py-2 text-xs font-bold border transition-colors flex items-center justify-center gap-1.5 ${
                  recordingState === "recording"
                    ? "bg-red-600 text-white border-red-600 animate-pulse"
                    : recordingState === "analyzing"
                    ? "bg-yellow-50 text-yellow-700 border-yellow-200 cursor-not-allowed"
                    : "bg-accent text-white border-accent hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed"
                }`}
              >
                {recordingState === "recording" ? (
                  <>
                    <span className="h-2 w-2 rounded-full bg-white animate-ping" />
                    Recording ({Math.round(recordingProgress)}%)
                  </>
                ) : recordingState === "analyzing" ? (
                  <>Analyzing Lip-Sync...</>
                ) : (
                  <>Record Command (3s)</>
                )}
              </button>
            </div>
          </div>

          {/* Rigid Vocabulary Glossary Card */}
          <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-3 font-mono">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wider border-b border-border pb-2">
              Rigid Grammar Tokens
            </h3>
            <p className="text-[10px] text-muted">
              Input strings must follow this strict 6-token sequence:
            </p>
            <div className="text-xs bg-background p-2.5 rounded border border-border space-y-1">
              <p className="text-accent">Verb → Color → Preposition → Letter → Digit → Adverb</p>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px] text-muted pt-1">
              <div>
                <span className="text-foreground font-bold">Verb:</span> place, lay, bin
              </div>
              <div>
                <span className="text-foreground font-bold">Color:</span> blue, green, red, white
              </div>
              <div>
                <span className="text-foreground font-bold">Prep:</span> at, by, in, with
              </div>
              <div>
                <span className="text-foreground font-bold">Letter:</span> A to Z (no W)
              </div>
              <div>
                <span className="text-foreground font-bold">Digit:</span> 0 to 9 (or words)
              </div>
              <div>
                <span className="text-foreground font-bold">Adverb:</span> again, now, please, soon
              </div>
            </div>
          </div>

          {/* Interactive Preset Controls */}
          <div className="rounded-xl border border-border bg-card p-5 shadow-sm space-y-3">
            <h3 className="text-xs font-semibold text-muted uppercase tracking-wider border-b border-border pb-2">
              Level Presets ({BLUEPRINTS[activeLevel].name})
            </h3>
            <p className="text-[10px] text-muted">
              Click a preset command below to execute immediately:
            </p>
            <div className="max-h-40 overflow-y-auto space-y-1.5 pr-1">
              {BLUEPRINTS[activeLevel].presets.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => handlePresetClick(preset)}
                  className="w-full text-left font-mono text-[11px] px-2.5 py-1.5 bg-background border border-border hover:border-accent rounded text-muted hover:text-accent transition-all truncate"
                >
                  &gt; {preset}
                </button>
              ))}
            </div>
          </div>

        </section>

        {/* Right Side Main HUD Panel */}
        <section className="flex flex-col gap-6">
          {/* Status Metrics HUD Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="rounded-xl border border-border bg-card p-4 text-center shadow-sm">
              <p className="text-[10px] text-muted font-mono uppercase tracking-wider">Sync Accuracy</p>
              <h2 className="text-2xl font-black mt-1 font-mono text-accent">
                {accuracy}%
              </h2>
            </div>
            <div className="rounded-xl border border-border bg-card p-4 text-center shadow-sm">
              <p className="text-[10px] text-muted font-mono uppercase tracking-wider">Robot Position</p>
              <h2 className="text-2xl font-black mt-1 font-mono text-green-600">
                {COLUMNS[robot.x]}, {robot.y}
              </h2>
            </div>
            <div className="rounded-xl border border-border bg-card p-4 text-center shadow-sm">
              <p className="text-[10px] text-muted font-mono uppercase tracking-wider">Showcase Timer</p>
              <h2 className="text-2xl font-black mt-1 font-mono text-yellow-600">
                {Math.floor(elapsedTime / 60).toString().padStart(2, "0")}
                :{(Math.floor(elapsedTime) % 60).toString().padStart(2, "0")}
                .{(Math.round((elapsedTime % 1) * 10)).toString()}
              </h2>
            </div>
            <div className="rounded-xl border border-border bg-card p-4 text-center shadow-sm">
              <p className="text-[10px] text-muted font-mono uppercase tracking-wider">Queue Status</p>
              <h2 className={`text-xl font-black mt-1.5 font-mono uppercase ${
                robotState === "idle" ? "text-muted" : robotState === "moving" ? "text-accent" : "text-yellow-600 animate-pulse"
              }`}>
                {robotState}
              </h2>
            </div>
          </div>

          {/* Interactive HTML5 Canvas Container */}
          <div className="rounded-xl border border-border bg-card p-5 shadow-sm flex flex-col items-center justify-center flex-1">
            <div className="w-full flex justify-between items-center mb-3">
              <p className="text-xs text-muted">
                Blueprint target overlay is visible as transparent dashed shapes. Build on top of it.
              </p>
              <button
                onClick={() => {
                  setUserGrid(
                    Array.from({ length: ROWS_COUNT }, () =>
                      Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null }))
                    )
                  );
                  showToast("Cleared user build grid.", "info");
                }}
                className="text-[10px] text-red-500 hover:text-red-600 font-mono"
              >
                [Clear Grid]
              </button>
            </div>

            <div className="w-full max-w-full overflow-x-auto p-1 bg-background rounded-lg border border-border">
              <div className="min-w-[810px]">
                {/* Canvas styled 810px width, 340px height */}
                <canvas
                  ref={canvasRef}
                  style={{ width: "810px", height: "340px" }}
                  className="block select-none mx-auto"
                />
              </div>
            </div>
            {/* Grid system hint label */}
            <p className="text-[10px] font-mono text-muted mt-2 select-none text-center">
              GRID Coordinate Target System: Column A to Z (no W) · Row 0 to 9
            </p>
          </div>

          {/* Pending Execution Queue visualizer */}
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <h4 className="text-[10px] font-bold text-muted uppercase tracking-wider mb-2 font-mono flex justify-between">
              Execution Action Queue ({queue.length + (activeQueueItem ? 1 : 0)} pending)
              {queue.length > 0 && (
                <button
                  onClick={() => {
                    setQueue([]);
                    showToast("Cleared execution queue.", "info");
                  }}
                  className="text-red-500 hover:text-red-600 hover:underline"
                >
                  Clear Queue
                </button>
              )}
            </h4>
            <div className="flex gap-2.5 overflow-x-auto py-1 min-h-[62px]">
              {activeQueueItem && (
                <div className="shrink-0 font-mono text-[10px] px-3 py-2 bg-accent/10 border border-accent/30 text-accent rounded flex flex-col justify-between max-w-[150px] animate-pulse">
                  <span className="font-bold truncate">&gt; {activeQueueItem.verb.toUpperCase()} {activeQueueItem.color}</span>
                  <span className="text-[9px] text-muted mt-1">{COLUMNS[activeQueueItem.col]}{activeQueueItem.row} ({activeQueueItem.adverb})</span>
                </div>
              )}
              {queue.map((item, idx) => (
                <div
                  key={item.id}
                  className="shrink-0 font-mono text-[10px] px-3 py-2 bg-background border border-border rounded flex flex-col justify-between max-w-[150px]"
                >
                  <span className="text-foreground truncate">&gt; {item.verb} {item.color}</span>
                  <span className="text-[9px] text-muted mt-1">{COLUMNS[item.col]}{item.row} ({item.adverb})</span>
                </div>
              ))}
              {!activeQueueItem && queue.length === 0 && (
                <p className="text-xs text-muted italic self-center py-2 w-full text-center">
                  Robot is idle. Submit a command below.
                </p>
              )}
            </div>
          </div>

          {/* Command input terminal */}
          <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
            <form onSubmit={handleTextSubmit} className="flex gap-3">
              <span className="text-accent font-mono text-lg self-center select-none font-bold">&gt;</span>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="place blue at M 5 now | lay green with E 8 soon"
                className="flex-1 bg-background border border-border rounded-lg px-4 py-2.5 text-sm font-mono text-foreground placeholder-muted outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/30"
              />
              <button
                type="submit"
                className="bg-accent text-white px-6 py-2.5 rounded-lg text-sm font-bold uppercase tracking-wider hover:bg-accent-hover transition-all font-mono"
              >
                Send
              </button>
            </form>
            <p className="text-[9px] text-muted mt-2 font-mono">
              Tip: Type a command and press Enter, or record a video command using your camera.
            </p>
          </div>
        </section>
      </main>

      {/* Floating Stack Toasts Notifications */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-2 max-w-sm pointer-events-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto p-4 rounded-lg shadow-xl border font-mono text-xs flex flex-col gap-1.5 transition-all duration-300 transform translate-y-0 ${
              toast.type === "error"
                ? "bg-red-50 border border-red-200 text-red-600"
                : toast.type === "success"
                ? "bg-green-50 border border-green-200 text-green-600"
                : "bg-blue-50 border border-blue-200 text-blue-600"
            }`}
          >
            <p className="font-bold uppercase">
              {toast.type === "error" ? "🚨 SYNTAX ERROR" : toast.type === "success" ? "⚡ ROBOT PIPELINE" : "⚙️ SYSTEM LOG"}
            </p>
            <p>{toast.message}</p>
          </div>
        ))}
      </div>

      {/* Win / Session Finished Modal */}
      {showWinModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-card border border-border rounded-xl max-w-md w-full p-6 shadow-xl text-center space-y-4">
            <h2 className="text-2xl font-black text-accent tracking-wider font-mono">
              {accuracy === 100 ? "🎉 BLUEPRINT SYNC SUCCESS!" : "🏁 SESSION FINISHED!"}
            </h2>
            <p className="text-sm text-muted">
              {accuracy === 100
                ? `Congratulations! You fully completed the building blueprint for "${BLUEPRINTS[activeLevel].name}"!`
                : `Session ended. You built ${accuracy}% of the blueprint for "${BLUEPRINTS[activeLevel].name}".`}
            </p>
            <div className="flex justify-center gap-4">
              <div className="p-4 bg-background rounded-lg border border-border inline-block min-w-[120px]">
                <p className="text-xs text-muted font-mono">Sync Accuracy</p>
                <h3 className="text-xl font-black text-accent font-mono">{accuracy}%</h3>
              </div>
              <div className="p-4 bg-background rounded-lg border border-border inline-block min-w-[120px]">
                <p className="text-xs text-muted font-mono">Elapsed Time</p>
                <h3 className="text-xl font-black text-green-600 font-mono">{elapsedTime.toFixed(1)}s</h3>
              </div>
            </div>

            <div className="flex gap-2 justify-center pt-2">
              <button
                type="button"
                onClick={() => {
                  setUserGrid(
                    Array.from({ length: ROWS_COUNT }, () =>
                      Array.from({ length: COLS_COUNT }, () => ({ type: "empty", color: null }))
                    )
                  );
                  setElapsedTime(0);
                  setShowWinModal(false);
                }}
                className="bg-transparent text-muted border border-border hover:border-muted px-4 py-2 rounded-lg text-xs font-bold font-mono"
              >
                Reset Grid
              </button>
              <button
                type="button"
                onClick={() => setShowWinModal(false)}
                className="bg-accent text-white px-6 py-2 rounded-lg text-xs font-bold font-mono hover:bg-accent-hover"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
