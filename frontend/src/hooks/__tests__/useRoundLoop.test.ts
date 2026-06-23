import { renderHook, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRoundLoop } from "../useRoundLoop";

function makeParams(over = {}) {
  return {
    active: true,
    roundIndex: 0,
    target: "blue",
    roundSeconds: 6,
    frameIntervalMs: 66,
    startRound: vi.fn(),
    endRound: vi.fn(),
    sendFrame: vi.fn(),
    grabFrame: vi.fn(() => "JPEGDATA"),
    onPass: vi.fn(),
    onTimeout: vi.fn(),
    ...over,
  };
}

describe("useRoundLoop", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("auto-starts the round and streams frames while active", () => {
    const p = makeParams();
    renderHook(() => useRoundLoop(p));
    expect(p.startRound).toHaveBeenCalledWith(0, "blue");
    act(() => { vi.advanceTimersByTime(200); });
    expect(p.sendFrame).toHaveBeenCalledWith("JPEGDATA");
  });

  it("passes on a winning result and stops capturing", () => {
    const p = makeParams();
    const { result } = renderHook(() => useRoundLoop(p));
    act(() => result.current.handleMessage({ type: "result", pass: true, word: "blue", confidence: 0.9, target: "blue" }));
    expect(p.onPass).toHaveBeenCalledOnce();
    p.sendFrame.mockClear();
    act(() => { vi.advanceTimersByTime(300); });
    expect(p.sendFrame).not.toHaveBeenCalled();
  });

  it("re-arms (retries) on timeout without calling onPass", () => {
    const p = makeParams({ roundSeconds: 2 });
    renderHook(() => useRoundLoop(p));
    expect(p.startRound).toHaveBeenCalledTimes(1);
    act(() => { vi.advanceTimersByTime(2000); });
    expect(p.endRound).toHaveBeenCalled();
    expect(p.onTimeout).toHaveBeenCalled();
    expect(p.startRound).toHaveBeenCalledTimes(2);
    expect(p.onPass).not.toHaveBeenCalled();
  });

  it("updates the meter on progress", () => {
    const p = makeParams();
    const { result } = renderHook(() => useRoundLoop(p));
    act(() => result.current.handleMessage({ type: "progress", word: "blue", confidence: 0.5, topk: [], face: true }));
    expect(result.current.meter).toEqual({ word: "blue", confidence: 0.5, face: true });
  });

  it("stops capturing when active goes false", () => {
    const p = makeParams();
    const { rerender } = renderHook((props) => useRoundLoop(props), { initialProps: p });
    act(() => { vi.advanceTimersByTime(200); });
    expect(p.sendFrame).toHaveBeenCalled();
    p.sendFrame.mockClear();
    rerender({ ...p, active: false });
    act(() => { vi.advanceTimersByTime(300); });
    expect(p.sendFrame).not.toHaveBeenCalled();
  });
});
