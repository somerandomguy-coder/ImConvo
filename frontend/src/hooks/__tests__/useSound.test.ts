import { renderHook, act } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSound } from "../useSound";

describe("useSound", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.spyOn(window.HTMLMediaElement.prototype, "play").mockImplementation(() => Promise.resolve());
  });

  it("defaults to unmuted and plays a cue", () => {
    const playSpy = window.HTMLMediaElement.prototype.play as unknown as ReturnType<typeof vi.fn>;
    const { result } = renderHook(() => useSound());
    expect(result.current.muted).toBe(false);
    act(() => result.current.play("pass"));
    expect(playSpy).toHaveBeenCalled();
  });

  it("does not play when muted, and persists the toggle", () => {
    const { result } = renderHook(() => useSound());
    act(() => result.current.toggleMuted());
    expect(result.current.muted).toBe(true);
    expect(localStorage.getItem("imconvo.game.muted")).toBe("true");
    const playSpy = window.HTMLMediaElement.prototype.play as unknown as ReturnType<typeof vi.fn>;
    playSpy.mockClear();
    act(() => result.current.play("win"));
    expect(playSpy).not.toHaveBeenCalled();
  });
});
