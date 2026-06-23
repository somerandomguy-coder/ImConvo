import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MuteToggle } from "../MuteToggle";

describe("MuteToggle", () => {
  it("labels by state and fires onToggle", () => {
    const onToggle = vi.fn();
    render(<MuteToggle muted={false} onToggle={onToggle} />);
    const btn = screen.getByLabelText("Mute");
    fireEvent.click(btn);
    expect(onToggle).toHaveBeenCalledOnce();
  });
  it("shows Unmute when muted", () => {
    render(<MuteToggle muted={true} onToggle={() => {}} />);
    expect(screen.getByLabelText("Unmute")).toBeInTheDocument();
  });
});
