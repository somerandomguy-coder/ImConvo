import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TimerPill } from "../TimerPill";

describe("TimerPill", () => {
  it("formats seconds as 0:0X", () => {
    render(<TimerPill secondsLeft={6} />);
    expect(screen.getByText("0:06")).toBeInTheDocument();
  });
  it("adds the low-time class under 2 seconds", () => {
    const { container } = render(<TimerPill secondsLeft={2} />);
    expect(container.querySelector(".timer-low")).not.toBeNull();
  });
  it("has no low-time class when there is time left", () => {
    const { container } = render(<TimerPill secondsLeft={5} />);
    expect(container.querySelector(".timer-low")).toBeNull();
  });
});
