import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProgressDots } from "../ProgressDots";

describe("ProgressDots", () => {
  it("marks passed, current and upcoming dots", () => {
    const { container } = render(<ProgressDots total={6} roundIndex={2} />);
    const dots = container.querySelectorAll("[data-state]");
    expect(dots).toHaveLength(6);
    expect(dots[0].getAttribute("data-state")).toBe("passed");
    expect(dots[1].getAttribute("data-state")).toBe("passed");
    expect(dots[2].getAttribute("data-state")).toBe("current");
    expect(dots[3].getAttribute("data-state")).toBe("upcoming");
  });
});
