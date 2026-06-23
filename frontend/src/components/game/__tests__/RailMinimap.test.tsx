import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RailMinimap } from "../RailMinimap";
describe("RailMinimap", () => {
  it("renders 6 walls and tags them passed/current/upcoming", () => {
    const { container } = render(<RailMinimap total={6} roundIndex={2} />);
    const walls = container.querySelectorAll("[data-wall]");
    expect(walls.length).toBe(6);
    expect(walls[0].getAttribute("data-state")).toBe("passed");
    expect(walls[1].getAttribute("data-state")).toBe("passed");
    expect(walls[2].getAttribute("data-state")).toBe("current");
    expect(walls[3].getAttribute("data-state")).toBe("upcoming");
  });
  it("includes a candy car", () => {
    const { container } = render(<RailMinimap total={6} roundIndex={0} />);
    expect(container.querySelector("[data-car]")).not.toBeNull();
  });
});
