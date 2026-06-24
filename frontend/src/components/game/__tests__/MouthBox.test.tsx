import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MouthBox } from "../MouthBox";

describe("MouthBox", () => {
  it("renders nothing when bbox is null", () => {
    const { container } = render(<MouthBox bbox={null} />);
    expect(container.querySelector('[data-testid="mouth-box"]')).toBeNull();
  });
  it("positions itself in percentages of the parent", () => {
    const { getByTestId } = render(<MouthBox bbox={{ x: 0.25, y: 0.4, w: 0.5, h: 0.2 }} />);
    const el = getByTestId("mouth-box");
    expect(el).toHaveStyle({ left: "25%", top: "40%", width: "50%", height: "20%" });
  });
  it("renders the label text when provided", () => {
    render(<MouthBox bbox={{ x: 0.1, y: 0.1, w: 0.1, h: 0.1 }} label="MOUTH 87%" />);
    expect(screen.getByText("MOUTH 87%")).toBeInTheDocument();
  });
});
