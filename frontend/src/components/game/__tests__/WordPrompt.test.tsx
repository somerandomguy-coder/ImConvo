import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { WordPrompt } from "../WordPrompt";
describe("WordPrompt", () => {
  it("renders slot uppercase and the word", () => {
    render(<WordPrompt slot="command" word="place" />);
    expect(screen.getByText("place")).toBeInTheDocument();
    expect(screen.getByText("command")).toHaveStyle({ textTransform: "uppercase" });
  });
});
