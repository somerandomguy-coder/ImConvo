import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ScoreBadge } from "../ScoreBadge";

describe("ScoreBadge", () => {
  it("shows the score", () => {
    render(<ScoreBadge score={12} />);
    expect(screen.getByText("12")).toBeInTheDocument();
  });
});
