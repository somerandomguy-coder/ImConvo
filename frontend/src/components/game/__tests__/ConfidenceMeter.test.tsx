import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ConfidenceMeter } from "../ConfidenceMeter";

describe("ConfidenceMeter", () => {
  it("shows the face prompt when no face is detected", () => {
    render(<ConfidenceMeter confidence={0} word="" face={false} />);
    expect(screen.getByText(/center your face/i)).toBeInTheDocument();
  });

  it("renders the current guess and percentage when a face is present", () => {
    render(<ConfidenceMeter confidence={0.83} word="blue" face={true} />);
    expect(screen.getByText(/blue/)).toBeInTheDocument();
    expect(screen.getByText(/83%/)).toBeInTheDocument();
  });
});
