import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useRef } from "react";
import { CorridorScene } from "../CorridorScene";

function Harness() {
  const ref = useRef<HTMLVideoElement>(null);
  return (
    <CorridorScene
      videoRef={ref}
      words={["bin", "blue", "place", "at", "f", "now"]}
      slots={["command", "color", "preposition", "letter", "digit", "adverb"]}
      roundIndex={2}
    />
  );
}

describe("CorridorScene", () => {
  it("shows the current word and slot on the front wall", () => {
    render(<Harness />);
    expect(screen.getByText("place")).toBeInTheDocument();
    expect(screen.getByText("preposition")).toBeInTheDocument();
  });
  it("renders a video backdrop", () => {
    const { container } = render(<Harness />);
    expect(container.querySelector("video")).not.toBeNull();
  });
});
