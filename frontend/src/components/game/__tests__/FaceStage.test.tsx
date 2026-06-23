import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { useRef } from "react";
import { FaceStage } from "../FaceStage";
function H() { const r = useRef<HTMLVideoElement | null>(null); return <FaceStage videoRef={r}><span data-testid="kid">k</span></FaceStage>; }
describe("FaceStage", () => {
  it("renders a video and overlay children", () => {
    const { container, getByTestId } = render(<H />);
    expect(container.querySelector("video")).not.toBeNull();
    expect(getByTestId("kid")).toBeInTheDocument();
  });
});
