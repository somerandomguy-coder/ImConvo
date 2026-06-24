import { render, screen, act } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { useRef } from "react";
import { MouthBox } from "../MouthBox";
import type { LipBBox } from "../../../hooks/useGameSocket";

function Harness({ bbox, label, videoWidth = 1280, videoHeight = 720, boxW = 1280, boxH = 720 }: {
  bbox: LipBBox | null;
  label?: string;
  videoWidth?: number;
  videoHeight?: number;
  boxW?: number;
  boxH?: number;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const setVideo = (el: HTMLVideoElement | null) => {
    videoRef.current = el;
    if (!el) return;
    Object.defineProperty(el, "videoWidth", { value: videoWidth, configurable: true });
    Object.defineProperty(el, "videoHeight", { value: videoHeight, configurable: true });
    if (el.parentElement) {
      Object.defineProperty(el.parentElement, "clientWidth", { value: boxW, configurable: true });
      Object.defineProperty(el.parentElement, "clientHeight", { value: boxH, configurable: true });
    }
  };
  return (
    <div style={{ position: "relative" }}>
      <video ref={setVideo} />
      <MouthBox bbox={bbox} label={label} videoRef={videoRef} />
    </div>
  );
}

describe("MouthBox", () => {
  it("renders nothing when bbox is null", () => {
    const { container } = render(<Harness bbox={null} />);
    expect(container.querySelector('[data-testid="mouth-box"]')).toBeNull();
  });

  it("positions the box correctly when video matches container aspect (no crop)", async () => {
    const { findByTestId } = render(
      <Harness bbox={{ x: 0.25, y: 0.4, w: 0.5, h: 0.2 }} videoWidth={1280} videoHeight={720} boxW={1280} boxH={720} />,
    );
    await act(async () => {});
    const el = await findByTestId("mouth-box");
    expect(el.style.left).toBe("320px");
    expect(el.style.top).toBe("288px");
    expect(el.style.width).toBe("640px");
    expect(el.style.height).toBe("144px");
  });

  it("accounts for object-fit: cover when container is wider than video (crops top/bottom)", async () => {
    const { findByTestId } = render(
      <Harness
        bbox={{ x: 0.5, y: 0.5, w: 0, h: 0 }}
        videoWidth={1280}
        videoHeight={720}
        boxW={1920}
        boxH={800}
      />,
    );
    await act(async () => {});
    const el = await findByTestId("mouth-box");
    expect(parseFloat(el.style.left)).toBeCloseTo(960, 0);
    expect(parseFloat(el.style.top)).toBeCloseTo(400, 0);
  });

  it("renders the label text", async () => {
    render(<Harness bbox={{ x: 0.1, y: 0.1, w: 0.1, h: 0.1 }} label="MOUTH 87%" />);
    await act(async () => {});
    expect(await screen.findByText("MOUTH 87%")).toBeInTheDocument();
  });
});
