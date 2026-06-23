import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const fire = vi.fn();
vi.mock("canvas-confetti", () => ({ default: (...args: unknown[]) => fire(...args) }));

import { Confetti } from "../Confetti";

describe("Confetti", () => {
  it("does not fire on initial key 0", () => {
    render(<Confetti fireKey={0} />);
    expect(fire).not.toHaveBeenCalled();
  });
  it("fires when fireKey increases", () => {
    const { rerender } = render(<Confetti fireKey={0} />);
    rerender(<Confetti fireKey={1} />);
    expect(fire).toHaveBeenCalled();
  });
});
