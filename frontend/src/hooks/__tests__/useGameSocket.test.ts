import { describe, expect, it } from "vitest";
import { wsUrlFromApi } from "../useGameSocket";

describe("wsUrlFromApi", () => {
  it("maps http to ws and appends the game path", () => {
    expect(wsUrlFromApi("http://localhost:8001")).toBe("ws://localhost:8001/ws/game");
  });
  it("maps https to wss", () => {
    expect(wsUrlFromApi("https://api.example.com")).toBe("wss://api.example.com/ws/game");
  });
});
