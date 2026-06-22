import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

export type ServerMessage =
  | { type: "progress"; word: string; confidence: number; topk: { word: string; confidence: number }[]; face: boolean }
  | { type: "result"; pass: boolean; word: string; confidence: number; target: string }
  | { type: "error"; message: string }
  | { type: "pong" };

export function wsUrlFromApi(apiBase: string): string {
  const proto = apiBase.startsWith("https") ? "wss" : "ws";
  const host = apiBase.replace(/^https?:\/\//, "").replace(/\/$/, "");
  return `${proto}://${host}/ws/game`;
}

export function useGameSocket(onMessage: (msg: ServerMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  useEffect(() => {
    let closed = false;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      const ws = new WebSocket(wsUrlFromApi(API_BASE));
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retry = setTimeout(connect, 1000);
      };
      ws.onmessage = (e) => {
        try {
          onMessageRef.current(JSON.parse(e.data) as ServerMessage);
        } catch {
          /* ignore malformed */
        }
      };
    };
    connect();

    return () => {
      closed = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  const send = useCallback((payload: object) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
  }, []);

  const startRound = useCallback((slotIndex: number, target: string) => send({ type: "start_round", slot_index: slotIndex, target }), [send]);
  const sendFrame = useCallback((b64: string) => send({ type: "frame", data: b64 }), [send]);
  const endRound = useCallback(() => send({ type: "end_round" }), [send]);

  return { connected, startRound, sendFrame, endRound };
}
