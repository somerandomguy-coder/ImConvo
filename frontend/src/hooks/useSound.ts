import { useCallback, useEffect, useRef, useState } from "react";

type Cue = "pass" | "timeout" | "win";
const MUTED_KEY = "imconvo.game.muted";
const MUSIC_KEY = "imconvo.game.music";

function readBool(key: string): boolean {
  if (typeof localStorage === "undefined") return false;
  return localStorage.getItem(key) === "true";
}

export function useSound() {
  const [muted, setMuted] = useState(false);
  const [musicOn, setMusicOn] = useState(false);
  const cache = useRef<Record<string, HTMLAudioElement>>({});

  useEffect(() => {
    setMuted(readBool(MUTED_KEY));
    setMusicOn(readBool(MUSIC_KEY));
  }, []);

  const play = useCallback((name: Cue) => {
    if (muted || typeof Audio === "undefined") return;
    try {
      let el = cache.current[name];
      if (!el) {
        el = new Audio(`/sounds/${name}.mp3`);
        cache.current[name] = el;
      }
      el.currentTime = 0;
      void el.play().catch(() => {});
    } catch {
      /* audio unavailable — ignore */
    }
  }, [muted]);

  const toggleMuted = useCallback(() => {
    setMuted((m) => {
      const next = !m;
      if (typeof localStorage !== "undefined") localStorage.setItem(MUTED_KEY, String(next));
      return next;
    });
  }, []);

  return { muted, musicOn, toggleMuted, play };
}
