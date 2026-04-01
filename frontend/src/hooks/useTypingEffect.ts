import { useEffect, useState } from "react";

export function useTypingEffect(text: string, speed: number = 30) {
  const [displayed, setDisplayed] = useState("");

  useEffect(() => {
    if (!text) {
      setDisplayed("");
      return;
    }

    setDisplayed("");
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, speed);

    return () => clearInterval(interval);
  }, [text, speed]);

  const isTyping = text.length > 0 && displayed.length < text.length;

  return { displayed, isTyping };
}
