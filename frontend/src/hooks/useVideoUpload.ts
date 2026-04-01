import { useCallback, useState } from "react";
import { useDropzone, type DropzoneOptions } from "react-dropzone";

const BROWSER_PLAYABLE = new Set(["video/mp4", "video/webm", "video/ogg"]);

const ACCEPT: DropzoneOptions["accept"] = {
  "video/*": [".mp4", ".mpg", ".mpeg", ".avi", ".mov", ".webm"],
};

export function useVideoUpload(onFileSelected: (file: File) => void) {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [canPlay, setCanPlay] = useState(false);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const f = acceptedFiles[0];
      if (!f) return;
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setCanPlay(BROWSER_PLAYABLE.has(f.type));
      onFileSelected(f);
    },
    [onFileSelected],
  );

  const remove = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setCanPlay(false);
  }, [preview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxFiles: 1,
    multiple: false,
  });

  return { file, preview, canPlay, isDragActive, getRootProps, getInputProps, remove };
}

export function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
