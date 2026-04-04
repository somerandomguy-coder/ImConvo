import Link from "next/link";
import Image from "next/image";

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
        <Link href="/" className="flex h-full items-center gap-3 text-lg font-bold tracking-tight text-foreground">
          <Image src="/logo.png" alt="ImConvo" width={64} height={64} className="h-full w-auto py-1" />
          ImConvo
        </Link>
        <a
          href="https://github.com/somerandomguy-coder/ImConvo/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-muted transition-colors hover:text-foreground"
        >
          GitHub
        </a>
      </div>
    </nav>
  );
}
