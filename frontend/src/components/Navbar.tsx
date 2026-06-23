import Link from "next/link";
import Image from "next/image";

export default function Navbar() {
  return (
    <nav className="border-b border-border bg-card shadow-sm">
      <div className="mx-auto flex h-14 max-w-5xl items-center justify-between px-6">
        <Link
          href="/"
          className="flex items-center gap-2 text-sm font-semibold text-foreground transition-opacity hover:opacity-70"
        >
          <Image src="/logo.png" alt="ImConvo" width={80} height={40} className="rounded" />
          ImConvo
        </Link>
        <div className="flex items-center gap-6 text-sm text-muted">
          <Link href="/demo/inference" className="transition-colors hover:text-foreground">
            Demo
          </Link>
          <Link href="/demo/sandbox" className="transition-colors hover:text-foreground font-medium text-accent">
            GRID-Bot Sandbox
          </Link>
          <a
            href="https://github.com/somerandomguy-coder/ImConvo/"
            target="_blank"
            rel="noopener noreferrer"
            className="transition-colors hover:text-foreground"
          >
            GitHub
          </a>
        </div>
      </div>
    </nav>
  );
}
