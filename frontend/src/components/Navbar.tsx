import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">
        <Link href="/" className="text-lg font-bold tracking-tight text-foreground">
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
