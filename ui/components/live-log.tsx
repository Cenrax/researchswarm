"use client";

import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { LogEntry } from "@/lib/types";
import { cn } from "@/lib/utils";

const TYPE_STYLES: Record<LogEntry["type"], string> = {
  info: "text-muted-foreground",
  warn: "text-amber-400",
  error: "text-red-400",
  agent: "text-violet-400",
  tool: "text-blue-400",
  result: "text-emerald-400",
};

const TYPE_PREFIX: Record<LogEntry["type"], string> = {
  info: "",
  warn: "⚠ ",
  error: "✗ ",
  agent: "● ",
  tool: "→ ",
  result: "✓ ",
};

export function LiveLog({ logs }: { logs: LogEntry[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <ScrollArea className="h-[400px] rounded-md border border-border/50 bg-background p-4 font-mono text-xs">
      {logs.length === 0 && (
        <p className="text-muted-foreground/50 text-center py-8">
          Pipeline logs will appear here...
        </p>
      )}
      {logs.map((entry) => (
        <div key={entry.id} className="flex gap-2 py-0.5 leading-relaxed">
          <span className="text-muted-foreground/50 shrink-0 w-[70px]">
            {entry.timestamp}
          </span>
          <span className={cn("break-all", TYPE_STYLES[entry.type])}>
            {TYPE_PREFIX[entry.type]}
            {entry.message}
          </span>
        </div>
      ))}
      <div ref={bottomRef} />
    </ScrollArea>
  );
}
