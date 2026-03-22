"use client";

import { cn } from "@/lib/utils";
import type { StageState } from "@/lib/types";
import { BookOpen, Code, FileSearch, CheckCircle2, Loader2, Circle, AlertCircle, ClipboardList } from "lucide-react";
import { Separator } from "@/components/ui/separator";

const STAGE_META: Record<string, { label: string; icon: React.ElementType }> = {
  read: { label: "Read Papers", icon: BookOpen },
  plan: { label: "Plan", icon: ClipboardList },
  code: { label: "Code", icon: Code },
  review: { label: "Review", icon: FileSearch },
};

function StatusIcon({ status }: { status: StageState["status"] }) {
  switch (status) {
    case "done":
      return <CheckCircle2 className="h-5 w-5 text-emerald-400" />;
    case "running":
      return <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />;
    case "error":
      return <AlertCircle className="h-5 w-5 text-red-400" />;
    default:
      return <Circle className="h-5 w-5 text-muted-foreground/40" />;
  }
}

export function PipelineStatus({
  stages,
  costUsd,
  totalTokens,
  durationS,
}: {
  stages: StageState[];
  costUsd: number;
  totalTokens: number;
  durationS: number;
}) {
  return (
    <div className="space-y-1">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
        Pipeline
      </h3>
      {stages.map((s, i) => {
        const meta = STAGE_META[s.stage];
        const Icon = meta?.icon || Circle;
        return (
          <div key={s.stage}>
            <div
              className={cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm",
                s.status === "running" && "bg-blue-500/10",
                s.status === "done" && "text-emerald-300",
                s.status === "error" && "text-red-400"
              )}
            >
              <StatusIcon status={s.status} />
              <Icon className="h-4 w-4" />
              <span className="font-medium">{meta?.label || s.stage}</span>
            </div>
            {i < stages.length - 1 && (
              <div className="ml-[22px] h-4 border-l border-muted-foreground/20" />
            )}
          </div>
        );
      })}

      <Separator className="my-4" />

      <div className="space-y-2 text-xs text-muted-foreground px-3">
        {durationS > 0 && (
          <div className="flex justify-between">
            <span>Duration</span>
            <span className="font-mono">{durationS.toFixed(1)}s</span>
          </div>
        )}
        {costUsd > 0 && (
          <div className="flex justify-between">
            <span>Cost</span>
            <span className="font-mono">${costUsd.toFixed(4)}</span>
          </div>
        )}
        {totalTokens > 0 && (
          <div className="flex justify-between">
            <span>Tokens</span>
            <span className="font-mono">{totalTokens.toLocaleString()}</span>
          </div>
        )}
      </div>
    </div>
  );
}
