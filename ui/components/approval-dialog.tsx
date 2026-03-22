"use client";

import { useState } from "react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import type { ApprovalRequest } from "@/lib/types";

export function ApprovalDialog({
  approval,
  onRespond,
}: {
  approval: ApprovalRequest | null;
  onRespond: (id: string, answer: string) => void;
}) {
  const [freeText, setFreeText] = useState("");

  if (!approval) return null;

  const hasOptions = approval.options.length > 0;

  return (
    <AlertDialog open={!!approval}>
      <AlertDialogContent className="max-w-lg">
        <AlertDialogHeader>
          <AlertDialogTitle>
            {approval.header || "Agent needs your input"}
          </AlertDialogTitle>
          <AlertDialogDescription className="whitespace-pre-wrap text-sm">
            {approval.question}
          </AlertDialogDescription>
        </AlertDialogHeader>

        {hasOptions ? (
          <div className="flex flex-col gap-2 my-2">
            {approval.options.map((opt, i) => {
              const label =
                typeof opt === "string"
                  ? opt
                  : opt.label || opt.value || String(opt);
              const desc =
                typeof opt === "object" ? opt.description : undefined;
              const value =
                typeof opt === "string"
                  ? opt
                  : opt.value || opt.label || String(opt);
              return (
                <Button
                  key={i}
                  variant="outline"
                  className="justify-start h-auto py-3 px-4 text-left"
                  onClick={() => onRespond(approval.id, value)}
                >
                  <div>
                    <div className="font-medium">{label}</div>
                    {desc && (
                      <div className="text-xs text-muted-foreground mt-0.5">
                        {desc}
                      </div>
                    )}
                  </div>
                </Button>
              );
            })}
          </div>
        ) : null}

        {/* Always show free text input */}
        <div className="flex gap-2 mt-2">
          <Input
            placeholder="Type your answer..."
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && freeText.trim()) {
                onRespond(approval.id, freeText.trim());
                setFreeText("");
              }
            }}
            className="flex-1"
          />
          <Button
            onClick={() => {
              if (freeText.trim()) {
                onRespond(approval.id, freeText.trim());
                setFreeText("");
              }
            }}
            disabled={!freeText.trim()}
          >
            Send
          </Button>
        </div>

        <AlertDialogFooter>
          <AlertDialogCancel
            onClick={() => onRespond(approval.id, "skip")}
          >
            Skip
          </AlertDialogCancel>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
