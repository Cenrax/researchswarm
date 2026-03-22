"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Rocket } from "lucide-react";

export function ObjectiveInput({
  onRun,
  disabled,
}: {
  onRun: (objective: string) => void;
  disabled: boolean;
}) {
  const [objective, setObjective] = useState("");

  return (
    <Card className="border-border/50 bg-card">
      <CardContent className="pt-6">
        <label className="text-sm font-medium text-foreground mb-2 block">
          Objective
        </label>
        <div className="flex gap-2">
          <Input
            placeholder='e.g. "Build a transformer text classifier from this paper"'
            value={objective}
            onChange={(e) => setObjective(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && objective.trim() && !disabled) {
                onRun(objective.trim());
              }
            }}
            disabled={disabled}
            className="flex-1 bg-background"
          />
          <Button
            onClick={() => onRun(objective.trim())}
            disabled={!objective.trim() || disabled}
            className="gap-2"
          >
            <Rocket className="h-4 w-4" />
            {disabled ? "Running..." : "Run Pipeline"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
