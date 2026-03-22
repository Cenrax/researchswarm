"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { FileNode } from "@/lib/types";
import { ChevronRight, File, Folder, FolderOpen } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

function TreeItem({
  node,
  depth,
  onSelect,
}: {
  node: FileNode;
  depth: number;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(depth < 2);

  if (node.type === "directory") {
    return (
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex items-center gap-1.5 w-full py-1 px-2 rounded-sm hover:bg-muted/50 text-sm">
          <ChevronRight
            className={cn(
              "h-3.5 w-3.5 text-muted-foreground transition-transform",
              open && "rotate-90"
            )}
          />
          {open ? (
            <FolderOpen className="h-4 w-4 text-blue-400" />
          ) : (
            <Folder className="h-4 w-4 text-blue-400" />
          )}
          <span className="truncate">{node.name}</span>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="ml-4 border-l border-border/30 pl-1">
            {node.children?.map((child) => (
              <TreeItem
                key={child.path}
                node={child}
                depth={depth + 1}
                onSelect={onSelect}
              />
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>
    );
  }

  return (
    <button
      onClick={() => onSelect(node.path)}
      className="flex items-center gap-1.5 w-full py-1 px-2 rounded-sm hover:bg-muted/50 text-sm text-left"
    >
      <span className="w-3.5" />
      <File className="h-4 w-4 text-muted-foreground" />
      <span className="truncate">{node.name}</span>
      {node.size !== undefined && (
        <span className="ml-auto text-[10px] text-muted-foreground/50 font-mono">
          {node.size > 1024
            ? `${(node.size / 1024).toFixed(1)}k`
            : `${node.size}b`}
        </span>
      )}
    </button>
  );
}

export function FileTree({
  tree,
  onSelect,
}: {
  tree: FileNode[];
  onSelect: (path: string) => void;
}) {
  if (tree.length === 0) {
    return (
      <div className="text-xs text-muted-foreground/50 text-center py-4">
        Files will appear here as the pipeline runs...
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      {tree.map((node) => (
        <TreeItem key={node.path} node={node} depth={0} onSelect={onSelect} />
      ))}
    </div>
  );
}
