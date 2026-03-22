"use client";

import { useState } from "react";
import { usePipeline } from "@/hooks/use-pipeline";
import { PipelineStatus } from "@/components/pipeline-status";
import { ObjectiveInput } from "@/components/objective-input";
import { PaperUpload } from "@/components/paper-upload";
import { LiveLog } from "@/components/live-log";
import { ApprovalDialog } from "@/components/approval-dialog";
import { FileTree } from "@/components/file-tree";
import { FilePreview } from "@/components/file-preview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Hexagon } from "lucide-react";

export default function Home() {
  const {
    stages,
    logs,
    approval,
    fileTree,
    isRunning,
    projectId,
    costUsd,
    totalTokens,
    durationS,
    startPipeline,
    sendApproval,
    loadFileContent,
  } = usePipeline();

  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-border/50">
        <div className="flex items-center gap-2">
          <Hexagon className="h-5 w-5 text-violet-400" />
          <h1 className="text-lg font-semibold tracking-tight">
            Research Swarm
          </h1>
          {isRunning && (
            <Badge variant="secondary" className="text-[10px] animate-pulse">
              Running
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          {costUsd > 0 && (
            <span className="font-mono">${costUsd.toFixed(4)}</span>
          )}
          {projectId && (
            <span className="font-mono text-muted-foreground/50 truncate max-w-[200px]">
              {projectId}
            </span>
          )}
        </div>
      </header>

      {/* Main content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        <aside className="w-64 border-r border-border/50 flex flex-col overflow-hidden">
          <div className="p-4 flex-1 overflow-y-auto">
            <PipelineStatus
              stages={stages}
              costUsd={costUsd}
              totalTokens={totalTokens}
              durationS={durationS}
            />

            <Separator className="my-4" />

            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
              Output Files
            </h3>
            <FileTree
              tree={fileTree}
              onSelect={(path) => {
                setSelectedFile(path);
              }}
            />
          </div>
        </aside>

        {/* Main area */}
        <main className="flex-1 overflow-y-auto p-6 space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <ObjectiveInput onRun={startPipeline} disabled={isRunning} />
            <PaperUpload />
          </div>

          <Tabs defaultValue="log" className="w-full">
            <TabsList className="bg-muted/50">
              <TabsTrigger value="log">Live Log</TabsTrigger>
              <TabsTrigger value="preview" disabled={!selectedFile}>
                File Preview
              </TabsTrigger>
            </TabsList>
            <TabsContent value="log" className="mt-3">
              <LiveLog logs={logs} />
            </TabsContent>
            <TabsContent value="preview" className="mt-3">
              <FilePreview
                path={selectedFile}
                loadContent={loadFileContent}
                onClose={() => setSelectedFile(null)}
              />
            </TabsContent>
          </Tabs>
        </main>
      </div>

      {/* Approval dialog (modal overlay) */}
      <ApprovalDialog approval={approval} onRespond={sendApproval} />
    </div>
  );
}
