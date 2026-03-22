"use client";

import { useCallback, useRef, useState } from "react";
import type {
  ApprovalRequest,
  FileNode,
  LogEntry,
  PipelineEvent,
  Stage,
  StageState,
} from "@/lib/types";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const WS_BASE = API.replace(/^http/, "ws");

let logIdCounter = 0;

const INITIAL_STAGES: StageState[] = [
  { stage: "read", status: "pending" },
  { stage: "plan", status: "pending" },
  { stage: "code", status: "pending" },
  { stage: "review", status: "pending" },
];

export function usePipeline() {
  const [stages, setStages] = useState<StageState[]>(INITIAL_STAGES);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [approval, setApproval] = useState<ApprovalRequest | null>(null);
  const [fileTree, setFileTree] = useState<FileNode[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [costUsd, setCostUsd] = useState(0);
  const [totalTokens, setTotalTokens] = useState(0);
  const [durationS, setDurationS] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);

  const addLog = useCallback(
    (type: LogEntry["type"], message: string, timestamp?: string) => {
      setLogs((prev) => [
        ...prev,
        {
          id: `log_${++logIdCounter}`,
          timestamp: timestamp || new Date().toLocaleTimeString("en-US", { hour12: false }),
          type,
          message,
        },
      ]);
    },
    []
  );

  const handleEvent = useCallback(
    (event: PipelineEvent) => {
      switch (event.type) {
        case "init":
          addLog("info", "Session initialized", event.timestamp);
          for (const s of event.mcp_servers) {
            addLog(
              s.status === "connected" ? "info" : "error",
              `MCP '${s.name}': ${s.status}`,
              event.timestamp
            );
          }
          break;

        case "stage_change":
          setStages((prev) =>
            prev.map((s) =>
              s.stage === event.stage ? { ...s, status: event.status } : s
            )
          );
          addLog("info", `Stage ${event.stage}: ${event.status}`, event.timestamp);
          break;

        case "agent_spawn":
          addLog("agent", `Spawning ${event.agent}: ${event.description}`, event.timestamp);
          // Infer stage from agent name
          if (event.agent.includes("reader")) {
            setStages((prev) =>
              prev.map((s) => (s.stage === "read" ? { ...s, status: "running" } : s))
            );
          } else if (event.agent === "planner") {
            setStages((prev) =>
              prev.map((s) => {
                if (s.stage === "read") return { ...s, status: "done" };
                if (s.stage === "plan") return { ...s, status: "running" };
                return s;
              })
            );
          } else if (event.agent === "coder") {
            setStages((prev) =>
              prev.map((s) => {
                if (s.stage === "plan") return { ...s, status: "done" };
                if (s.stage === "code") return { ...s, status: "running" };
                return s;
              })
            );
          } else if (event.agent === "reviewer") {
            setStages((prev) =>
              prev.map((s) => {
                if (s.stage === "code") return { ...s, status: "done" };
                if (s.stage === "review") return { ...s, status: "running" };
                return s;
              })
            );
          }
          break;

        case "tool_call":
          addLog("tool", `${event.tool}`, event.timestamp);
          break;

        case "tool_result":
          addLog(
            event.is_error ? "error" : "result",
            event.preview.slice(0, 200),
            event.timestamp
          );
          break;

        case "task_started":
          addLog("info", `Task: ${event.description}`, event.timestamp);
          break;

        case "task_progress":
          setTotalTokens(event.tokens);
          break;

        case "task_complete":
          addLog(
            event.status === "completed" ? "info" : "error",
            `Task ${event.status}: ${event.summary.slice(0, 150)}`,
            event.timestamp
          );
          break;

        case "approval_needed":
          setApproval({
            id: event.id,
            question: event.question,
            header: event.header,
            options: event.options,
          });
          addLog("warn", `Approval needed: ${event.question.slice(0, 100)}`, event.timestamp);
          break;

        case "log":
          addLog(
            event.level === "error" ? "error" : event.level === "warn" ? "warn" : "info",
            event.message,
            event.timestamp
          );
          break;

        case "file_created":
          addLog("info", `File: ${event.path}`, event.timestamp);
          refreshFileTree();
          break;

        case "complete":
          setCostUsd(event.cost_usd);
          setDurationS(event.duration_s);
          setIsRunning(false);
          setStages((prev) =>
            prev.map((s) => ({
              ...s,
              status: s.status === "running" ? "done" : s.status,
            }))
          );
          addLog(
            event.is_error ? "error" : "info",
            `Pipeline complete — ${event.duration_s}s, $${event.cost_usd}, ${event.turns} turns`,
            event.timestamp
          );
          break;

        case "error":
          addLog("error", event.message, event.timestamp);
          break;
      }
    },
    [addLog]
  );

  const refreshFileTree = useCallback(async () => {
    if (!projectId) return;
    try {
      const res = await fetch(`${API}/api/projects/${projectId}/files`);
      const data = await res.json();
      setFileTree(data.tree || []);
    } catch {
      // ignore
    }
  }, [projectId]);

  const startPipeline = useCallback(
    async (objective: string) => {
      setIsRunning(true);
      setLogs([]);
      setStages(INITIAL_STAGES.map((s) => ({ ...s, status: "pending" })));
      setCostUsd(0);
      setTotalTokens(0);
      setDurationS(0);
      setFileTree([]);

      try {
        const res = await fetch(`${API}/api/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ objective }),
        });
        const data = await res.json();

        if (data.error) {
          addLog("error", data.error);
          setIsRunning(false);
          return;
        }

        const pid = data.project_id;
        setProjectId(pid);
        addLog("info", `Project: ${pid}`);
        addLog("info", `Mode: ${data.mode}, Papers: ${data.papers.length}`);

        // Connect WebSocket
        const ws = new WebSocket(`${WS_BASE}/ws/${pid}`);
        wsRef.current = ws;

        ws.onmessage = (e) => {
          try {
            const event: PipelineEvent = JSON.parse(e.data);
            handleEvent(event);
          } catch {
            // ignore
          }
        };

        ws.onclose = () => {
          setIsRunning(false);
        };

        ws.onerror = () => {
          addLog("error", "WebSocket connection error");
          setIsRunning(false);
        };
      } catch (err) {
        addLog("error", `Failed to start: ${err}`);
        setIsRunning(false);
      }
    },
    [addLog, handleEvent]
  );

  const sendApproval = useCallback((approvalId: string, answer: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: "approval_response",
          id: approvalId,
          answer,
        })
      );
    }
    setApproval(null);
  }, []);

  const loadFileContent = useCallback(
    async (path: string): Promise<string> => {
      if (!projectId) return "";
      try {
        const res = await fetch(
          `${API}/api/projects/${projectId}/file?path=${encodeURIComponent(path)}`
        );
        const data = await res.json();
        return data.content || "";
      } catch {
        return "Error loading file";
      }
    },
    [projectId]
  );

  return {
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
    refreshFileTree,
  };
}
