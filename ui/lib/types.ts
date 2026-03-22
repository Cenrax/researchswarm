export type Stage = "read" | "plan" | "code" | "review" | "complete";
export type StageStatus = "pending" | "running" | "done" | "error";

export interface StageState {
  stage: Stage;
  status: StageStatus;
}

export interface LogEntry {
  id: string;
  timestamp: string;
  type: "info" | "warn" | "error" | "agent" | "tool" | "result";
  message: string;
  source?: string;
}

export interface FileNode {
  name: string;
  path: string;
  type: "file" | "directory";
  size?: number;
  children?: FileNode[];
}

export interface ApprovalRequest {
  id: string;
  question: string;
  header: string;
  options: { label: string; value?: string; description?: string }[];
}

export interface ProjectInfo {
  id: string;
  path: string;
  created: string;
  objective: string;
  stages: {
    summaries: number;
    plans: number;
    code: number;
    reviews: number;
  };
}

export type PipelineEvent =
  | { type: "init"; mcp_servers: { name: string; status: string }[]; timestamp: string }
  | { type: "stage_change"; stage: Stage; status: StageStatus; timestamp: string }
  | { type: "agent_spawn"; agent: string; description: string; timestamp: string }
  | { type: "tool_call"; tool: string; input: Record<string, string>; timestamp: string }
  | { type: "tool_result"; tool: string; preview: string; is_error: boolean; timestamp: string }
  | { type: "task_started"; description: string; timestamp: string }
  | { type: "task_progress"; description: string; tokens: number; last_tool: string; timestamp: string }
  | { type: "task_complete"; status: string; summary: string; timestamp: string }
  | { type: "approval_needed"; id: string; question: string; header: string; options: any[]; timestamp: string }
  | { type: "log"; level: "info" | "warn" | "error"; message: string; source?: string; timestamp: string }
  | { type: "complete"; duration_s: number; cost_usd: number; turns: number; is_error: boolean; result: string; timestamp: string }
  | { type: "file_created"; path: string; timestamp: string }
  | { type: "error"; message: string; timestamp: string };
