"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { FileText, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function FilePreview({
  path,
  loadContent,
  onClose,
}: {
  path: string | null;
  loadContent: (path: string) => Promise<string>;
  onClose: () => void;
}) {
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!path) {
      setContent("");
      return;
    }
    setLoading(true);
    loadContent(path).then((c) => {
      setContent(c);
      setLoading(false);
    });
  }, [path, loadContent]);

  if (!path) return null;

  const fileName = path.split("/").pop() || path;
  const isMarkdown = fileName.endsWith(".md");

  return (
    <Card className="border-border/50 bg-card">
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-sm font-mono">{path}</CardTitle>
          <Badge variant="secondary" className="text-[10px]">
            {isMarkdown ? "Markdown" : fileName.split(".").pop()?.toUpperCase()}
          </Badge>
        </div>
        <button
          onClick={onClose}
          className="rounded-full p-1 hover:bg-muted"
        >
          <X className="h-4 w-4" />
        </button>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[350px] rounded-md border border-border/30 bg-background p-4">
          {loading ? (
            <div className="text-muted-foreground/50 text-center py-8">
              Loading...
            </div>
          ) : isMarkdown ? (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {content}
              </ReactMarkdown>
            </div>
          ) : (
            <pre className="text-xs font-mono whitespace-pre-wrap">{content}</pre>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
